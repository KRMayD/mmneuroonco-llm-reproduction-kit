import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def encode_atomic_text_batch(model, token_batch, normalize=True):
    if token_batch.dim() == 2:
        return model.encode_text(token_batch, normalize=normalize)
    if token_batch.dim() == 3:
        batch_size, num_attrs, ctx_len = token_batch.shape
        flat_tokens = token_batch.reshape(batch_size * num_attrs, ctx_len)
        flat_features = model.encode_text(flat_tokens, normalize=normalize)
        return flat_features.reshape(batch_size, num_attrs, -1)
    raise ValueError(f"Unsupported atomic token shape: {token_batch.shape}")


def encode_atomic_image_batch(model, image_batch, chunk_size=None, normalize=True):
    if image_batch is None:
        return None
    if image_batch.dim() == 4:
        return model.encode_image(image_batch, normalize=normalize)
    if image_batch.dim() == 5:
        batch_size, num_attrs = image_batch.shape[:2]
        flat_images = image_batch.reshape(batch_size * num_attrs, *image_batch.shape[2:])
        if chunk_size is None or chunk_size <= 0:
            chunk_size = flat_images.shape[0]

        flat_features = []
        for start_idx in range(0, flat_images.shape[0], chunk_size):
            end_idx = start_idx + chunk_size
            flat_features.append(
                model.encode_image(flat_images[start_idx:end_idx], normalize=normalize)
            )
        return torch.cat(flat_features, dim=0).reshape(batch_size, num_attrs, -1)
    raise ValueError(f"Unsupported atomic image shape: {image_batch.shape}")


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()
    
    # Initialize Reference Model for DPO
    ref_model = None
    if args.dpo_loss:
        import copy
        from open_clip import create_model, load_checkpoint
        
        print("Loading Reference Model for DPO...")
        # Create a new model instance
        ref_model = create_model(
            args.model,
            pretrained=None, # Load manually
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            force_image_size=args.force_image_size,
            pretrained_image=args.pretrained_image,
        )
        
        # Load specific checkpoint
        ref_checkpoint = args.dpo_ref_checkpoint
        print(f"Loading reference weights from {ref_checkpoint}")
        load_checkpoint(ref_model, ref_checkpoint, weights_only=False)
        
        ref_model.eval()
        ref_model.to(device)
        print("Reference Model initialized.")

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.dpo_loss and ref_model is not None:
            if args.dpo_mode == "atomic":
                atomic_attr_idx = None
                atomic_text_weight = None
                atomic_neg_images = None
                atomic_visual_neg_found_mask = None
                atomic_visual_neg_mismatch_count = None
                if args.atomic_train_mode == "prop_sample":
                    (
                        images,
                        images_neg,
                        texts,
                        texts_neg,
                        atomic_texts_pos,
                        atomic_texts_neg,
                        atomic_attr_idx,
                        atomic_mask,
                    ) = batch
                    atomic_attr_idx = atomic_attr_idx.to(device=device, non_blocking=True)
                else:
                    (
                        images,
                        images_neg,
                        texts,
                        texts_neg,
                        atomic_texts_pos,
                        atomic_texts_neg,
                        atomic_mask,
                    ) = batch[:7]
                    batch_offset = 7
                    if args.atomic_text_weight_mode == "semantic_ref_cache":
                        atomic_text_weight = batch[batch_offset]
                        atomic_text_weight = atomic_text_weight.to(device=device, non_blocking=True)
                        batch_offset += 1
                    if getattr(args, "atomic_visual_neg_source", "global_neg") == "atomic_matched":
                        (
                            atomic_neg_images,
                            atomic_visual_neg_found_mask,
                            atomic_visual_neg_mismatch_count,
                        ) = batch[batch_offset:batch_offset + 3]
                atomic_texts_pos = atomic_texts_pos.to(device=device, non_blocking=True)
                atomic_texts_neg = atomic_texts_neg.to(device=device, non_blocking=True)
                atomic_mask = atomic_mask.to(device=device, non_blocking=True)
                if atomic_neg_images is not None:
                    atomic_neg_images = atomic_neg_images.to(
                        device=device,
                        dtype=input_dtype,
                        non_blocking=True,
                    )
                    atomic_visual_neg_found_mask = atomic_visual_neg_found_mask.to(
                        device=device,
                        non_blocking=True,
                    )
                    atomic_visual_neg_mismatch_count = atomic_visual_neg_mismatch_count.to(
                        device=device,
                        non_blocking=True,
                    )
            else:
                images, images_neg, texts, texts_neg = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            images_neg = images_neg.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            texts_neg = texts_neg.to(device=device, non_blocking=True)
        else:
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                if args.dpo_loss and ref_model is not None:
                    logit_scale = unwrap_model(model).logit_scale.exp()
                    normalize_features = not args.disable_feature_normalization

                    if args.dpo_mode == "atomic":
                        img_feat_pos = model.encode_image(images, normalize=normalize_features)
                        img_feat_neg = model.encode_image(images_neg, normalize=normalize_features)
                        txt_feat_pos = model.encode_text(texts.clone(), normalize=normalize_features)
                        txt_feat_neg = model.encode_text(texts_neg.clone(), normalize=normalize_features)
                        atomic_txt_feat_pos = encode_atomic_text_batch(
                            model, atomic_texts_pos.clone(), normalize=normalize_features
                        )
                        atomic_txt_feat_neg = encode_atomic_text_batch(
                            model, atomic_texts_neg.clone(), normalize=normalize_features
                        )
                        atomic_img_feat_neg = encode_atomic_image_batch(
                            model,
                            atomic_neg_images,
                            chunk_size=images.shape[0],
                            normalize=normalize_features,
                        )

                        with torch.no_grad():
                            ref_img_pos = ref_model.encode_image(images, normalize=normalize_features)
                            ref_img_neg = ref_model.encode_image(images_neg, normalize=normalize_features)
                            ref_txt_pos = ref_model.encode_text(texts.clone(), normalize=normalize_features)
                            ref_txt_neg = ref_model.encode_text(texts_neg.clone(), normalize=normalize_features)
                            ref_atomic_txt_feat_pos = encode_atomic_text_batch(
                                ref_model, atomic_texts_pos.clone(), normalize=normalize_features
                            )
                            ref_atomic_txt_feat_neg = encode_atomic_text_batch(
                                ref_model, atomic_texts_neg.clone(), normalize=normalize_features
                            )
                            ref_atomic_img_feat_neg = encode_atomic_image_batch(
                                ref_model,
                                atomic_neg_images,
                                chunk_size=images.shape[0],
                                normalize=normalize_features,
                            )

                        losses = loss(
                            image_features_pos=img_feat_pos,
                            image_features_neg=img_feat_neg,
                            text_features_pos=txt_feat_pos,
                            text_features_neg=txt_feat_neg,
                            ref_image_features_pos=ref_img_pos,
                            ref_image_features_neg=ref_img_neg,
                            ref_text_features_pos=ref_txt_pos,
                            ref_text_features_neg=ref_txt_neg,
                            atomic_text_features_pos=atomic_txt_feat_pos,
                            atomic_text_features_neg=atomic_txt_feat_neg,
                            ref_atomic_text_features_pos=ref_atomic_txt_feat_pos,
                            ref_atomic_text_features_neg=ref_atomic_txt_feat_neg,
                            atomic_mask=atomic_mask,
                            atomic_text_weight=atomic_text_weight,
                            atomic_attr_idx=atomic_attr_idx,
                            atomic_image_features_neg=atomic_img_feat_neg,
                            ref_atomic_image_features_neg=ref_atomic_img_feat_neg,
                            atomic_visual_neg_found_mask=atomic_visual_neg_found_mask,
                            atomic_visual_neg_mismatch_count=atomic_visual_neg_mismatch_count,
                            output_dict=True,
                        )
                    else:
                        if normalize_features:
                            out_pos = model(images, texts.clone())
                            img_feat_pos = out_pos["image_features"]
                            txt_feat_pos = out_pos["text_features"]
                            logit_scale = out_pos["logit_scale"]

                            out_neg = model(images_neg, texts_neg.clone())
                            img_feat_neg = out_neg["image_features"]
                            txt_feat_neg = out_neg["text_features"]

                            with torch.no_grad():
                                ref_out_pos_raw = ref_model(images, texts.clone())
                                ref_out_neg_raw = ref_model(images_neg, texts_neg.clone())

                                if isinstance(ref_out_pos_raw, dict):
                                    ref_img_pos = ref_out_pos_raw["image_features"]
                                    ref_txt_pos = ref_out_pos_raw["text_features"]
                                else:
                                    ref_img_pos = ref_out_pos_raw[0]
                                    ref_txt_pos = ref_out_pos_raw[1]

                                if isinstance(ref_out_neg_raw, dict):
                                    ref_img_neg = ref_out_neg_raw["image_features"]
                                    ref_txt_neg = ref_out_neg_raw["text_features"]
                                else:
                                    ref_img_neg = ref_out_neg_raw[0]
                                    ref_txt_neg = ref_out_neg_raw[1]
                        else:
                            img_feat_pos = model.encode_image(images, normalize=False)
                            txt_feat_pos = model.encode_text(texts.clone(), normalize=False)
                            img_feat_neg = model.encode_image(images_neg, normalize=False)
                            txt_feat_neg = model.encode_text(texts_neg.clone(), normalize=False)

                            with torch.no_grad():
                                ref_img_pos = ref_model.encode_image(images, normalize=False)
                                ref_txt_pos = ref_model.encode_text(texts.clone(), normalize=False)
                                ref_img_neg = ref_model.encode_image(images_neg, normalize=False)
                                ref_txt_neg = ref_model.encode_text(texts_neg.clone(), normalize=False)

                        losses = loss(
                            image_features_pos=img_feat_pos,
                            text_features_pos=txt_feat_pos,
                            image_features_neg=img_feat_neg,
                            text_features_neg=txt_feat_neg,
                            ref_image_features_pos=ref_img_pos,
                            ref_text_features_pos=ref_txt_pos,
                            ref_image_features_neg=ref_img_neg,
                            ref_text_features_neg=ref_txt_neg,
                            output_dict=True,
                        )

                    total_loss = losses["dpo_loss"]

                elif args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out = model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                    losses = loss(**model_out, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                else:
                    model_out = model(images, texts)
                    logit_scale = model_out["logit_scale"]
                    losses = loss(**model_out, output_dict=True)
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
             # Accumulation not supported for DPO in this quick patch
             # defaulting to standard logic or raising error would be safer
            with torch.no_grad():

                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
