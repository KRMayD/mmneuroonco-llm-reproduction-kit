# Codex Handoff Prompt

아래 내용을 새 서버의 Codex에게 그대로 전달하십시오.

```text
현재 디렉터리는 MM-NeuroOnco Closed-VQA LLM 재현 kit이다. 먼저 README.md,
src/docs/MMNEURO_FIXED_DOWNSTREAM_PROTOCOL.md, artifacts/vision_checkpoints.tsv를
끝까지 읽어라.

목표는 새로 학습된 GMPO upstream vision encoder를 MM-NeuroOnco LLM에
연결하고, 나머지 downstream 조건을 완전히 동일하게 고정해 평가하는 것이다.
BiomedCLIP baseline, CLIP post-trained, CLIPrefine post-trained checkpoint는
고정 control이며, 이번에 바꾸는 유일한 upstream 모델은 GMPO다.

비교 encoder:
1. BiomedCLIP baseline
2. CLIP post-trained encoder
3. CLIPrefine post-trained encoder
4. 새로 학습된 GMPO candidate encoder

반드시 지켜야 할 조건:
- MM-NeuroOnco train variant는 open_closed_nocot만 사용한다.
- manifest를 다른 서버에서 복사하지 말고, 이 서버의 절대경로로 다시 생성한다.
- train_data.json=163267, closed_test_data.json=3190인지 확인한다.
- vision encoder는 네 조건 모두 frozen이다.
- baseline/CLIP/CLIPrefine checkpoint를 재학습하거나 교체하지 않는다.
- GMPO_CKPT에는 사용자가 새로 학습해 전달한 GMPO checkpoint만 넣는다.
- 새 GMPO checkpoint가 OpenCLIP/BiomedCLIP 호환 state dict인지 확인한다.
- Q-Former 초기화는 dmis-lab/biobert-v1.1이다.
- 생성 LLM은 Qwen/Qwen3-0.6B이다.
- visual L2 normalization은 반드시 끈다.
- batch size=8/GPU, epoch=3, LR=1e-4, min LR=5e-6, warmup=200,
  weight decay=0.05, BF16 mixed precision, seed=0, validation skip으로 고정한다.
- GPU 0,1,2,3에서 네 독립 job을 동시에 실행한다. 한 모델에 4-GPU DDP를
  적용하는 것이 아니다.
- 평가는 generated-answer parsing이 아니라 Closed-VQA의 선택지 A-E에 대한
  MC likelihood로 수행한다.
- src/scripts/run_mmneuro_fixed_protocol_4encoders.sh만 사용한다. legacy
  run_brain_mri_*.sh를 수정하거나 사용하지 않는다.
- checkpoint와 data가 없으면 임의 모델로 대체하지 말고, 필요한 파일 및
  SHA256을 명시해서 사용자에게 요청한다.
- 시스템 CUDA, cuDNN, NVIDIA driver, 기존 PyTorch CUDA build를 변경하지 않는다.

수행 순서:
1. bash bootstrap_mmneuro_assets.sh /data/mmneuro_assets 를 실행해 공개
   MM-NeuroOnco 데이터와 BioBERT/Qwen snapshot을 받는다.
2. Dataset.zip 및 Benchmark_Images.zip이 해제되었는지 확인한다.
3. prepare_mmneuro_official_vqa.py로 /data/mmneuro_assets/mm_neuroonco_official
   manifest를 재생성한다.
4. 고정 control 세 checkpoint와 새 GMPO checkpoint를
   artifacts/vision_checkpoints.tsv의 SHA256으로 검증한다. 새 GMPO 행의
   placeholder SHA256이 남아 있으면 사용자에게 실제 SHA256을 요청하고 실행하지 않는다.
5. src/configs/mmneuro_fixed_protocol.env.example을 복사해 현재 서버의
   절대경로만 채운 config를 만든다. REPO_ROOT는 이 kit의 src 디렉터리,
   OPENCLIP_SRC는 이 kit의 third_party/open_clip_src가 되어야 한다.
6. bash src/scripts/run_mmneuro_fixed_protocol_4encoders.sh <config>를 실행한다.
7. RUN_ROOT의 protocol.json, 모델별 *.summary.json, metrics_comparison.json,
   logs, 사용한 LLM checkpoint 경로를 검토하고, 새 GMPO와 세 고정 control의
   결과를 표로 보고한다.

코드 구조나 학습 알고리즘을 변경하지 말고, 경로 및 서버 환경에 필요한 최소한의
설정만 수정하라.
```
