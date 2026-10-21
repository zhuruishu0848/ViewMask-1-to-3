#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n ${ENV_FILE:-} ]]; then set -a; source "$ENV_FILE"; set +a; fi
exec bash scripts/launch_train.sh scripts.train_stage2_i2mv \
    --load_from_checkpoint "${STAGE1_CHECKPOINT:-./outputs/stage1_pretrain}" \
    --output_dir "${STAGE2_OUTPUT:-./outputs/stage2_i2mv}" \
    --per_device_train_batch_size 4 --learning_rate 3.5e-5 \
    --num_train_epochs 100 --save_steps 1000 --warmup_steps 1800 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" "$@"
