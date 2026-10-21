#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n ${ENV_FILE:-} ]]; then set -a; source "$ENV_FILE"; set +a; fi
exec bash scripts/launch_train.sh scripts.train_stage3_t2mv \
    --load_from_checkpoint "${STAGE2_CHECKPOINT:-./outputs/stage2_i2mv}" \
    --output_dir "${STAGE3_OUTPUT:-./outputs/stage3_t2mv}" \
    --per_device_train_batch_size 8 --learning_rate 1e-5 \
    --num_train_epochs 10 --save_steps 1000 --warmup_steps 1000 \
    --logging_steps 10 --gradient_accumulation_steps "${GRAD_ACCUM:-2}" "$@"
