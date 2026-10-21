#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n ${ENV_FILE:-} ]]; then set -a; source "$ENV_FILE"; set +a; fi
exec bash scripts/launch_train.sh scripts.train_stage1_pretrain \
    --output_dir "${STAGE1_OUTPUT:-./outputs/stage1_pretrain}" \
    --per_device_train_batch_size 4 --learning_rate 4e-5 \
    --num_train_epochs 100 --save_steps 3000 --warmup_steps 8000 \
    --gradient_accumulation_steps "${GRAD_ACCUM:-4}" "$@"
