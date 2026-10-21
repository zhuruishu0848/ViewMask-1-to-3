#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n ${ENV_FILE:-} ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
exec "${PYTHON_BIN:-python}" -u -m eval.evaluate_gso_checkpoint \
    --model_path "${FINAL_CHECKPOINT:-./checkpoints/viewmask-final}" \
    --tokenizer_path "${INFERENCE_TOKENIZER:-./checkpoints/viewmask-tokenizer}" \
    --magvit_path "${MAGVIT_PATH:-./checkpoints/magvitv2}" \
    --json_path "${GSO_TEST_JSON:-./data/GSO/NVG_dataset/data_split/test_samples.json}" \
    --gso100_csv "${GSO100_CSV:-./data/GSO/gso_metadata_object_prompt_100.csv}" \
    --inception_weights "${INCEPTION_WEIGHTS:-./checkpoints/inception_v3_google-0cc3c7bd.pth}" \
    --output_dir "${GSO_OUTPUT:-./outputs/gso_evaluation}" \
    --num_steps 50 --temperature 0.8 --max_length 1536 --batch_size 4 "$@"
