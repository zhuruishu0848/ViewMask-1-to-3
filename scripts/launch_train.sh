#!/usr/bin/env bash
# Shared Accelerate/DeepSpeed launcher. Called by the three stage scripts.
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
TRAIN_MODULE=${1:?Training module required}
shift
if [[ -n ${CONDA_ENV:-} ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-20}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NUM_MACHINES=${NUM_MACHINES:-1}
NUM_PROCESSES=${NUM_PROCESSES:-1}
RANK=${FORCE_RANK:-${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-0}}
if [[ $NUM_MACHINES == 1 ]]; then
    RANK=${FORCE_RANK:-0}
fi
MASTER_ADDR=${MASTER_ADDR:-${GEMINI_HOST_IP_taskrole1_0:-}}
if [[ $NUM_MACHINES == 1 ]]; then
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi
MASTER_PORT=${MASTER_PORT:-21456}
if [[ -z $MASTER_ADDR ]]; then
    echo 'Set MASTER_ADDR or GEMINI_HOST_IP_taskrole1_0 for multi-node training' >&2
    exit 1
fi
if (( NUM_MACHINES < 1 || NUM_PROCESSES < 1 || NUM_PROCESSES % NUM_MACHINES != 0 || RANK < 0 || RANK >= NUM_MACHINES )); then
    echo 'Invalid NUM_MACHINES, NUM_PROCESSES (total GPUs) or FORCE_RANK' >&2
    exit 1
fi
# Keep DeepSpeed and TrainingArguments in sync, including command-line overrides.
GRAD_ACCUM=${GRAD_ACCUM:-4}
EXPECT_ACCUM=0
for arg in "$@"; do
    if (( EXPECT_ACCUM )); then GRAD_ACCUM=$arg; EXPECT_ACCUM=0; fi
    if [[ $arg == --gradient_accumulation_steps ]]; then EXPECT_ACCUM=1; fi
    if [[ $arg == --gradient_accumulation_steps=* ]]; then GRAD_ACCUM=${arg#*=}; fi
done
CONFIG=$(mktemp "${TMPDIR:-.}/viewmask-accelerate.XXXXXX.yaml")
trap 'rm -f "$CONFIG"' EXIT
cat > "$CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_machines: $NUM_MACHINES
num_processes: $NUM_PROCESSES
machine_rank: $RANK
main_process_ip: $MASTER_ADDR
main_process_port: $MASTER_PORT
main_training_function: main
rdzv_backend: static
same_network: true
use_cpu: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: $GRAD_ACCUM
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
  zero3_save_16bit_model: true
EOF
CMD=(accelerate launch --config_file "$CONFIG" --deepspeed_multinode_launcher standard
    --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
    --machine_rank "$RANK" --num_machines "$NUM_MACHINES" --num_processes "$NUM_PROCESSES"
    --module "$TRAIN_MODULE" "$@")
if [[ ${DRY_RUN:-0} == 1 ]]; then
    cat "$CONFIG"
    printf '%q ' "${CMD[@]}"
    printf '\n'
else
    "${CMD[@]}"
fi
