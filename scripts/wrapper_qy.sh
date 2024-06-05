#!/bin/bash

CLUSTER_NAME=qy
PARTITION=gpu3-2-low
# PARTITION=gpu4-low
HOST=None
NNODES=1
GPUS_PER_NODE=4
# GPUS_PER_NODE=8

export MASTER_PORT=$((RANDOM % 12000 + 10000))

MEM_PER_CPU=256G
# --mem-per-cpu $MEM_PER_CPU \
MEM_PER_NODE=256G

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$GPUS_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

export TRACE_NAME=${CLUSTER_NAME}
TB_DIR=./tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

# LOGGING_ARGS="${LOGGING_ARGS} \
# --profiler-with-tensorboard \
# --tb-dir $TB_DIR \
# "

# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET

# Run with MPI
GPU_NUM=16
HOST_CONFIG="g3021:8,g3022:8"
# GPU_NUM=8
# HOST_CONFIG="g3021:8"
# GPU_NUM=4
# HOST_CONFIG="g3027:2,g4003:2"
# HOST_CONFIG="g3021:2,g3022:2"
export MASTER_ADDR=$(echo ${HOST_CONFIG} | awk -F: '{print $1}')
RUNNER_CMD="mpirun --prefix $(dirname `which mpirun`)/../ \
    -x MASTER_ADDR -x MASTER_PORT \
    -x LD_LIBRARY_PATH -x PATH \
    -x TRACE_NAME \
    -x NCCL_DEBUG \
    -x NCCL_NET_GDR_LEVEL \
    -x NCCL_DEBUG_SUBSYS \
    -x NCCL_IB_DISABLE \
    -np $GPU_NUM --host $HOST_CONFIG"


# # Run with Slurm
# RUNNER_CMD="srun $SLURM_ARGS"

set -x
export CUDA_LAUNCH_BLOCKING=1 # for debugging
export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: important for cc overlap !!!
$RUNNER_CMD \
python bench_ring_attn.py \
    $LOGGING_ARGS \

set +x
