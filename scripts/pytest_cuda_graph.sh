#!/bin/bash

CLUSTER_NAME=qy
# PARTITION=gpu3-2-low
NNODES=1
GPUS_PER_NODE=1
PARTITION=arch
# HOST="g3025"
PARTITION=rag
# HOST="g3017,g3018"
# HOST="g3017,g3022"
HOST="g3015,g3017"
HOST="g3015,g3018"
# GPU_NUMs="24"
HOST="g3015,g3018,g3021"
HOST="g3017"
PARTITION=hit
HOST="g4005"


HOST=None

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
--pty \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

export TRACE_NAME=pytest_cuda_graph
TB_DIR=./tb
mkdir -p $TB_DIR
LOGGING_ARGS=""

# LOGGING_ARGS="${LOGGING_ARGS} \
# --profiler-with-tensorboard \
# --tb-dir $TB_DIR \
# "

# Nsight System:
export USE_NSYS=True
export USE_NSYS=False
export NSYS_DIR=./prof_results/nsys
mkdir -p $NSYS_DIR
if [ $USE_NSYS == "True" ]; then
    NSIGHT_CMD="nsys profile --trace=cuda,nvtx,osrt,mpi,nvtx --output=${NSYS_DIR}/${TRACE_NAME}_$(date "+%Y%m%d-%H%M%S").qdrep"
    NSIGHT_CMD="nsys profile \
        --mpi-impl=openmpi \
        --cuda-graph-trace=node \
        -o ${NSYS_DIR}/${TRACE_NAME}_$(date "+%Y%m%d-%H%M%S")"
else
    NSIGHT_CMD=""
fi

# NCCL Args:
export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_LEVEL=5
# export NCCL_NET_GDR_LEVEL=0   # Disable GDR
export NCCL_IB_DISABLE=0
export NCCL_DEBUG_SUBSYS=NET

# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS"


set -x
# export TORCH_USE_CUDA_DSA=1 # use it in **compile-time** of pytorch for debugging
# export TORCH_SHOW_CPP_STACKTRACES=1 # for debugging
# export CUDA_LAUNCH_BLOCKING=1 # for debugging
# export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: important for cc overlap !!!
$RUNNER_CMD \
-c 16 \
$NSIGHT_CMD \
pytest ./tests/test_cuda_graph.py \
    -s \
    
set +x
exit 0

# Run with MPI
# salloc -p rag -w g3017 -N 1 -n 128 -t 3600
# salloc -p arch -w g3029 -N 1 -n 128 -t 3600
# salloc -p rag -w g3013 -N 1 -n 128 -t 3600
# salloc -p hit -w g4008 -N 1 -n 128 -t 3600

GPU_NUM=16
HOST_CONFIG="g3021:8,g3022:8"
GPU_NUM=8
HOST_CONFIG="g3017:8"
GPU_NUM=4
HOST_CONFIG="g3017:4"
HOST_CONFIG="g4008:4"
GPU_NUM=1
HOST_CONFIG="g4005:1"
# HOST_CONFIG="g3029:4"
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
    --map-by ppr:4:numa --bind-to core --report-bindings \
    -np $GPU_NUM --host $HOST_CONFIG"
NSIGHT_CMD="nsys profile --mpi-impl=openmpi --cuda-graph-trace=node -o ${NSYS_DIR}/${TRACE_NAME}_$(date "+%Y%m%d-%H%M%S")"
# NSIGHT_CMD=""
set -x
# export CUDA_LAUNCH_BLOCKING=1 # for debugging
# export CUDA_DEVICE_MAX_CONNECTIONS=1    # [NOTE]: important for cc overlap !!!
$NSIGHT_CMD \
$RUNNER_CMD \
python ./tests/test_cuda_graph.py \

# pytest ./tests/test_cuda_graph.py \
#     -s \

set +x
