#!/bin/bash

CLUSTER_NAME=qy
# PARTITION=gpu3-2-low
NNODES=1
NTASKS_PER_NODE=1
PARTITION=arch
# HOST="g3025"
# PARTITION=rag
# HOST="g3017,g3018"
# HOST="g3017,g3022"
HOST="g3015,g3017"
HOST="g3015,g3018"
# GPU_NUMs="24"
HOST="g3015,g3018,g3021"
HOST="g3017"
# HOST="g3021"
# PARTITION=hit
# HOST="g4001"


HOST=None

export MASTER_PORT=$((RANDOM % 12000 + 10000))

MEM_PER_CPU=256G
# --mem-per-cpu $MEM_PER_CPU \
MEM_PER_NODE=256G

SLURM_ARGS="
-p $PARTITION \
-N $NNODES \
--ntasks-per-node=$NTASKS_PER_NODE \
--mem $MEM_PER_NODE \
-K \
"
if [ "$HOST" != "None" ]; then
    SLURM_ARGS="$SLURM_ARGS \
        -w $HOST \
    "
fi

# # Run with Slurm
RUNNER_CMD="srun $SLURM_ARGS"

set -x
$RUNNER_CMD \
$@

set +x
