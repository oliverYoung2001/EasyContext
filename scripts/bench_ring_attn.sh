#!/bin/bash
if [ -z $MASTER_ADDR ]
then
    if [ -z $SLURM_JOB_ID ]
    then
        export MASTER_ADDR=localhost
    else
        export MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
    fi
fi
if [ -z $MASTER_PORT ]
then
    export MASTER_PORT=12215
fi

if [ ! -z $OMPI_COMM_WORLD_RANK ]
then
    export RANK=$OMPI_COMM_WORLD_RANK
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    localrank=$OMPI_COMM_WORLD_LOCAL_RANK
elif [ ! -z $SLURM_PROCID ]
then
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NPROCS
    localrank=$SLURM_LOCALID
else
    RANK=0
    localrank=0
    WORLD_SIZE=1
fi

# echo "RANK: $RANK, WORLD_SIZE: $WORLD_SIZE, localrank $localrank"
# echo "MASTER_PORT: $MASTER_PORT"
# export CUDA_VISIBLE_DEVICES=$localrank

# set -x

# echo "USE_NSYS: $USE_NSYS"
NSIGHT_CMD=""
# if [ $USE_NSYS == "True" ]
# then
#     NSIGHT_CMD="nsys profile --output=${NSYS_DIR}/${TRACE_NAME}_w${WORLD_SIZE}_r${RANK}_$(date "+%Y%m%d-%H%M%S")"
# fi

exec ${NSIGHT_CMD} $@ # 2>&1 | tee logs/out/$RANK
