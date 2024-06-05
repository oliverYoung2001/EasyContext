# source /data/apps/tools/spack/share/spack/setup-env.sh
# spack load cuda@11.8

# # Openmpi
# export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi
# export PATH="$OPENMPI_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

# export C_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$C_INCLUDE_PATH"  # for #include <mpi.h>
# export CPLUS_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$CPLUS_INCLUDE_PATH"  # for #include <mpi.h>
# export LD_LIBRARY_PATH="$(dirname `which nvcc`)/../lib64:$LD_LIBRARY_PATH"  # for -lcudart

# cuda
CUDA_PATH=~/yhy/.local/cuda-11.8
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# conda
conda deactivate && conda deactivate
conda activate yhy_easycontext
