source ~/yhy/.local/spack/share/spack/setup-env.sh
spack load cuda@11.8.0
# source /data/apps/tools/spack/share/spack/setup-env.sh
# spack load cuda@11.8

# # Openmpi
# export OPENMPI_HOME=/home/zhaijidong/yhy/.local/openmpi
# export PATH="$OPENMPI_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$OPENMPI_HOME/lib/:$LD_LIBRARY_PATH"

# export C_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$C_INCLUDE_PATH"  # for #include <mpi.h>
# export CPLUS_INCLUDE_PATH="$(dirname `which mpicxx`)/../include:$CPLUS_INCLUDE_PATH"  # for #include <mpi.h>
# export LD_LIBRARY_PATH="$(dirname `which nvcc`)/../lib64:$LD_LIBRARY_PATH"  # for -lcudart

# # cuda
# export CUDA_HOME=~/yhy/.local/cuda-11.8
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# conda
conda deactivate && conda deactivate && conda deactivate
conda activate yhy_easycontext

# [NOTE]: tmux of qy has bug along with conda, thus the solution is copy the $PATH out of tmux into tmux  !!!

# export http_proxy="http://127.0.0.1:8901"
# export https_proxy="http://127.0.0.1:8901"

# export http_proxy="http://nico0:8901"
# export https_proxy="http://nico0:8901"

# unset http_proxy
# unset https_proxy

# nico0: 127.23.18.10
# nico2: 127.23.18.2

# git config --global http.proxy http://127.0.0.1:8901
# git config --global https.proxy http://127.0.0.1:8901
# git config --global --unset http.proxy
# git config --global --unset https.proxy

# git config http.proxy http://127.0.0.1:8901
# git config https.proxy http://127.0.0.1:8901
# git config --unset http.proxy
# git config --unset https.proxy

# git config user.name "oliverYoung2001"
# git config user.email "XXXXXXX"