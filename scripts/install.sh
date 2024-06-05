#!/bin/bash

source ./scripts/env_qy.sh
conda create -n yhy_easycontext python=3.10 -y && conda activate yhy_easycontext
# pip install --pre torch==2.4.0.dev20240324  --index-url https://download.pytorch.org/whl/nightly/cu118
# pip install --pre torch==2.4.0.dev20240324  --index-url https://download.pytorch.org/whl/nightly/cu118
pip install --pre torch==2.1.2  --index-url https://download.pytorch.org/whl/nightly/cu118
# conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
