#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/111_wcy/anaconda3}"
ENV_NAME="${VOC_GLUON_ENV_NAME:-voc-gluoncv}"
CONDA="$CONDA_ROOT/bin/conda"

if [[ ! -x "$CONDA" ]]; then
  echo "Conda is unavailable: $CONDA" >&2
  exit 2
fi

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA" create -y -n "$ENV_NAME" python=3.8 pip
fi

PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
"$CONDA" install -y -n "$ENV_NAME" cudatoolkit=11.8 cudnn=8.9
"$PYTHON" -m pip install --upgrade 'pip<25' 'setuptools<81' wheel
"$PYTHON" -m pip install \
  'numpy==1.23.5' \
  'mxnet-cu112==1.9.1' \
  'gluoncv==0.10.5.post0' \
  'Pillow==9.5.0' \
  'opencv-python==4.8.1.78' \
  'nvidia-nccl-cu11==2.15.5' \
  'portalocker==2.8.2'

NCCL="$CONDA_ROOT/envs/$ENV_NAME/lib/python3.8/site-packages/nvidia/nccl/lib/libnccl.so.2"
if [[ ! -f "$NCCL" ]]; then
  echo "NCCL runtime is unavailable: $NCCL" >&2
  exit 3
fi
ln -sfn "$NCCL" "$CONDA_ROOT/envs/$ENV_NAME/lib/libnccl.so.2"

"$PYTHON" - <<'PY'
import gluoncv
import mxnet as mx
import numpy as np
print({
    "python_ok": True,
    "mxnet": mx.__version__,
    "gluoncv": gluoncv.__version__,
    "numpy": np.__version__,
    "gpu_count": mx.context.num_gpus(),
})
PY
