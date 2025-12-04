import os

os.environ["ROCBLAS_LAYER"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    y = torch.randn(512, 512, device="cuda")
    z = x @ y
    print("ok, matmul", z.shape)
else:
    print("no gpu")
