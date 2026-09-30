"""Import right after torch: on a CPU-only node, e3nn codegen submodules (TorchScript buffers saved on CUDA) are
loaded by torch.jit.load(buffer) without map_location inside the model's unpickling; default it to cpu."""
import os
import torch

DEVICE = os.environ.get("KIT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == "cpu":
    _jit_load = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kw: _jit_load(f, map_location="cpu", **kw)
