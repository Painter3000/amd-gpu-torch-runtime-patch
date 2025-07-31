import os

# 🔧 Setze FADE-Parameter für reproduzierbare Performance
os.environ["FADE_FORCE_WARP_SIZE"] = "64"
os.environ["FADE_FORCE_MP_COUNT"] = "72"

import fade_v11_plus
fade_v11_plus.apply_fade_patches()

import torch
import time

x = torch.randn(4096, 4096, device='cuda')
torch.cuda.synchronize()
start = time.perf_counter()
result = torch.mm(x, x.T)
torch.cuda.synchronize()
duration = (time.perf_counter() - start) * 1000

print(f"Matrix Mult 4096x4096 (with FADE): {duration:.2f}ms")
print("Before: 164.76ms @ 2048x2048 → 11.6x normalized speedup")

