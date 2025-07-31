#!/usr/bin/env python3
# examples/benchmark_test.py
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 🔧 Setze FADE-Parameter für reproduzierbare Performance
os.environ["FADE_FORCE_WARP_SIZE"] = "64"
os.environ["FADE_FORCE_MP_COUNT"] = "72" # Wert für andere GPUs dementsprechend anpassen / Adjust value for other GPUs accordingly

# 🔕 Deaktiviere FADE-Logger für saubere Ausgabe
logging.getLogger("FADE").setLevel(logging.CRITICAL)

import fade_v11_plus
fade_v11_plus.apply_fade_patches()

import torch
import time

# 🔁 Stabiler Benchmark mit Warmup und Seed
torch.manual_seed(42)

print("🔥 GPU Warmup...")
warmup = torch.randn(1024, 1024, device='cuda')
torch.mm(warmup, warmup.T)
torch.cuda.synchronize()

durations = []
num_runs = 5
print(f"🔁 Running {num_runs} iterations...")

for i in range(num_runs):
    torch.manual_seed(42 + i)  # leicht variieren, um Kernel-Caching zu verhindern
    x = torch.randn(4096, 4096, device='cuda')

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = torch.mm(x, x.T)
    torch.cuda.synchronize()
    duration = (time.perf_counter() - start) * 1000
    durations.append(duration)
    print(f"Run {i+1}: {duration:.2f} ms")

print("\n📋 Matrix Mult - Einzel-Ergebnisse:")
for i, dur in enumerate(durations):
    print(f"Run {i+1}: {dur:.2f} ms")

avg = sum(durations) / len(durations)
print(f"\n⏱️ Avg duration over {num_runs} runs: {avg:.2f} ms @ 4096×4096")
print(f"🚀 vs Baseline (164.76ms @ 2048x2048): ~{164.76 / (avg/8):.1f}x speedup")
