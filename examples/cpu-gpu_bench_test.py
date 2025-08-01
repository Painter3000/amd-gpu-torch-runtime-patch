#!/usr/bin/env python3
# cpu-gpu_bench_test.py
# python fade_bench.py --size 4096 --runs 5 --cpu --event
# --size: Matrixgröße (z. B. 2048, 4096)
# --runs: Anzahl der Wiederholungen
# --cpu: Optionaler CPU-Vergleich
# --event: Nutzt torch.cuda.Event für präziseres GPU-Timing

import sys
import os
import logging
import argparse
import torch
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 📦 FADE Setup
os.environ["FADE_FORCE_WARP_SIZE"] = "64"
os.environ["FADE_FORCE_MP_COUNT"] = "72"  # Passe diesen Wert je nach GPU an

logging.getLogger("FADE").setLevel(logging.CRITICAL)

try:
    import fade_v11_plus
    fade_v11_plus.apply_fade_patches()
except ImportError:
    print("⚠️ FADE-Modul nicht gefunden. Stelle sicher, dass fade_v11_plus installiert ist.")
    sys.exit(1)

# 🧮 Benchmark-Funktion
def benchmark_matrix_mult(size: int, runs: int, use_event: bool = False):
    print(f"🔥 GPU Warmup @ {size}×{size}...")
    warmup = torch.randn(size, size, device='cuda')
    torch.mm(warmup, warmup.T)
    torch.cuda.synchronize()

    durations = []
    for i in range(runs):
        torch.manual_seed(42 + i)
        x = torch.randn(size, size, device='cuda')

        if use_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = torch.mm(x, x.T)
            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event)
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = torch.mm(x, x.T)
            torch.cuda.synchronize()
            duration = (time.perf_counter() - start) * 1000

        durations.append(duration)
        print(f"Run {i+1}: {duration:.2f} ms")

    avg = sum(durations) / len(durations)
    print(f"\n⏱️ Avg duration over {runs} runs: {avg:.2f} ms @ {size}×{size}")
    return avg

# 🐢 Optional: CPU-Vergleich
def benchmark_cpu(size: int):
    x = torch.randn(size, size)
    start = time.perf_counter()
    result = torch.mm(x, x.T)
    duration = (time.perf_counter() - start) * 1000
    print(f"🐢 CPU duration: {duration:.2f} ms @ {size}×{size}")
    return duration

# 🧰 CLI-Interface
def main():
    parser = argparse.ArgumentParser(description="FADE GPU Matrix Multiplication Benchmark")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (NxN)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--cpu", action="store_true", help="Include CPU benchmark")
    parser.add_argument("--event", action="store_true", help="Use torch.cuda.Event for timing")
    args = parser.parse_args()

    print(f"\n🚀 FADE Benchmark: Matrix Multiplication on GPU")
    print(f"📐 Size: {args.size}×{args.size}, 🔁 Runs: {args.runs}\n")

    avg_gpu = benchmark_matrix_mult(args.size, args.runs, use_event=args.event)

    if args.cpu:
        avg_cpu = benchmark_cpu(args.size)
        print(f"\n⚖️ GPU vs CPU Speedup: ~{avg_cpu / avg_gpu:.1f}x")

if __name__ == "__main__":
    main()

