# GPU Scheduler Simulator

Simulates and compares 3 GPU scheduling strategies for AI workloads:
- FIFO
- Priority scheduling  
- Work-stealing

## Results
Work-stealing reduced GPU idle time from 18.8% → 5% and finished 14% faster than FIFO.
Priority scheduling cut high-priority job wait time by 40% but didn't improve throughput.

## How to run
pip install simpy matplotlib numpy
python gpu_scheduler_sim.py

## Why this matters
This simulates the core problem that NVIDIA Triton Inference Server solves —
balancing throughput vs latency across multi-GPU systems.
