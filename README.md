# gpu-alignment-benchmark

I built this to answer a question that kept bugging me after 15-213: when people say "cache locality matters," how much does it actually matter on a GPU?

As such, made this small C++20/CUDA framework that benchmarks two attention kernels, a naive 3-kernel version and a tiled fused version, on a Tesla T4. Same math, same FLOPs (within ~1%), but different data placement strategies. The tiled kernel keeps intermediates in shared memory and registers instead of writing them out to DRAM between steps.


The framework also has an analytical roofline model so you can predict where a kernel *should* land before you measure it, and then Nsight Compute validation to see where it *actually* lands. The analytical model is interesting; it's too optimistic for naive (ignores L2 thrashing) and too pessimistic for tiled (ignores L2 absorbing re-reads).

## What's in here

```
include/
  core/           Algorithm base class, ProblemSize, RunResult
  kernels/        Naive attention, tiled attention, SSM scan
  modeling/       FLOP/byte estimators, roofline model, T4 specs
  profiling/      CUDA event timer, ncu metric collection, validation reports
  utils/          CUDA error-check macros
src/
  kernels/        .cu implementations
  modeling/        Analytical model + roofline
  profiling/      ncu CSV parsing, validation logic, plot data generation
  main.cpp        Entry point — runs all phases
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./benchmark
```

For Nsight Compute validation (needs sudo for GPU perf counters):

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PROFILING=ON
make -j$(nproc)
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
./benchmark
```

This dumps CSV/JSON to `data/` and gnuplot scripts to `data/plots/`.

## Hardware

Tested on Tesla T4 (sm_75, Turing). Peak FP32: 8.1 TFLOP/s, peak BW: 320 GB/s, ridge point: 25.31 FLOP/Byte. You can target other architectures with `-DCMAKE_CUDA_ARCHITECTURES=XX` but the hardcoded T4 specs in the analytical model would need updating.

## Key numbers (N=1024)

| | Naive | Tiled |
|---|---|---|
| DRAM traffic | 409 MB | 18.2 MB |
| Arithmetic intensity | 5.35 FLOP/B | 119 FLOP/B |
| Throughput | 166 GFLOP/s | 2,910 GFLOP/s |
| % of roofline | 3.4% | 35.9% |
| Occupancy | 97% | 25% |

The naive kernel has 97% occupancy and still only hits 3.4% of peak. Occupancy is not performance.
