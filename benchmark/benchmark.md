# Benchmark Report for */home/carlo/.julia/dev/GraphNeuralNetworks*

## Job Properties
* Time of benchmark: 25 Feb 2022 - 16:17
* Package commit: dirty
* Julia commit: ac5cc9
* Julia command flags: None
* Environment variables: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 1`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                               | time            | GC time | memory          | allocations |
|------------------------------------------------------------------|----------------:|--------:|----------------:|------------:|
| `["gcnconv", "coo_100_GraphNeuralNetworks.GCNConv", "CPU_FWD"]`  |  86.176 μs (5%) |         | 249.62 KiB (1%) |          58 |
| `["gcnconv", "coo_100_GraphNeuralNetworks.GCNConv", "CPU_GRAD"]` |   2.688 ms (5%) |         | 757.31 KiB (1%) |        1887 |
| `["gcnconv", "coo_10_GraphNeuralNetworks.GCNConv", "CPU_FWD"]`   |  20.919 μs (5%) |         |  27.80 KiB (1%) |          51 |
| `["gcnconv", "coo_10_GraphNeuralNetworks.GCNConv", "CPU_GRAD"]`  | 910.193 μs (5%) |         | 219.61 KiB (1%) |        1869 |
| `["propagate", "copy_xj", "_baseline"]`                          | 408.494 μs (5%) |         | 800.05 KiB (1%) |           2 |
| `["propagate", "copy_xj", "fused"]`                              | 561.269 μs (5%) |         |   1.21 MiB (1%) |          31 |
| `["propagate", "copy_xj", "unfused"]`                            |   2.065 ms (5%) |         |   8.95 MiB (1%) |          10 |
| `["propagate", "e_mul_xj", "_baseline"]`                         | 373.447 μs (5%) |         | 800.05 KiB (1%) |           2 |
| `["propagate", "e_mul_xj", "fused"]`                             | 552.230 μs (5%) |         |   1.13 MiB (1%) |          32 |
| `["propagate", "e_mul_xj", "unfused"]`                           |   3.596 ms (5%) |         |  17.11 MiB (1%) |          12 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["gcnconv", "coo_100_GraphNeuralNetworks.GCNConv"]`
- `["gcnconv", "coo_10_GraphNeuralNetworks.GCNConv"]`
- `["propagate", "copy_xj"]`
- `["propagate", "e_mul_xj"]`

## Julia versioninfo
```
Julia Version 1.7.1
Commit ac5cc99908 (2021-12-22 19:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
      "Manjaro Linux"
  uname: Linux 5.16.7-1-MANJARO #1 SMP PREEMPT Sun Feb 6 12:22:29 UTC 2022 x86_64 unknown
  CPU: Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz: 
                 speed         user         nice          sys         idle          irq
       #1-12  1185 MHz     484978 s       9416 s     116850 s    2788141 s      43648 s
       
  Memory: 15.464141845703125 GB (286.4765625 MB free)
  Uptime: 63306.98 sec
  Load Avg:  3.98  3.07  3.06
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake)
```