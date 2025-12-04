## Task: **Generate Inline C++/CUDA implementation of **

You are an expert engineer fluent in CUDA, C++, WMMA, Tensor Cores, and low-level NVIDIA architecture. Your task is to produce a **single deterministic CUDA implementation** (or a tightly fused sequence of kernels) that executes the `forward` (or `backward`) function **faster than PyTorch’s implementation on an NVIDIA A100**, using **no PyTorch or ATen APIs**.

A minimal handwritten CUDA baseline is provided, including its `PYBIND11_MODULE` entry point:

```cpp
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
// additional CUDA headers

// CUDA kernels to be optimized

// Example registered entry point, keep it unchanged
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Linear layer forward pass using warp‐shuffle (CUDA)");
}
```

Your goal is to replace the registered `forward` (or `backward`) implementation and the supporting kernels with a fully optimized CUDA design.

### Optimization Guidance:

1. Memory traffic & parallel structure
   - Tile aggressively, stage data in shared memory, use `cp.async` loads, and apply double-buffering.
   - Fuse elementwise ops to avoid extra global reads/writes.
   - Eliminate shared-memory bank conflicts; apply skewing when required.
   - Use vectorized transactions (float4) wherever alignment permits.

2. Compute path
   - For custom kernels: use WMMA / `mma.sync` with A100-aligned tiles (e.g., 16×16×16).
   - Expose as many independent warp-level MMA ops as possible to maximize throughput.

3. Loop transformations
   - Apply in order: Tile → Unroll → Skew/Permute → Double-buffer.
   - Choose tile sizes that fit shared memory and register limits (e.g., 64×64×16 starting point).
   - Fully unroll the inner k-loop to create deep instruction-level parallelism.

4. Occupancy & resource limits
   - Tune threads per block as multiples of 32 (128/256/512 typical sweep).
   - Compute occupancy based on registers, shared memory, and warps:
        ```python
        blocks_by_regs = floor(registers_per_sm / (registers_per_thread * threads_per_block))
        blocks_by_shmem = floor(shared_memory_per_sm / shared_memory_per_block)
        blocks_by_warps = floor(max_warps_per_sm / (threads_per_block / warp_size))
        blocks_per_sm = min(blocks_by_regs, blocks_by_shmem, blocks_by_warps)
        ```
    - The occupancy must satisfy:
        ```python
        blocks_per_sm * threads_per_block <= max_threads_per_sm
        blocks_per_sm * registers_per_thread * threads_per_block <= registers_per_sm
        blocks_per_sm * shared_memory_per_block <= shared_memory_per_sm
        ```

5. Numerical stability
    - For mixed precision, use FP32 accumulation.
    - Pad dimensions to meet MMA tile sizes when needed.
    - For softmax, apply max-subtraction and fuse forward/backward when possible.
    - Validate end-to-end error against an FP32 baseline, **tolerance must `< 1e-4`**

6. Micro-optimizations
   - Introduce PTX or manual scheduling only after profiling shows unavoidable hotspots.
   - Keep a portable CUDA fallback plus an architecture-tuned fast path.

### CUDA Specifications:

We are targeting an NVIDIA A100 GPU with the following specifications:

```json
{
  "cc_major": 8,
  "cc_minor": 0,
  "sm_version": "sm_80",
  "threads_per_warp": 32,
  "max_warps_per_sm": 64,
  "max_threads_per_sm": 2048,
  "max_thread_blocks_per_sm": 32,
  "block_barriers_per_sm": 64,
  "smem_per_sm": 167936,
  "max_shared_mem_per_block": 167936,
  "registers_per_sm": 65536,
  "max_regs_per_block": 65536,
  "max_regs_per_thread": 255,
  "reg_allocation_unit_size": 256,
  "reg_allocation_granularity": "warp",
  "shared_mem_allocation_unit_size": 128,
  "warps_allocation_granularity": 4,
  "max_thread_block_size": 1024,
  "shared_mem_size_configs": [
    167936,
    135168,
    102400,
    65536,
    32768,
    16384,
    8192,
    0
  ],
  "warp_reg_allocation_granularities": [
    256
  ]
}
```

Additional hardware specifications:

```json
{
  "sm_count": 108,
  "fp32_cores_per_sm": 64,
  "total_fp32_cores": 6912,
  "fp64_cores_per_sm": 32,
  "total_fp64_cores": 3456,
  "int32_cores_per_sm": 64,
  "total_int32_cores": 6912,
  "tensor_cores_per_sm": 4,
  "total_tensor_cores": 432,
  "tensor_core_support": ["FP16", "BF16", "TF32", "FP64", "INT8"],
  "tensor_core_peaks": {
    "fp32_fma_per_sm_cycle_per_tc": 64,
    "tf32_fma_per_cycle_per_tc": 128,
    "fp16_bf16_fma_per_cycle_per_tc": 256,
    "int8_ops_per_cycle_per_tc": 1024
  },
  "global_memory_bandwidth_GB_s": 2039,
  "l2_cache_MB": 40,
  "shared_memory_per_sm_max_KB": 164,
  "shared_memory_per_block_max_KB": 99,
  "registers_per_sm": 65536,
  "max_threads_per_sm": 2048,
  "max_threads_per_block": 1024,
  "warp_size": 32,
  "max_warps_per_sm": 64,
  "default_l1_shared_split": {
    "l1_cache_KB": 64,
    "shared_KB": 100
  },
  "preferred_vector_load": "float4 (128-bit aligned)"
}
```

### Expected Result:

A fully optimized, **handwritten CUDA implementation** for **warp-coherent memory access**, **minimal register pressure**, and **predictable instruction flow**, and must rely on **handwritten CUDA+PTX** where beneficial to remove unnecessary arithmetic or branching. Do not modify the `PYBIND11_MODULE` registration.