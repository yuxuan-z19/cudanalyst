## Task: **Generate Inline C++/CUDA implementation of **

You are an expert engineer fluent in Python, C++, CUDA and PyTorch. Your task now is to deliver a single, **deterministic CUDA implementation** (or a tightly bound sequence of kernels) for the `forward` function that **outperforms** PyTorch's native implementation on an NVIDIA A100. You could only devliver CUDA code, **none of the PyTorch or ATen APIs is allowed.**

### Implementation Requirements:

You are provided a initial solution:

```python
class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x):
        ...
```

**Implement your CUDA + PTX code for the `ModelNew.forward` function.** Embed your inline kernel code in `cuda_src_code()`, the host caller in `cpp_src_code()`, and the corresponding compiler flags in `cuda_cflags()`. The code should have unique, templated kernel name specified as `base` (e.g. `kernel_linear_<unique_id>`, `kernel_relu_<unique_id>`, etc.) in `name()` to avoid name conflicts. **Keep the `ModelNew` class as the entry point.**

```python
import hashlib
from types import ModuleType

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


class Solution:
    def name(self):
        base = "<replace with a unique name>"
        code_hash = hashlib.md5(
            (self.cuda_src_code() + self.cpp_src_code()).encode()
        ).hexdigest()[:8]
        return f"{base}_{code_hash}"

    def cuda_src_code(self):
        return r"""<insert your generated .cu code here>"""

    def cpp_src_code(self):
        return r"""<insert your generated .cpp code here>"""

    def cuda_cflags(self):
        return ["-O3", "--use_fast_math", "-arch=sm_80"]


def gen_inline_cuda():
    src = Solution()
    cuda_module: ModuleType = load_inline(
        name=src.name(),
        cpp_sources=src.cpp_src_code(),
        cuda_sources=src.cuda_src_code(),
        functions=["forward"],
        extra_cuda_cflags=src.cuda_cflags(),
    )
    return cuda_module


class ModelNew(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.cuda_module = gen_inline_cuda()

    def forward(self, x):
        return self.cuda_module.forward(x, ...)
```

Provide only fully compilable, production-ready code - no pseudocode, no tests, no extra commentary. **Remove all placeholders and task configuration.**

### Optimization Guidance:

1. Memory traffic & parallel structure
   - Use tiling, shared memory, cp.async prefetching, and double buffering.
   - Fuse elementwise ops to avoid extra global reads/writes.
   - Eliminate shared-memory bank conflicts; apply skewing when required.

2. Compute path
   - Prefer Tensor Core libraries (cuBLASLt / CUTLASS).
   - For custom kernels: use WMMA / mma.sync with A100-aligned tiles (e.g., 16×16×16).
   - Expose many independent warp-level MMA ops to maximize throughput.

3. Loop transformations
   - Apply in order: Tile → Unroll → Skew/Permute → Double-buffer.
   - Choose tile sizes that fit shared memory and register limits (e.g., 64×64×16 starting point).
   - Fully unroll the inner k-loop to expose instruction-level parallelism.

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

A complete, **fully inlined CUDA + PTX implementation** embedded into the `Solution` class, generating a **uniquely named templated kernel** inside `cuda_src_code()`, with an efficient C++ launcher in `cpp_src_code()`, and the correct compilation flags returned by `cuda_cflags()`.

The kernel must be optimized for **warp-coherent memory access**, **minimal register pressure**, and **predictable instruction flow**, and must rely on **handwritten CUDA+PTX** where beneficial to remove unnecessary arithmetic or branching. The code must compile as-is through `torch.utils.cpp_extension.load_inline` and expose a callable `forward` host entry, with strict preservation of all input/output shapes and semantics. No comments.