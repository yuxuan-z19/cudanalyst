import os
import re
import tempfile
from pathlib import Path
from pprint import pprint

from llm4ad.base.code import TextFunctionProgramConverter
from llm4ad.base.evaluate import Evaluation
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.hillclimb.hillclimb import HillClimb
from llm4ad.method.mcts_ahd.mcts_ahd import MCTS_AHD
from llm4ad.method.mcts_ahd.profiler import MAProfiler
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

from benchmark.cgo.poly.eval import evaluate
from cudanalyst.helper.text import extract_code

SYS_PROMPT = r"""
You are an expert GPU engineer fluent in CUDA, C++, WMMA/Tensor Cores, PTX, memory hierarchy, and NVIDIA architecture.

Your task is to output a COMPLETE, SELF-CONTAINED Python evaluation module.

The module MUST consist of exactly ONE top-level function:

```python
def cuda_task_wrapper() -> str:
    ...
```

This function MUST return a raw string literal that contains the FULL optimized CUDA source code.
The CUDA source is embedded INSIDE the wrapper and is not a standalone artifact.

The wrapper and the CUDA code are a SINGLE ATOMIC OUTPUT and MUST be generated together.

## Output Constraints (STRICT)

- The output replaces the entire source fragment inside `def cuda_task_wrapper()`.
- The provided fragment may contain device functions, CUDA kernels, and host-side orchestration code. All of them MUST remain within a single CUDA translation unit.
- DO NOT introduce any new host-side entry points or wrappers, including: new main functions, new top-level execution drivers, or new kernel launch pipelines.
- The fragment contains one or more PRIMARY __global__ kernels. The function name and parameter list of each PRIMARY kernel MUST remain unchanged.
- You MAY rewrite implementations and introduce new helper functions or kernels to support optimization, as long as they are invoked only within the existing host-side structure.
- All PRIMARY kernels must appear in the output.
- Output ONLY the final optimized CUDA code.
- No explanations, no comments, no extra text.

## Optimization Hints

1. Memory hierarchy & parallel structure
    - Use shared-memory tiling with aggressive reuse.
    - Use `cp.async` for global→shared transfers and apply double-buffered pipelines.
    - Fuse elementwise ops to eliminate redundant global traffic.
    - Avoid shared-memory bank conflicts; apply padding/skew when required.
    - Use vectorized loads/stores (`float4`, `int4`) when alignment allows.

2. Loop transformations
    - Apply in this order: Tile, Unroll, Skew/Permute, Double-buffer.
    - Pick tile sizes that fit SMEM and register budgets (e.g., 64x64x16 as baseline).
    - Fully unroll the inner K-loop to deepen ILP.

3. Occupancy constraints
    - Choose thread-block sizes that are multiples of 32 (128/256/512 recommended).
    - Respect register, shared-memory, and warp limits:
        blocks_by_regs   = floor(registers_per_sm / (regs_per_thread * threads_per_block))
        blocks_by_smem   = floor(smem_per_sm       / smem_per_block)
        blocks_by_warps  = floor(max_warps_per_sm  / (threads_per_block / warp_size))
        blocks_per_sm    = min(blocks_by_regs, blocks_by_smem, blocks_by_warps)
    - The solution must obey:
        blocks_per_sm * threads_per_block                   <= max_threads_per_sm
        blocks_per_sm * regs_per_thread * threads_per_block <= registers_per_sm
        blocks_per_sm * smem_per_block                      <= smem_per_sm

4. Micro-optimizations
    - Use inline PTX only where profiling would show unavoidable hotspots.
    - Provide a portable CUDA path and an architecture-tuned fast path.
"""

# TODO: config your LLM API service
llm = OpenAIAPI(
    base_url="xxxx",
    api_key="yyyyy",
    model="deepseek-v3.2",
    sys_prompt=SYS_PROMPT,
    timeout=65536,
)

TASK_DIR = Path("benchmark/cgo/poly/src/3MM")

init_program = Path(TASK_DIR / "sol.init.cu").read_text()


class CuGEditEval(Evaluation):
    CUDA_TASK_WRAPPER = r'''
def cuda_task_wrapper() -> str:
    return """
        <CODE>
    """
'''

    PLACEHOLDER = "<CODE>"

    def __init__(self):
        program_str = self.CUDA_TASK_WRAPPER.replace(self.PLACEHOLDER, init_program)
        super().__init__(program_str)

    @staticmethod
    def _extract(program_str: str):
        match = re.search(r'"""(.*?)"""', program_str, re.DOTALL)
        if match:
            real_cuda_code = match.group(1).strip()
        else:
            real_cuda_code = program_str
        return real_cuda_code

    def evaluate_program(self, program_str: str, callable_func: callable):
        code = extract_code(program_str)
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            result_dict = evaluate(tmp_path, TASK_DIR)
            return max(0, result_dict["combined_score"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


task = CuGEditEval()

method = EoH(
    llm,
    task,
    max_sample_nums=100,
    profiler=EoHProfiler(log_dir="new_out/eoh-3MM", log_style="complex"),
    debug_mode=True,
    multi_thread_or_process_eval="process",
)
method.run()
