import os
import shutil
import tempfile
from pathlib import Path
from pprint import pp
from typing import Any

from cugedit.config import *
from cugedit.utils import pick_idle_gpu
from robust_kbench.robust_kbench.primitives.evaluate import (
    correct_cuda_kernel,
    eval_cuda_kernel,
    eval_torch_runtime,
)


@dataclass
class RKbenchResult(Result):
    speedup_naive: float = -1
    speedup_compile: float = -1
    runtime_torch_naive: float = -1
    runtime_torch_compile: float = -1
    runtime_cuda: float = -1


EVALUATOR = Path(__file__).parent / "robust_kbench/run_kernel.py"

COMPILE_LOG_TEMPLATE = r"""
# CUDA Compilation Debug

## STDOUT
<stdout>

## STDERR
<stderr>

## NVCC LOG
<nvcc_output>

---

Instructions: 
Analyze why the CUDA compilation failed. Focus on:
- Errors/warnings in STDERR
- NVCC flags, include paths, library links
- Architecture (-gencode) and macro definitions
"""


def parse_invalid(correct_result: dict[str, Any]) -> RKbenchResult:
    error = COMPILE_LOG_TEMPLATE
    for key in ("stdout", "stderr", "nvcc_output"):
        error = error.replace(f"<{key}>", correct_result[key])
    return RKbenchResult(error=error)


def is_backward(file_path: os.PathLike):
    code = Path(file_path).read_text()
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    backward_pattern = r'm\.def\s*\(\s*["\']backward["\']'
    return re.search(backward_pattern, code) is not None


# * Reference: https://github.com/SakanaAI/robust-kbench/blob/main/run_kernel.py
@return_asdict
def evaluate(program_path: os.PathLike, problem_dir: os.PathLike, perf: bool = False):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        task_dir = tmp_dir / Path(problem_dir).stem
        shutil.copytree(problem_dir, task_dir)

        dst_program = tmp_dir / Path(program_path).name
        shutil.copy(program_path, dst_program)
        dst_program = str(dst_program)

        eval_kwargs = dict(
            multi_init_settings=True,
            multi_input_settings=True,
            gpu_id=pick_idle_gpu(),
            forward=not is_backward(program_path),
        )

        torch_naive_result, torch_compile_result = eval_torch_runtime(
            task_dir, **eval_kwargs
        )
        correct_result = correct_cuda_kernel(task_dir, dst_program, **eval_kwargs)

        if not correct_result["summary"]["correct"]:
            return parse_invalid(correct_result)

        cuda_result = eval_cuda_kernel(task_dir, dst_program, **eval_kwargs)
        runtime_cuda = cuda_result["summary"]["avg_mean_time"]
        speedup_naive = torch_naive_result["summary"]["avg_mean_time"] / runtime_cuda
        speedup_compile = (
            torch_compile_result["summary"]["avg_mean_time"] / runtime_cuda
        )

        # TODO: add analyst report
        report = ""

        return RKbenchResult(
            combined_score=(speedup_naive + speedup_compile) / 2,
            report=report,
            speedup_naive=speedup_naive,
            speedup_compile=speedup_compile,
            runtime_torch_naive=torch_naive_result["summary"]["avg_mean_time"],
            runtime_torch_compile=torch_compile_result["summary"]["avg_mean_time"],
            runtime_cuda=runtime_cuda,
        )


if __name__ == "__main__":
    pp(
        evaluate(
            "./robust_kbench/highlighted/mnist_cross_entropy/forward/kernel.cu",
            "./robust_kbench/tasks/mnist_cross_entropy",
        )
    )
