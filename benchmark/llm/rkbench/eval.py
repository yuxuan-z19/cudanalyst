import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_kbench.primitives.evaluate import (
    correct_cuda_kernel,
    eval_cuda_kernel,
    eval_torch_runtime,
)

from cugedit.helper import parse_cmd_failure, pick_idle_gpu
from cugedit.result import Result, ResultConstant, ResultMeta, Status, return_asdict

SUITE_ROOT = Path(__file__).parent / "robust_kbench"
BASELINE_DIR = SUITE_ROOT / "highlighted"
EVALUATOR = SUITE_ROOT / "run_kernel.py"
# REPEAT_TIME = 10000
REPEAT_TIME = 100
EVAL_KWARGS_BASE = dict(
    multi_init_settings=True,
    multi_input_settings=True,
    timeout=1_000_000,
)

# * Backward kernels are evaluated only with time_function_kernel_bench (see sandbox/eval_backward_fn.py)
EVAL_TYPE = "kernelbench"


@dataclass
class RKbenchMeta(ResultMeta):
    runtime: float = ResultConstant.INVALID_FLOAT


@dataclass
class RKbenchResult(Result):
    base_runtime: float = ResultConstant.INVALID_FLOAT
    custom_runtime: float = ResultConstant.INVALID_FLOAT
    naive_runtime: float = ResultConstant.INVALID_FLOAT
    compile_runtime: float = ResultConstant.INVALID_FLOAT


def parse_invalid(correct_result: dict[str, Any]) -> RKbenchResult:
    e = subprocess.CalledProcessError(
        -1,
        correct_result["nvcc_output"],
        correct_result["stdout"],
        correct_result["stderr"],
    )
    return parse_cmd_failure(e)


def is_backward(file_path: os.PathLike):
    code = Path(file_path).read_text()
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    backward_pattern = r'm\.def\s*\(\s*["\']backward["\']'
    return re.search(backward_pattern, code) is not None


def execute(program_path: Path, problem_dir: Path, eval_kwargs: dict[str, Any]):
    dst_program_path = problem_dir / program_path.name
    shutil.copy(program_path, dst_program_path)
    dst_program = dst_program_path.as_posix()

    correct_result = correct_cuda_kernel(problem_dir, dst_program, **eval_kwargs)
    if not correct_result["summary"]["correct"]:
        return RKbenchMeta(error=parse_invalid(correct_result))

    cuda_result = eval_cuda_kernel(
        problem_dir,
        dst_program,
        eval_type=EVAL_TYPE,
        repetition_time=REPEAT_TIME,
        **eval_kwargs,
    )
    return RKbenchMeta(runtime=cuda_result["summary"]["avg_mean_time"])


# * Reference: https://github.com/SakanaAI/robust-kbench/blob/main/run_kernel.py
@return_asdict
def evaluate(program_path: os.PathLike, problem_dir: os.PathLike):
    program_path = Path(program_path)
    problem_dir = Path(problem_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)

        is_back = is_backward(program_path)
        gpu_id = pick_idle_gpu()
        eval_kwargs = {**EVAL_KWARGS_BASE, "gpu_id": gpu_id, "forward": not is_back}

        base_program = (
            BASELINE_DIR
            / problem_dir.stem
            / ("backward" if is_back else "forward")
            / "kernel.cu"
        )
        base_dir = tmp_dir / "base"
        custom_dir = tmp_dir / "custom"
        shutil.copytree(problem_dir, base_dir)
        shutil.copytree(problem_dir, custom_dir)

        torch_naive_res, torch_compile_res = eval_torch_runtime(
            base_dir, eval_type=EVAL_TYPE, repetition_time=REPEAT_TIME, **eval_kwargs
        )
        naive_runtime = torch_naive_res["summary"]["avg_mean_time"]
        compile_runtime = torch_compile_res["summary"]["avg_mean_time"]

        base_result = execute(base_program, base_dir, eval_kwargs)
        if base_result.status != Status.PASS:
            return RKbenchResult(status=base_result.status, error=base_result.error)

        custom_result = execute(program_path, custom_dir, eval_kwargs)
        if custom_result.status != Status.PASS:
            return RKbenchResult(status=custom_result.status, error=custom_result.error)

        speedup = base_result.runtime / custom_result.runtime
        return RKbenchResult(
            combined_score=speedup,
            base_runtime=base_result.runtime,
            custom_runtime=custom_result.runtime,
            naive_runtime=naive_runtime,
            compile_runtime=compile_runtime,
        )


if __name__ == "__main__":
    import time
    from pprint import pprint

    from tqdm.rich import tqdm

    workload_list = []
    for dirpath, _, filenames in os.walk(BASELINE_DIR):
        if "kernel.cu" in filenames:
            kernel_path = Path(dirpath) / "kernel.cu"
            label = kernel_path.parent.stem
            op_name = kernel_path.parent.parent.stem
            workload_list.append((op_name, label, kernel_path))

    for op_name, label, kernel_path in tqdm(workload_list):
        print(op_name, label)
        tstart = time.perf_counter()
        result = evaluate(kernel_path, SUITE_ROOT / "tasks" / op_name)
        tend = time.perf_counter()
        if result:
            print(f"--- Result for {op_name} ({label}) n {tend - tstart:.4f} sec ---")
            pprint(result)
        else:
            break
