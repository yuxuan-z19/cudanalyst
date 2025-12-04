import os
import sys
import tempfile
from pathlib import Path
from pprint import pp

import torch
from cugedit.analyst import Analyst
from cugedit.config import *
from cugedit.utils import pick_idle_gpu
from kernelbench.src.eval import (
    KernelExecResult,
    eval_kernel_against_ref,
    graceful_eval_cleanup,
    load_custom_model,
    load_original_model_and_inputs,
)


@dataclass
class KbenchResult(Result):
    base_runtime_stats: dict = field(default_factory=dict)
    custom_runtime_stats: dict = field(default_factory=dict)


NO_SOLUTION_ERROR = (
    "Error: Missing a valid `Solution` class. "
    "Expected methods: name(), cuda_src_code(), cpp_src_code(), cuda_cflags(). "
    "Implement your own CUDA/C++ source generation here."
)

NO_CUDA_MODULE_ERROR = (
    "Error: Missing `gen_inline_cuda()` loader. "
    "This function must construct and return a CUDA module using torch.utils.cpp_extension.load_inline()."
)


def check_cuda_module(code: str) -> KbenchResult:
    try:
        ctx = {}
        compile(code, "<string>", "exec")
        exec(code, ctx)

        Solution = ctx.get("Solution")
        gen_inline_cuda = ctx.get("gen_inline_cuda")

        if gen_inline_cuda is None or not callable(gen_inline_cuda):
            return KbenchResult(error=NO_CUDA_MODULE_ERROR)

        if Solution is None or not isinstance(Solution, type):
            return KbenchResult(error=NO_SOLUTION_ERROR)

        required_methods = ["name", "cuda_src_code", "cpp_src_code", "cuda_cflags"]
        for m in required_methods:
            if not hasattr(Solution, m):
                return KbenchResult(error=NO_SOLUTION_ERROR)

        return None
    except Exception as e:
        return KbenchResult(error=str(e))


def extract_kernels(Model, get_init_inputs, get_inputs, device="cuda"):
    device = torch.device(device)

    init_inputs = get_init_inputs()
    inputs = get_inputs()

    init_inputs = [
        x.to(device) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]
    inputs = [x.clone().to(device) for x in inputs]

    torch.cuda.set_device(device)

    with torch.no_grad():
        model = Model(*init_inputs).to(device)

        _ = model(*inputs)
        torch.cuda.synchronize(device)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ]
        ) as prof:
            _ = model(*inputs)

        torch.cuda.synchronize(device)

    del model
    del init_inputs
    del inputs
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    graceful_eval_cleanup({}, device)

    return {evt.key for evt in prof.key_averages()}


def is_kernel_hacked(problem: str, code: str):
    try:
        device = pick_idle_gpu()

        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(problem, {})
        ModelNew = load_custom_model(code, {})

        base_kernels = extract_kernels(Model, get_init_inputs, get_inputs, device)
        custom_kernels = extract_kernels(ModelNew, get_init_inputs, get_inputs, device)

        if base_kernels <= custom_kernels:
            return KbenchResult(error=HACKED_ERROR_MESSAGE)

        return None

    except Exception as e:
        return KbenchResult(error=str(e))


def check_error(res: KernelExecResult) -> str:
    meta = res.metadata

    if not res.compiled:
        return str(meta.get("compilation_error", ""))

    if res.correctness:
        return ""

    errors = []
    trials = meta.get("correctness_trials")
    max_diff = meta.get("max_difference")
    avg_diff = meta.get("avg_difference")

    for key, val in meta.items():
        if key.endswith("error"):
            errors.append(f"{key}: {val}")
            continue

        if key.endswith("issue"):
            parts = [f"{key}: {val}"]
            if trials is not None:
                parts.append(f"success rate: {trials}")
            if max_diff is not None:
                parts.append(f"max diff: {max_diff}")
            if avg_diff is not None:
                parts.append(f"avg diff: {avg_diff}")
            errors.append(", ".join(parts))

    return "\n".join(errors)


WORKLOAD_TEMPLATE = r"""
if __name__ == "__main__":
    torch.set_default_device("cuda")
    model = ModelNew(*get_init_inputs())
    model.eval()
    inputs = get_inputs()
    with torch.no_grad():
        for _ in range(5):
            _ = model(*inputs)
    with torch.no_grad():
        torch.cuda.nvtx.range_push("cugedit")
        _ = model(*inputs)
        torch.cuda.nvtx.range_pop()
"""


def analyze(problem: str, code: str, valid: bool = True):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        workload_file = tmp_path / "workload.py"
        workload_file.write_text(problem + "\n" + code + "\n" + WORKLOAD_TEMPLATE)
        return Analyst.analyze([sys.executable, workload_file], tmp_path, valid)


@return_asdict
def evaluate(program_path: os.PathLike, problem_path: os.PathLike, perf: bool = False):
    try:
        problem = Path(problem_path).read_text()
        custom_code = Path(program_path).read_text()

        if err := check_cuda_module(custom_code):
            return err

        test_cases = {
            "base": re.sub(r"\bModel\b", "ModelNew", problem),
            "custom": custom_code,
        }
        result: dict[str, KernelExecResult] = {}
        device = pick_idle_gpu()
        for name, code in test_cases.items():
            res = eval_kernel_against_ref(
                problem,
                code,
                device=device,
                measure_performance=True,
                num_correct_trials=8,
                num_perf_trials=32,
            )
            if error := check_error(res):
                report = analyze(problem, code, False) if perf and res.compiled else ""
                return KbenchResult(error=f"[Error @ {name}] " + error, report=report)
            result[name] = res

        if err := is_kernel_hacked(problem, custom_code):
            return err

        report = analyze(problem, code) if perf else ""
        return KbenchResult(
            combined_score=result["base"].runtime / result["custom"].runtime,
            report=report,
            base_runtime_stats=result["base"].runtime_stats,
            custom_runtime_stats=result["custom"].runtime_stats,
        )
    except Exception as e:
        return KbenchResult(error=str(e))


if __name__ == "__main__":
    cwd = Path(__file__).parent
    # program_path = cwd / "test/l3_3_init.py"
    program_path = cwd / "test/l3_3_custom.py"
    problem_path = cwd / "kernelbench/KernelBench/level3/3_DeepNarrowMLP.py"
    print(check_cuda_module(program_path.read_text()))
    print(is_kernel_hacked(problem_path.read_text(), program_path.read_text()))
    pp(evaluate(program_path, problem_path, True))
