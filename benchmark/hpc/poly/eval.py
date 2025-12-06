import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from cugedit.config import *
from cugedit.utils import strip_codeblock

SUITE_ROOT = Path(__file__).parent
COMMON_DIR = SUITE_ROOT / "common"
SRC_DIR = SUITE_ROOT / "src"

NUM_RUNS = 10


@dataclass
class PolyMeta:
    error: str = None
    avg_gpu_sec: float = INVALID_FLOAT
    avg_mismatch_cnt: int = INVALID_INT
    gpu_sec_list: list[float] = field(default_factory=list)
    mismatch_cnt_list: list[float] = field(default_factory=list)


@dataclass
class PolyResult(Result):
    base_result: PolyMeta = None
    custom_result: PolyMeta = None


def parse_result(output: str):
    result = {}
    for line in output.strip().splitlines():
        key, value = line.split("=")
        result[key] = float(value) if "." in value else int(value)
    return result


def execute(program_path: Path, task_dir: Path) -> PolyMeta:
    (task_dir / "sol.cu").write_text(strip_codeblock(program_path.read_text()))

    try:
        subprocess.run(
            ["make"],
            cwd=task_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return PolyMeta(error=e.stderr)

    exe_path = task_dir / f"{task_dir.name}.exe"
    gpu_sec_list = []
    cpu_sec_list = []
    mismatch_cnt_list = []

    for _ in range(NUM_RUNS):
        try:
            run_proc = subprocess.run(
                [str(exe_path)],
                cwd=task_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            result = parse_result(run_proc.stdout)
            gpu_sec = result.get("GPU_Seconds", float("-inf"))
            cpu_sec = result.get("CPU_Seconds", float("-inf"))
            mismatch_cnt = result.get("Mismatch_Count", 0)

            gpu_sec_list.append(gpu_sec)
            cpu_sec_list.append(cpu_sec)
            mismatch_cnt_list.append(mismatch_cnt)
        except subprocess.CalledProcessError as e:
            return PolyMeta(error=e.stderr)

    avg_gpu_sec = sum(gpu_sec_list) / len(gpu_sec_list)
    avg_mismatch_cnt = sum(mismatch_cnt_list) / len(mismatch_cnt_list)

    error_msg = None
    if (
        any(sec == float("-inf") for sec in gpu_sec_list + cpu_sec_list)
        or max(mismatch_cnt_list) > 3
    ):
        error_msg = "Mismatch output"

    return PolyMeta(
        error=error_msg,
        avg_gpu_sec=avg_gpu_sec,
        avg_mismatch_cnt=avg_mismatch_cnt,
        gpu_sec_list=gpu_sec_list,
        mismatch_cnt_list=mismatch_cnt_list,
    )


@return_asdict
def evaluate(program_path: Path, problem_dir: Path, perf: bool = False):
    program_path = Path(program_path)
    problem_dir = Path(problem_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        common_src = COMMON_DIR
        base_task_dir = tmpdir_path / "base" / problem_dir.name
        custom_task_dir = tmpdir_path / "custom" / problem_dir.name

        shutil.copytree(common_src, tmpdir_path / "common")
        shutil.copytree(problem_dir, base_task_dir)
        shutil.copytree(problem_dir, custom_task_dir)

        base_path = base_task_dir / "sol.init.cu"
        base_result = execute(base_path, base_task_dir)
        if base_result.error:
            return PolyResult(error=f"Base build/run failed: {base_result.error}")

        custom_result = execute(program_path, custom_task_dir)
        if custom_result.error:
            return PolyResult(error=f"Custom build/run failed: {custom_result.error}")

        if custom_result.avg_mismatch_cnt > base_result.avg_mismatch_cnt:
            error = "Mismatch output"
            combined_score = float("-inf")
        else:
            error = None
            combined_score = base_result.avg_gpu_sec / custom_result.avg_gpu_sec

        return PolyResult(
            combined_score=combined_score,
            error=error,
            base_result=base_result,
            custom_result=custom_result,
        )


if __name__ == "__main__":
    from tqdm.rich import tqdm

    for workload_name in tqdm(os.listdir(SRC_DIR)):
        workload_path = SRC_DIR / workload_name
        init_kernel_path = workload_path / "sol.init.cu"

        if not init_kernel_path.exists():
            print(f"Skipped {workload_name}, no sol.init.cu found.")
            continue

        result = evaluate(init_kernel_path, workload_path)
        if result:
            print(f"--- Result for {workload_name} ---")
            print(result)
        else:
            break
