import os
import shutil
import statistics as stats
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cugedit.result import (
    Result,
    ResultConstant,
    ResultMeta,
    Status,
    check_fast,
    return_asdict,
)
from cugedit.utils import make_gpu_env, pick_idle_gpu, strip_codeblock

SUITE_ROOT = Path(__file__).parent
COMMON_DIR = SUITE_ROOT / "common"
SRC_DIR = SUITE_ROOT / "src"

NUM_RUNS = 10


@dataclass
class PolyMeta(ResultMeta):
    med_gpu_sec: float = ResultConstant.INVALID_FLOAT
    med_mismatch_cnt: int = ResultConstant.INVALID_INT
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


def execute(
    program_path: Path, task_dir: Path, gpu_id: int = 0, num_runs: int = 1
) -> PolyMeta:
    (task_dir / "sol.cu").write_text(strip_codeblock(program_path.read_text()))

    env = make_gpu_env(gpu_id)

    try:
        subprocess.run(
            ["make"],
            cwd=task_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        return PolyMeta(error=e.stderr)

    exe_path = task_dir / f"{task_dir.name}.exe"
    gpu_sec_list = []
    cpu_sec_list = []
    mismatch_cnt_list = []

    for _ in range(num_runs):
        try:
            run_proc = subprocess.run(
                [str(exe_path)],
                cwd=task_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                env=env,
            )
            result = parse_result(run_proc.stdout)
            gpu_sec = result.get("GPU_Seconds", float("-inf"))
            cpu_sec = result.get("CPU_Seconds", float("-inf"))
            mismatch_cnt = result.get("Mismatch_Count", 0)

            gpu_sec_list.append(gpu_sec)
            cpu_sec_list.append(cpu_sec)
            mismatch_cnt_list.append(mismatch_cnt)
        except subprocess.CalledProcessError as e:
            return PolyMeta(Status.COMPILE, e.stderr)

    med_gpu_sec = stats.median(gpu_sec_list)
    med_mismatch_cnt = stats.median(mismatch_cnt_list)

    error = None
    status = Status.PASS
    if (
        any(sec == float("-inf") for sec in gpu_sec_list + cpu_sec_list)
        or max(mismatch_cnt_list) > 3
    ):
        error = ResultConstant.ERR_MISMATCH
        status = Status.COMPILE

    return PolyMeta(
        status, error, med_gpu_sec, med_mismatch_cnt, gpu_sec_list, mismatch_cnt_list
    )


@return_asdict
def evaluate(program_path: Path, problem_dir: Path, perf: bool = False):
    program_path = Path(program_path)
    problem_dir = Path(problem_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        base_task_dir = tmpdir_path / "base" / problem_dir.name
        custom_task_dir = tmpdir_path / "custom" / problem_dir.name

        os.symlink(COMMON_DIR, tmpdir_path / "common", target_is_directory=True)
        shutil.copytree(problem_dir, base_task_dir)
        shutil.copytree(problem_dir, custom_task_dir)

        gpu_id = pick_idle_gpu()

        # * check correctness
        base_path = base_task_dir / "sol.init.cu"
        base_result = execute(base_path, base_task_dir, gpu_id)
        if base_result.error:
            return PolyResult(
                status=base_result.status,
                error=f"Base build/run failed: {base_result.error}",
            )

        custom_result = execute(program_path, custom_task_dir, gpu_id)
        if custom_result.error:
            return PolyResult(
                status=custom_result.status,
                error=f"Custom build/run failed: {custom_result.error}",
            )

        if custom_result.med_mismatch_cnt > base_result.med_mismatch_cnt:
            return PolyResult(status=Status.COMPILE, error=ResultConstant.ERR_MISMATCH)

        # * eval perf
        base_result = execute(base_path, base_task_dir, gpu_id, NUM_RUNS)
        custom_result = execute(program_path, custom_task_dir, gpu_id, NUM_RUNS)

        speedup = base_result.med_gpu_sec / custom_result.med_gpu_sec
        return PolyResult(
            combined_score=speedup,
            status=check_fast(speedup),
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
