import os
import shutil
import string
import subprocess
import tempfile
from pathlib import Path

from cugedit.helper import *
from cugedit.result import *

SUITE_ROOT = Path(__file__).parent / "src"
COMMON_INC_DIRS = ["common", "config", "sys", "Makefile"]
TESTCLASS = ["S", "W"]
DATACLASS = list(string.ascii_uppercase[:5])
NUM_RUNS = 3


@dataclass
class NPBMeta(ResultMeta):
    med_mops_list: list[float] = field(default_factory=list)
    mops_res_list: list[list[float]] = field(default_factory=dict)


@dataclass
class NPBResult(Result):
    mops_improve_list: list[float] = field(default_factory=list)
    base_result: ResultMeta = None
    custom_result: ResultMeta = None


def prepare_task_env(tmpdir_path: Path, problem_dir: Path, name: str = "base") -> Path:
    base_dir = tmpdir_path / name
    base_task_dir = base_dir / problem_dir.stem
    shutil.copytree(problem_dir, base_task_dir)
    (base_dir / "bin").mkdir(exist_ok=True)
    for inc in COMMON_INC_DIRS:
        src = SUITE_ROOT / inc
        dst = base_dir / inc
        os.symlink(src, dst, target_is_directory=(inc != "Makefile"))
    return base_dir


def parse_result(output: str):
    result = {}
    for line in output.strip().splitlines():
        key, value = line.split("=")
        result[key] = float(value) if "." in value else bool(value)
    return result


def exec_impl(
    task_name: str, task_dir: Path, data_class: str, gpu_id: int = 0, num_runs: int = 1
) -> NPBMeta:
    mops_list = []
    try:
        run_cmd(["make", task_name, f"CLASS={data_class}"], task_dir, gpu_id)
    except subprocess.CalledProcessError as e:
        return NPBMeta(error=parse_cmd_failure(e))

    for _ in range(num_runs):
        try:
            run_proc = run_cmd(
                [f"./bin/{task_name.lower()}.{data_class}"], task_dir, gpu_id
            )
        except subprocess.CalledProcessError as e:
            return NPBMeta(Status.COMPILE, error=parse_cmd_failure(e, stage=Stage.RUN))

        result = parse_result(run_proc.stdout)
        if not result.get("verified", False):
            return NPBMeta(
                Status.COMPILE,
                error=parse_cmd_failure(ResultConstant.ERR_MISMATCH, Stage.VERIFY),
            )
        mops_list.append(result.get("Mops", ResultConstant.INVALID_INT))

    return NPBMeta(mops_res_list=[mops_list])


def execute(
    program_path: Path,
    task_name: str,
    task_dir: Path,
    gpu_id: int = 0,
    num_runs: int = 1,
):
    (task_dir / task_name / "sol.cu").write_text(
        strip_codeblock(program_path.read_text())
    )
    for case in TESTCLASS:
        res = exec_impl(task_name, task_dir, case, gpu_id, num_runs)
        if res.status != Status.PASS:
            return res

    med_mops_list = []
    mops_res_list = []
    for case in DATACLASS:
        res = exec_impl(task_name, task_dir, case, gpu_id, num_runs)
        if res.status != Status.PASS:
            return res
        med_mops_list.append(stats_med(res.mops_res_list[0]))
        mops_res_list.append(res.mops_res_list[0])

    return NPBMeta(med_mops_list=med_mops_list, mops_res_list=mops_res_list)


@return_asdict
def evaluate(program_path: os.PathLike, problem_dir: os.PathLike):
    program_path = Path(program_path)
    problem_dir = Path(problem_dir)
    task_name = problem_dir.stem

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        base_dir = prepare_task_env(tmpdir_path, problem_dir)
        custom_dir = prepare_task_env(tmpdir_path, problem_dir, "custom")

        gpu_id = pick_idle_gpu()

        # * check correctness
        base_program = problem_dir / "sol.init.cu"
        base_result = execute(base_program, task_name, base_dir, gpu_id)
        if base_result.status != Status.PASS:
            return NPBResult(
                status=base_result.status,
                error=f"Base build/run failed: {base_result.error}",
            )
        custom_result = execute(program_path, task_name, custom_dir, gpu_id)
        if custom_result.status != Status.PASS:
            return NPBResult(
                status=custom_result.status,
                error=f"Base build/run failed: {custom_result.error}",
            )

        # * eval perf
        base_result = execute(base_program, task_name, base_dir, gpu_id, NUM_RUNS)
        custom_result = execute(program_path, task_name, custom_dir, gpu_id, NUM_RUNS)

        base_med_mops = base_result.med_mops_list
        custom_med_mops = custom_result.med_mops_list
        mops_improv_list = [c / b for c, b in zip(custom_med_mops, base_med_mops)]
        combined_score = stats_med(mops_improv_list)

        return NPBResult(
            combined_score=combined_score,
            mops_improve_list=mops_improv_list,
            base_result=base_result,
            custom_result=custom_result,
        )


if __name__ == "__main__":
    import time
    from pprint import pprint

    from tqdm.rich import tqdm

    workload_list = [
        p.name
        for p in SUITE_ROOT.iterdir()
        if p.is_dir() and p.name not in (COMMON_INC_DIRS + ["bin"])
    ]

    for workload in tqdm(workload_list):
        print(workload)
        workload_path = SUITE_ROOT / workload
        init_kernel_path = workload_path / "sol.init.cu"

        if not init_kernel_path.exists():
            print(f"Skipped {workload}, no sol.init.cu found.")
            continue

        tstart = time.perf_counter()
        result = evaluate(init_kernel_path, workload_path)
        tend = time.perf_counter()

        if result:
            print(f"--- Result for {workload} in {tend - tstart:.4f} sec ---")
            pprint(result)
        else:
            break
