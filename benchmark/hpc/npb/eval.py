import os
import shutil
import string
import tempfile
from pathlib import Path

from cugedit import AnalysisCfg, ToolContext, planning
from cugedit.helper import *
from cugedit.helper.stat import stats_med
from cugedit.helper.text import extract_codeblock
from cugedit.result import *

SUITE_ROOT = Path(__file__).parent / "src"
COMMON_INC_DIRS = ["common", "config", "sys", "Makefile"]

BUILD_TIMEOUT = 3 * 60  # 3 min
TEST_TIMEOUT = 5 * 60  # 5 min
EVAL_TIMEOUT = 15 * 60  # 15 min

TEST_CLASS = ["S", "W"]
EVAL_CLASS = list(string.ascii_uppercase[:3])

NUM_RUNS = 10


@dataclass
class NPBMeta(ResultMeta):
    med_mops_list: list[float] = field(default_factory=list)
    mops_res_list: list[list[float]] = field(default_factory=dict)


@dataclass
class NPBResult(Result):
    mops_improve_list: list[float] = field(default_factory=list)
    base_result: NPBMeta = None
    custom_result: NPBMeta = None


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
        result[key] = float(value) if "." in value else int(value)
    return result


def exec_impl(
    task_name: str,
    task_dir: Path,
    case: str,
    gpu_id: int,
    num_runs: int,
    timeout: float,
    config: AnalysisCfg = None,
) -> NPBMeta:
    ctx = ToolContext(code_path=(task_dir / task_name / "sol.cu"), cwd=task_dir)

    make_cmd = ["make", task_name, f"CLASS={case}"]
    ctx.cmd = make_cmd
    try:
        run_cmd(make_cmd, task_dir, Stage.BUILD, gpu_id, BUILD_TIMEOUT)
    except ExecError as e:
        return NPBMeta(error=str(e), reports=planning(config, ctx, str(e)))

    exec_cmd = [(task_dir / f"bin/{task_name.lower()}.{case}").as_posix()]
    ctx.cmd = exec_cmd

    mops_list = []
    for _ in range(num_runs):
        try:
            run_proc = run_cmd(exec_cmd, task_dir, Stage.RUN, gpu_id, timeout)
            result = parse_result(run_proc.stdout)
            if result.get("verified", 0) <= 0:
                raise ExecError(Stage.VERIFY, exec_cmd, ExecFailReason.VERIFY_MISMATCH)
        except ExecError as e:
            return NPBMeta(
                Status.COMPILE, error=str(e), reports=planning(config, ctx, str(e))
            )

        mops_list.append(result.get("Mops", Score.INVALID_FLOAT))

    med_mops = stats_med(mops_list)
    return NPBMeta(
        reports=planning(config, ctx),
        med_mops_list=[med_mops],
        mops_res_list=[mops_list],
    )


def execute(
    program_path: Path,
    task_name: str,
    task_dir: Path,
    gpu_id: int = 0,
    num_runs: int = 1,
    config: AnalysisCfg = None,
):
    (task_dir / task_name / "sol.cu").write_text(
        extract_codeblock(program_path.read_text())
    )

    def _exec_case(case: str, timeout: float, use_analyst: bool = False):
        return exec_impl(
            task_name,
            task_dir,
            case,
            gpu_id,
            num_runs,
            timeout,
            config if use_analyst else None,
        )

    def _run_or_plan(case: str):
        timeout = TEST_TIMEOUT if case in TEST_CLASS else EVAL_TIMEOUT
        res = _exec_case(case, timeout)
        if res.status == Status.PASS:
            return res
        return _exec_case(case, timeout, True)

    for case in TEST_CLASS:
        res = _run_or_plan(case)
        if res.status != Status.PASS:
            return res

    med_mops_list = []
    mops_res_list = []
    for case in EVAL_CLASS:
        res = _run_or_plan(case)
        if res.status != Status.PASS:
            return res

        med_mops_list.append(res.med_mops_list[0])
        mops_res_list.append(res.mops_res_list[0])

    reports = []
    if num_runs >= NUM_RUNS:
        reports = _exec_case(EVAL_CLASS[-1], EVAL_TIMEOUT, True).reports

    return NPBMeta(
        reports=reports, med_mops_list=med_mops_list, mops_res_list=mops_res_list
    )


@return_asdict
def evaluate(
    program_path: os.PathLike, problem_dir: os.PathLike, config: AnalysisCfg = None
):
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
                base_result=base_result,
            )

        custom_result = execute(
            program_path, task_name, custom_dir, gpu_id, config=config
        )
        if custom_result.status != Status.PASS:
            return NPBResult(
                status=custom_result.status,
                error=f"Custom build/run failed: {custom_result.error}",
                reports=custom_result.drain_reports(),
                base_result=base_result,
                custom_result=custom_result,
            )

        # * eval perf
        base_result = execute(base_program, task_name, base_dir, gpu_id, NUM_RUNS)
        custom_result = execute(
            program_path, task_name, custom_dir, gpu_id, NUM_RUNS, config
        )

        base_med_mops = base_result.med_mops_list
        custom_med_mops = custom_result.med_mops_list
        mops_improv_list = [c / b for c, b in zip(custom_med_mops, base_med_mops)]

        return NPBResult(
            combined_score=min(mops_improv_list),
            reports=custom_result.drain_reports(),
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
