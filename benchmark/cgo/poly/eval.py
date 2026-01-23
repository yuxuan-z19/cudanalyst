import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from cugedit import AnalysisCfg, ToolContext, planning
from cugedit.helper import *
from cugedit.helper.stat import *
from cugedit.helper.text import extract_codeblock
from cugedit.result import *

SUITE_ROOT = Path(__file__).parent
COMMON_DIR = SUITE_ROOT / "common"
SRC_DIR = SUITE_ROOT / "src"

BUILD_TIMEOUT = 3 * 60
TEST_TIMEOUT = 5 * 60  # 5 min
EVAL_TIMEOUT = 15 * 60  # 15 min per run

TEST_DATASET = ["MINI"]
EVAL_DATASET = ["SMALL", "STANDARD", "LARGE", "EXTRALARGE"]

NUM_RUNS = 10


@dataclass
class PolyMeta(ResultMeta):
    med_gpu_sec: list[float] = field(default_factory=list)
    gpu_sec_per_case: list[list[float]] = field(default_factory=list)

    # * for correctness checking
    med_mismatch_cnt: list[int] = field(default_factory=list)
    mismatch_cnt_per_case: list[list[int]] = field(default_factory=list)


@dataclass
class PolyResult(Result):
    speedup_list: list[float] = field(default_factory=list)
    base_result: PolyMeta = None
    custom_result: PolyMeta = None


def parse_result(output: str):
    result = {}
    for line in output.strip().splitlines():
        key, value = line.split("=")
        result[key] = float(value) if "." in value else int(value)
    return result


def exec_impl(
    task_dir: Path,
    case: str,
    gpu_id: int,
    num_runs: int,
    timeout: float,
    config: AnalysisCfg = None,
):
    ctx = ToolContext(code_path=(task_dir / "sol.cu"), cwd=task_dir)
    make_cmd = ["make", f"DATASET={case}"]
    ctx.cmd = make_cmd

    try:
        run_cmd(make_cmd, task_dir, Stage.BUILD, gpu_id, BUILD_TIMEOUT)
    except ExecError as e:
        return PolyMeta(error=str(e), reports=planning(config, ctx, str(e)))

    exec_cmd = [(task_dir / f"{task_dir.name}.{case}").as_posix()]
    ctx.cmd = exec_cmd

    gpu_sec_list = []
    mismatch_cnt_list = []
    for _ in range(num_runs):
        try:
            run_proc = run_cmd(exec_cmd, task_dir, Stage.RUN, gpu_id, timeout)
            result = parse_result(run_proc.stdout)
            gpu_sec_list.append(result.get("GPU_Seconds", Score.INVALID_FLOAT))
            mismatch_cnt_list.append(result.get("Mismatch_Count", Score.INVALID_INT))
        except ExecError as e:
            return PolyMeta(
                Status.COMPILE, error=str(e), reports=planning(config, ctx, str(e))
            )

    med_gpu_sec = stats_med(gpu_sec_list)
    med_mismatch_cnt = stats_med(mismatch_cnt_list)
    return PolyMeta(
        reports=planning(config, ctx),
        med_gpu_sec=[med_gpu_sec],
        gpu_sec_per_case=[gpu_sec_list],
        med_mismatch_cnt=[med_mismatch_cnt],
        mismatch_cnt_per_case=[mismatch_cnt_list],
    )


def execute(
    program_path: Path,
    task_dir: Path,
    gpu_id: int = 0,
    num_runs: int = 1,
    config: AnalysisCfg = None,
    base_result: PolyMeta = None,
) -> PolyMeta:
    (task_dir / "sol.cu").write_text(extract_codeblock(program_path.read_text()))

    def _exec_case(case: str, timeout: float, use_analyst: bool = False):
        return exec_impl(
            task_dir,
            case,
            gpu_id,
            num_runs,
            timeout,
            config if use_analyst else None,
        )

    def _run_or_plan(case: str):
        timeout = TEST_TIMEOUT if case in TEST_DATASET else EVAL_TIMEOUT
        res = _exec_case(case, timeout)
        if res.status == Status.PASS:
            return res
        return _exec_case(case, timeout, True)

    for case in TEST_DATASET:
        res = _run_or_plan(case)
        if res.status != Status.PASS:
            return res

    med_gpu_sec = []
    gpu_sec_per_case = []
    med_mismatch_cnt = []
    mismatch_cnt_per_case = []

    for case in EVAL_DATASET:
        res = _run_or_plan(case)
        if res.status != Status.PASS:
            return res

        med_gpu_sec.append(res.med_gpu_sec[0])
        gpu_sec_per_case.append(res.gpu_sec_per_case[0])
        med_mismatch_cnt.append(res.med_mismatch_cnt[0])
        mismatch_cnt_per_case.append(res.mismatch_cnt_per_case[0])

    if base_result and not elemwise_equal(
        base_result.med_mismatch_cnt, med_mismatch_cnt
    ):
        e = str(ExecError(Stage.VERIFY, None, ExecFailReason.VERIFY_MISMATCH))
        return PolyMeta(
            Status.COMPILE,
            error=e,
            reports=planning(config, ToolContext(program_path, cwd=task_dir), e),
            med_mismatch_cnt=med_mismatch_cnt,
            mismatch_cnt_per_case=mismatch_cnt_per_case,
        )

    reports = []
    if num_runs >= NUM_RUNS:
        reports = _exec_case(EVAL_DATASET[-1], EVAL_TIMEOUT, True).reports

    return PolyMeta(
        reports=reports,
        med_gpu_sec=med_gpu_sec,
        gpu_sec_per_case=gpu_sec_per_case,
        med_mismatch_cnt=med_mismatch_cnt,
        mismatch_cnt_per_case=mismatch_cnt_per_case,
    )


@return_asdict
def evaluate(
    program_path: os.PathLike, problem_dir: os.PathLike, config: AnalysisCfg = None
):
    program_path = Path(program_path)
    problem_dir = Path(problem_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        base_task_dir = tmpdir_path / "base" / problem_dir.name
        custom_task_dir = tmpdir_path / "custom" / problem_dir.name

        shutil.copytree(problem_dir, base_task_dir)
        shutil.copytree(problem_dir, custom_task_dir)
        os.symlink(COMMON_DIR, tmpdir_path / "common", target_is_directory=True)

        gpu_id = pick_idle_gpu()

        # * check correctness
        base_program = base_task_dir / "sol.init.cu"
        base_result = execute(base_program, base_task_dir, gpu_id)
        if base_result.status != Status.PASS:
            return PolyResult(
                status=base_result.status,
                error=f"Base build/run failed: {base_result.error}",
                base_result=base_result,
            )

        custom_result = execute(
            program_path,
            custom_task_dir,
            gpu_id,
            config=config,
            base_result=base_result,
        )
        reports = custom_result.drain_reports()
        if custom_result.status != Status.PASS:
            return PolyResult(
                status=custom_result.status,
                error=f"Custom build/run failed: {custom_result.error}",
                reports=reports,
                base_result=base_result,
                custom_result=custom_result,
            )

        # * eval perf
        base_result = execute(base_program, base_task_dir, gpu_id, NUM_RUNS)
        custom_result = execute(program_path, custom_task_dir, gpu_id, NUM_RUNS, config)
        speedup_list = [
            b / c for b, c in zip(base_result.med_gpu_sec, custom_result.med_gpu_sec)
        ]

        return PolyResult(
            combined_score=min(speedup_list),
            reports=custom_result.drain_reports(),
            speedup_list=speedup_list,
            base_result=base_result,
            custom_result=custom_result,
        )


if __name__ == "__main__":
    import time
    from pprint import pprint

    from tqdm.rich import tqdm

    for workload_name in tqdm(os.listdir(SRC_DIR)):
        workload_path = SRC_DIR / workload_name
        init_kernel_path = workload_path / "sol.init.cu"

        if not init_kernel_path.exists():
            print(f"Skipped {workload_name}, no sol.init.cu found.")
            continue

        tstart = time.perf_counter()
        result = evaluate(init_kernel_path, workload_path)
        tend = time.perf_counter()

        if result:
            print(f"--- Result for {workload_name} in {tend - tstart:.4f} sec ---")
            pprint(result)
        else:
            break
