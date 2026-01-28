import os
import re
import shutil
import tempfile
from pathlib import Path

from cudanalyst import AnalysisCfg, ToolContext, planning
from cudanalyst.helper import *
from cudanalyst.helper.stat import stats_med
from cudanalyst.helper.text import extract_codeblock
from cudanalyst.result import *

NITER = 4
TIMEOUT = 5 * 60  # 5 min
SRC_DIR = Path(__file__).parent / "src"


@dataclass
class XsbenchMeta(ResultMeta):
    med_lookup: int = Score.INVALID_INT
    lookup_list: list[int] = field(default_factory=list)


@dataclass
class XsbenchResult(Result):
    base_result: XsbenchMeta = None
    custom_result: XsbenchMeta = None


def get_lookup_rate(raw: str):
    is_valid = bool(re.search(r"Verification checksum:\s*\d+\s*\(Valid\)", raw))
    lookups = Score.INVALID_INT
    if is_valid and (m2 := re.search(r"Lookups/s:\s*([\d,]+)", raw)):
        lookups = m2.group(1).replace(",", "")
    return is_valid, int(lookups)


@return_asdict
def evaluate(program_path: os.PathLike, config: AnalysisCfg = None):
    program_path = Path(program_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        dst_path = tmp_path / "src"
        shutil.copytree(SRC_DIR, dst_path)

        code_path = dst_path / "Solution.cu"
        code_path.write_text(extract_codeblock(program_path.read_text()))

        ctx = ToolContext(code_path=code_path, cwd=dst_path)

        gpu_id = pick_idle_gpu()

        make_cmd = ["make"]
        ctx.cmd = make_cmd
        try:
            run_cmd(make_cmd, dst_path, Stage.BUILD, gpu_id, TIMEOUT)
        except ExecError as e:
            return XsbenchResult(error=str(e), reports=planning(config, ctx, str(e)))

        def _exec_impl(cmd: list[str], niter: int = 1):
            lookup_list = []
            ctx.cmd = cmd

            for _ in range(niter):
                try:
                    proc = run_cmd(cmd, dst_path, Stage.RUN, gpu_id, TIMEOUT)
                    is_valid, lookup = get_lookup_rate(proc.stdout)
                    if not is_valid:
                        raise ExecError(
                            stage=Stage.VERIFY,
                            cmd=cmd,
                            reason=ExecFailReason.VERIFY_MISMATCH,
                            stdout=proc.stdout,
                        )
                except ExecError as e:
                    return XsbenchMeta(
                        status=Status.COMPILE,
                        error=str(e),
                        reports=planning(config, ctx, str(e)),
                    )

                lookup_list.append(lookup)
            return XsbenchMeta(
                med_lookup=stats_med(lookup_list), lookup_list=lookup_list
            )

        test_cmds = {
            "base": ["./XSBench.exe", "-m", "event", "-k", "6"],
            "custom": ["./XSBench.exe", "-m", "event", "-k", "7"],
        }

        # * check correctness
        for _, cmd in test_cmds.items():
            if (res := _exec_impl(cmd)) and res.status != Status.PASS:
                return XsbenchResult(
                    status=res.status, error=res.error, reports=res.drain_reports()
                )

        # * eval perf
        base_result = _exec_impl(test_cmds["base"], NITER)
        custom_result = _exec_impl(test_cmds["custom"], NITER)

        score = custom_result.med_lookup / base_result.med_lookup

        ctx.cmd = test_cmds["custom"]
        return XsbenchResult(
            combined_score=score,
            reports=planning(config, ctx),
            base_result=base_result,
            custom_result=custom_result,
        )


if __name__ == "__main__":
    import time
    from pprint import pprint

    tstart = time.perf_counter()
    result = evaluate(Path(__file__).parent / "./src/Solution.init.cu")
    tend = time.perf_counter()

    if result:
        print(f"--- Result in {tend - tstart:.4f} sec ---")
        pprint(result)
