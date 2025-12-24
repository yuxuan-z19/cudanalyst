import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from cugedit.helper import *
from cugedit.result import *

NITER = 4
SRC_DIR = Path(__file__).parent / "src"


@dataclass
class XsbenchMeta(ResultMeta):
    med_lookup: int = ResultConstant.INVALID_INT
    lookup_list: list[int] = field(default_factory=list)


@dataclass
class XsbenchResult(Result):
    base_result: XsbenchMeta = None
    custom_result: XsbenchMeta = None


def get_lookup_rate(raw: str):
    is_valid = bool(re.search(r"Verification checksum:\s*\d+\s*\(Valid\)", raw))
    lookups = ResultConstant.INVALID_INT
    if is_valid and (m2 := re.search(r"Lookups/s:\s*([\d,]+)", raw)):
        lookups = m2.group(1).replace(",", "")
    return is_valid, int(lookups)


@return_asdict
def evaluate(program_path: os.PathLike):
    program_path = Path(program_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        dst_path = tmp_path / "src"
        shutil.copytree(SRC_DIR, dst_path)

        (dst_path / "Solution.cu").write_text(strip_codeblock(program_path.read_text()))

        gpu_id = pick_idle_gpu()

        try:
            run_cmd(["make"], dst_path, gpu_id)
        except subprocess.CalledProcessError as e:
            return XsbenchResult(error=parse_cmd_failure(e))

        def _exec_impl(cmd: list[str], niter: int = 1):
            lookup_list = []
            for _ in range(niter):
                try:
                    proc = run_cmd(cmd, dst_path, gpu_id)
                except Exception as e:
                    return XsbenchMeta(
                        status=Status.COMPILE, error=parse_cmd_failure(e, Stage.RUN)
                    )
                is_valid, lookup = get_lookup_rate(proc.stdout)
                if not is_valid:
                    return XsbenchMeta(
                        status=Status.COMPILE,
                        error=parse_cmd_failure(
                            ResultConstant.ERR_MISMATCH, Stage.VERIFY
                        ),
                    )
                lookup_list.append(lookup)
            return XsbenchMeta(
                med_lookup=stats_med(lookup_list), lookup_list=lookup_list
            )

        test_cases = {
            "base": ["./XSBench.exe", "-m", "event", "-k", "6"],
            "custom": ["./XSBench.exe", "-m", "event", "-k", "7"],
        }

        # * check correctness
        for _, cmd in test_cases.items():
            if (res := _exec_impl(cmd)) and res.status != Status.PASS:
                return XsbenchResult(status=res.status, error=res.error)

        # * eval perf
        base_result = _exec_impl(test_cases["base"], NITER)
        custom_result = _exec_impl(test_cases["custom"], NITER)

        score = custom_result.med_lookup / base_result.med_lookup
        return XsbenchResult(
            combined_score=score, base_result=base_result, custom_result=custom_result
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
