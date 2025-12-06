import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from pprint import pp

from cugedit.analyst import Analyst
from cugedit.config import *
from cugedit.utils import pick_idle_gpu, strip_codeblock

ITER = 4


@dataclass
class XsbenchResult(Result):
    base_lookups: int = INVALID_INT
    custom_lookups: int = INVALID_INT


def get_lookup_rate(raw: str):
    is_valid = bool(re.search(r"Verification checksum:\s*\d+\s*\(Valid\)", raw))
    lookups_s = INVALID_INT
    if is_valid:
        m2 = re.search(r"Lookups/s:\s*([\d,]+)", raw)
        if m2:
            lookups_s = m2.group(1).replace(",", "")
    return int(lookups_s)


@return_asdict
def evaluate(program_path: os.PathLike, perf: bool = False):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        code = strip_codeblock(Path(program_path).read_text())

        src_path = Path(__file__).parent / "src"
        dst_path = tmp_path / "src"
        shutil.copytree(src_path, dst_path)

        sol_file = dst_path / "Solution.cu"
        sol_file.write_text(code)

        make_proc = subprocess.run(
            ["make"], cwd=dst_path, capture_output=True, text=True
        )
        if make_proc.returncode != 0:
            return Result(error="make failed! " + make_proc.stderr)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(pick_idle_gpu())

        def _run_and_score(cmd):
            total = 0
            for _ in range(ITER):
                proc = subprocess.run(
                    cmd, capture_output=True, cwd=dst_path, env=env, text=True
                )
                if proc.returncode != 0:
                    report = Analyst.analyze(cmd, dst_path, False) if perf else ""
                    return None, Result(
                        error="exec failed! " + proc.stderr, report=report
                    )
                total += get_lookup_rate(proc.stdout)
            return total / ITER, None

        test_cases = {
            "base": ["./XSBench.exe", "-m", "event", "-k", "6"],
            "custom": ["./XSBench.exe", "-m", "event", "-k", "7"],
        }

        scores = {}
        for name, cmd in test_cases.items():
            score, err = _run_and_score(cmd)
            if err:
                return err
            scores[name] = score

        report = Analyst.analyze(test_cases["custom"], dst_path) if perf else ""
        ratio = scores["custom"] / scores["base"] if scores["custom"] > 0 else 0
        return Result(combined_score=ratio, report=report)


if __name__ == "__main__":
    pp(evaluate(Path(__file__).parent / "./src/Solution.init.cu"))
