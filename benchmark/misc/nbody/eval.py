import importlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from pyanalyst import *

from cudanalyst.result import *

TEST_N_BODIES = 100
EVAL_N_BODIES = 4096


@dataclass
class NBodyResult(Result):
    t_base: float = 0.0
    t_custom: float = 0.0


def prepare_workspace(program_path, problem_dir):
    tmp = Path(tempfile.mkdtemp())

    shutil.copytree(problem_dir, tmp, dirs_exist_ok=True)
    shutil.copy2(program_path, tmp / "sol.py")

    return tmp


def benchmark(fn, positions, weights, repeat=5, warmup=2):
    for _ in range(warmup):
        fn(positions, weights)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(positions, weights)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times)


@return_asdict
def evaluate(
    program_path: os.PathLike, problem_dir: os.PathLike, mask: int = 0
) -> NBodyResult:
    tmp = prepare_workspace(program_path, problem_dir)

    sys.path.insert(0, str(tmp))

    try:
        sol = importlib.import_module("sol")
        sol_init = importlib.import_module("sol_init")
        main = importlib.import_module("main")

        importlib.reload(sol)
        importlib.reload(sol_init)
        importlib.reload(main)

        run_custom = sol.run_custom
        run_base = sol_init.run_custom

        ok, msg = main.sanity(run_custom)
        if not ok:
            positions, weights = main.make_nbody_samples(TEST_N_BODIES)
            reports = analyze(mask & SET_DEBUG, run_custom, positions, weights)
            return NBodyResult(error=msg, reports=reports)

        positions, weights = main.make_nbody_samples(EVAL_N_BODIES)
        if reports := analyze(mask & SET_DEBUG, run_custom, positions, weights):
            return NBodyResult(error=reports["debug"], reports=reports)

        t_base = benchmark(run_base, positions, weights)
        t_custom = benchmark(run_custom, positions, weights)

        speedup = t_base / t_custom
        reports = analyze(
            mask,
            run_custom,
            positions,
            weights,
        )

        return NBodyResult(
            combined_score=speedup, reports=reports, t_base=t_base, t_custom=t_custom
        )

    finally:
        sys.path.pop(0)


if __name__ == "__main__":
    res = evaluate("./src/sol_init.py", "./src", SET_ANLZ | SET_PERF)
    print(res)
