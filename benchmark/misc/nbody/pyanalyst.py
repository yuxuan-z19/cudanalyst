"""
A lightweight analysis layer for pure Python-written program analysis.
"""

import cProfile
import linecache
import pstats
import traceback
from collections.abc import Callable
from functools import partial
from typing import Any

from line_profiler import LineProfiler

from cudanalyst.module.config import ModuleCfg
from cudanalyst.module.module import Planner, Report
from cudanalyst.module.prompts import PLAN_PROMPT

SET_PERF = 1 << 0  # 0001
SET_ANLZ = 1 << 1  # 0010
SET_DEBUG = 1 << 2  # 0100


def run_debug(target: partial):
    try:
        _ = target()
        return None
    except Exception as e:
        error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return error_msg


def run_perf(target: partial):
    pr = cProfile.Profile()
    pr.enable()
    target()
    pr.disable()

    ps = pstats.Stats(pr)
    stats = {}
    for func_info, func_stats in ps.stats.items():
        cc, nc, tt, ct, callers = func_stats
        stats[str(func_info)] = {
            "callcount": cc,
            "reccallcount": nc,
            "totaltime_ms": tt * 1e3,
            "cumtime_ms": ct * 1e3,
        }

    return stats


def run_anlz(target: partial):
    lp = LineProfiler()
    lp.add_function(target.func)
    lp.enable_by_count()
    target()
    lp.disable_by_count()

    stats = lp.get_stats()
    formatted = {}

    for (filename, start_lineno, func_name), timings in stats.timings.items():
        total_time = sum(t[2] for t in timings)

        lines = []
        for lineno, nhits, time_ns in timings:
            code = linecache.getline(filename, lineno).strip()
            time_ms = time_ns / 1e6
            percent = (time_ns / total_time * 100) if total_time else 0

            lines.append(
                {
                    "lineno": lineno,
                    "hits": nhits,
                    "time_ms": time_ms,
                    "percent": percent,
                    "code": code,
                }
            )

        formatted[f"{func_name}@{filename}:{start_lineno}"] = lines

    return formatted


def analyze(mask: int, func: Callable, *args, **kwargs):
    results = {}

    if mask == 0:
        return results

    target = partial(func, *args, **kwargs)

    if mask & SET_DEBUG:
        if err := run_debug(target):
            results["debug"] = err
            return results

    if mask & SET_ANLZ:
        results["anlz"] = run_anlz(target)

    if mask & SET_PERF:
        results["perf"] = run_perf(target)

    return results
