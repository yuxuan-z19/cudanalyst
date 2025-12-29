import json
import os
from collections import defaultdict
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Final, final


class Status(str, Enum):
    FAIL = "fail"
    COMPILE = "compile"
    PASS = "pass"

    # only be set with check_fast()
    FAST = "fast"


@final
class ResultConstant:
    INVALID_INT: Final[int] = 0
    INVALID_FLOAT: Final[float] = float("-inf")

    ERR_MISMATCH: Final[str] = "Mismatch output, implementation error"
    ERR_FAULT: Final[str] = "Execution fault"


@dataclass
class ResultMeta:
    status: Status = Status.FAIL
    error: str = None

    def __post_init__(self):
        if self.error is None:
            self.status = Status.PASS


@dataclass
class Result:
    combined_score: float = ResultConstant.INVALID_FLOAT
    status: Status = Status.FAIL
    error: str = None

    tol: InitVar[float] = 0.12

    def __post_init__(self, tol):
        if self.combined_score != ResultConstant.INVALID_FLOAT:
            self.status = (
                Status.FAST if self.combined_score > (1 + tol) else Status.PASS
            )


def return_asdict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if is_dataclass(res):
            return asdict(res)
        return res

    return wrapper


def get_evolve_stats(
    output_dir: os.PathLike,
    ckpt_interval: int | None = None,
):
    if ckpt_interval is not None and ckpt_interval <= 0:
        raise ValueError("ckpt_interval must be None or a positive integer")

    base = Path(output_dir)
    if base.name != "checkpoints":
        base = base / "checkpoints"

    if not base.exists():
        raise FileNotFoundError(base)

    stats = defaultdict(
        lambda: {
            "total": 0,
            "scores": [],
            **{s.value: 0 for s in Status},
        }
    )

    for sample in base.rglob("programs/*.json"):
        with open(sample) as f:
            d = json.load(f)

        metrics = d["metrics"]
        status = metrics["status"]

        if ckpt_interval is None:
            key = d.get("generation", 0)
        else:
            key = d.get("iteration_found", 0) // ckpt_interval

        s = stats[key]
        s["total"] += 1
        s[status] += 1
        s["scores"].append(metrics["combined_score"])

    return stats
