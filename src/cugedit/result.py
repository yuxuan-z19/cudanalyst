from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import wraps
from typing import Final, final


class Status(str, Enum):
    FAIL = "fail"
    COMPILE = "compile"
    PASS = "pass"
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


@dataclass
class Result:
    combined_score: float = ResultConstant.INVALID_FLOAT
    status: Status = Status.FAIL
    error: str = None


def return_asdict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if is_dataclass(res):
            return asdict(res)
        return res

    return wrapper


def check_fast(speedup: float, tol: float = 0.1) -> Status:
    return Status.FAST if speedup > (1 + tol) else Status.PASS
