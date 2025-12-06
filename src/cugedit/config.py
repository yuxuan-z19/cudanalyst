import re
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import wraps

INVALID_INT = 0
INVALID_FLOAT = float("-inf")


@dataclass
class Result:
    combined_score: float = INVALID_FLOAT
    error: str = None
    report: str = None


def return_asdict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if is_dataclass(res):
            return asdict(res)
        return res

    return wrapper


def check_banned_call(code: str, banned_patterns: list[str] = []) -> bool:
    return any(
        pattern.search(code)
        for pattern in [re.compile(pattern) for pattern in banned_patterns]
    )


HACKED_ERROR_MESSAGE = "Error: Use only custom CUDA kernel code. Do not use official libraries like cuBLAS, ATen, or any other high-level APIs."
