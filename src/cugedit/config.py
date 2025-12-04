import re
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import wraps


@dataclass
class Result:
    combined_score: float = field(default=float("-inf"))
    error: str = field(default="")
    report: str = field(default="")


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
