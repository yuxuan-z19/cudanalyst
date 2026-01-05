from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from functools import wraps

from .helper.exec import Score, Status
from .module.module import Report


@dataclass
class ResultMeta:
    status: Status = Status.FAIL
    error: str = None
    reports: list[Report] = field(default_factory=list)

    def __post_init__(self):
        if self.error is None:
            self.status = Status.PASS

    def drain_reports(self):
        reports, self.reports = self.reports, []
        return reports


@dataclass
class Result:
    combined_score: float = Score.INVALID_FLOAT
    status: Status = Status.FAIL
    error: str = None
    reports: list[Report] = field(default_factory=list)

    tol: InitVar[float] = 0.12

    def __post_init__(self, tol):
        if self.combined_score != Score.INVALID_FLOAT:
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
