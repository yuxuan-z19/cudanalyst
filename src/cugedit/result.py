import json
import os
from collections import Counter, defaultdict
from dataclasses import InitVar, asdict, dataclass, field, is_dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Final, final

from .helper import iter_program_json, resolve_ckpt_dir
from .module.module import Report


class Status(str, Enum):
    FAIL = "fail"
    COMPILE = "compile"
    PASS = "pass"

    # only be set with check_fast()
    FAST = "fast"

    @property
    def rank(self):
        _ranks = {Status.FAIL: 0, Status.COMPILE: 1, Status.PASS: 2, Status.FAST: 3}
        return _ranks.get(self, -1)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.rank < other.rank
        return NotImplemented


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
    reports: list[Report] = field(default_factory=list)

    def __post_init__(self):
        if self.error is None:
            self.status = Status.PASS

    def drain_reports(self):
        reports, self.reports = self.reports, []
        return reports


@dataclass
class Result:
    combined_score: float = ResultConstant.INVALID_FLOAT
    status: Status = Status.FAIL
    error: str = None
    reports: list[Report] = field(default_factory=list)

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
    ckpt_dir: os.PathLike,
    subdir_name: str = None,
    ckpt_interval: int = None,
    collect_scores: bool = False,
    collect_gen_dist: bool = False,
) -> dict[int, dict]:
    if ckpt_interval is not None and ckpt_interval <= 0:
        raise ValueError(
            f"ckpt_interval must be a positive integer or None, got {ckpt_interval}"
        )

    ckpt_base = Path(ckpt_dir)

    def make_stat():
        stat = {"total": 0, **{s.value: 0 for s in Status}}
        if collect_scores:
            stat["scores"] = []
        if collect_gen_dist and ckpt_interval:
            stat["gen"] = Counter()
        return stat

    stats: defaultdict[int, dict] = defaultdict(make_stat)

    for _, data in iter_program_json(ckpt_base, subdir_name):
        metrics = data["metrics"]
        status = metrics["status"]

        key = (
            data.get("generation", 0)
            if ckpt_interval is None
            else data.get("iteration_found", 0) // ckpt_interval
        )
        s = stats[key]

        s["total"] += 1
        s[status] += 1

        if collect_scores:
            s["scores"].append(metrics["combined_score"])
        if collect_gen_dist and ckpt_interval:
            s["gen"][data.get("generation", 0)] += 1

    return dict(sorted(stats.items()))


def groupby_gen(src_ckpt_dir: os.PathLike, dst_gen_dir: os.PathLike) -> None:
    ckpt_base = resolve_ckpt_dir(src_ckpt_dir)
    if not ckpt_base.exists():
        raise FileNotFoundError(f"Source checkpoint directory not found: {ckpt_base}")

    dst_base = Path(dst_gen_dir)
    dst_base.mkdir(parents=True, exist_ok=True)

    id_to_prompts = {}
    gen_to_samples = defaultdict(list)

    for _, data in iter_program_json(ckpt_base):
        gen = data.get("generation", 0)
        sample_id = data.get("id")
        parent_id = data.get("parent_id")

        if parent_id and "prompts" in data:
            id_to_prompts[parent_id] = data["prompts"]

        gen_to_samples[gen].append(data)

    count_saved = 0
    count_skipped = 0

    for gen, samples in gen_to_samples.items():
        gen_folder = dst_base / f"gen{gen}"

        for data in samples:
            sample_id = data.get("id")

            if sample_id not in id_to_prompts:
                count_skipped += 1
                continue

            data["prompts"] = id_to_prompts[sample_id]

            gen_folder.mkdir(parents=True, exist_ok=True)

            output_path = gen_folder / f"{sample_id}.json"
            output_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            count_saved += 1

    print(f"saved {count_saved} + skipped {count_skipped}")
