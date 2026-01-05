import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .exec import Status


def drain_plans(metrics: dict[str, Any]):
    reports = metrics.pop("reports", [])
    return metrics, {
        r["name"]: f"Plan decision: {r['summary'] or r['feedback']}\n\n"
        for r in reports
    }


def resolve_ckpt_dir(ckpt_dir: os.PathLike) -> Path:
    base = Path(ckpt_dir)
    return base if base.name == "checkpoints" else base / "checkpoints"


def iter_program_json(
    ckpt_base: os.PathLike, subdir_name: str = None
) -> Iterator[tuple[Path, dict]]:
    base = Path(ckpt_base)
    sample_pattern = f"{subdir_name}/*.json" if subdir_name else "**/*.json"
    seen_ids = set()
    for json_file in base.rglob(sample_pattern):
        if json_file.name in {"metadata.json", "best_program_info.json"}:
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            sample_id = data.get("id")
            if sample_id is None or sample_id in seen_ids:
                continue

            seen_ids.add(sample_id)

            yield json_file, data

        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {json_file}: {e}")


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
