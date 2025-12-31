import json
import os
import random
import re
import statistics as stats
import subprocess
import time
import warnings
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pynvml
import tiktoken

try:
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    TORCH_INCLUDES = [f"-I{path}" for path in include_paths("cuda")]
    TORCH_INCLUDES.append(sysconfig.get_path("include", scheme="posix_prefix"))
except:
    TORCH_INCLUDES = []


# Reference: https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/utils/code_utils.py#L95
def extract_codeblock(text: str) -> str:
    text = text.strip()
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return text
    return max(matches, key=len).strip()


def render_feedback_md(feedback: dict[str, Any]) -> str:
    sections = []
    for key, value in feedback.items():
        pretty_json = json.dumps(value, indent=2, ensure_ascii=False)
        sections.append(f"### {key}\n\n ```json\n{pretty_json}\n```")
    return "\n\n".join(sections)


def stats_med(result: list):
    return stats.median(result)


def elemwise_equal(
    a_list: list | np.ndarray, b_list: list | np.ndarray, rtol=1e-9, atol=0.0
) -> bool:
    a = np.asarray(a_list)
    b = np.asarray(b_list)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        return np.array_equal(a, b)


# Reference: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
def ntoken(message: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(message)
    return len(tokens)


# * set CUDA Compute Compatibility with env var "CUDA_CC_VER"
CUDA_CC_VER = os.getenv("CUDA_CC_VER", "sm_89")

CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


def pick_idle_gpu(
    get_logic_id: bool = False,
    mem_thres_mb: int = 2048,
    util_thres_percent: float = 32,
    wait_interval: float = 1.0,
    timeout: float | None = None,
    strategy: str = "random",
) -> int:
    pynvml.nvmlInit()
    tstart = time.time()
    try:
        visible = os.environ.get(CUDA_VISIBLE_DEVICES_ENV)
        if visible is not None:
            visible_ids = [int(x) for x in visible.split(",")]
        else:
            visible_ids = list(range(pynvml.nvmlDeviceGetCount()))

        phy2log = {p: i for i, p in enumerate(visible_ids)}

        while True:
            candidates = []

            for phy_id in visible_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(phy_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                mem_used_mb = mem.used // (1 << 20)
                gpu_util = util.gpu

                if mem_used_mb <= mem_thres_mb and gpu_util <= util_thres_percent:
                    candidates.append((phy_id, phy2log[phy_id], mem_used_mb, gpu_util))

            if candidates:
                match strategy:
                    case "random":
                        chosen = random.choice(candidates)
                    case "min_mem":
                        chosen = min(candidates, key=lambda x: x[2])
                    case "min_util":
                        chosen = min(candidates, key=lambda x: x[3])
                    case _:
                        chosen = candidates[0]

                return chosen[1] if get_logic_id else chosen[0]

            if timeout and time.time() - tstart > timeout:
                raise TimeoutError(
                    f"Timeout waiting for idle GPU in {timeout} seconds."
                )

            warnings.warn(
                f"No idle GPUs found, waiting for {wait_interval} seconds...",
                category=UserWarning,
                stacklevel=2,
            )
            time.sleep(wait_interval)
    finally:
        pynvml.nvmlShutdown()


def make_gpu_env(gpu_id: int = None):
    if gpu_id is None:
        gpu_id = pick_idle_gpu()
    env = os.environ.copy()
    env[CUDA_VISIBLE_DEVICES_ENV] = str(gpu_id)
    return env


def run_cmd(
    cmd: list[str], cwd: os.PathLike, gpu_id: int = None, timeout: float = None
):
    env = os.environ.copy()
    if gpu_id is not None:
        env.update(make_gpu_env(gpu_id))

    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=True,
        timeout=timeout,
    )


class Stage(str, Enum):
    BUILD = "build"
    RUN = "run"
    VERIFY = "verify"


def parse_cmd_failure(
    err: subprocess.CalledProcessError | str,
    stage: Stage = Stage.BUILD,
) -> str:
    res = {"stage": stage}
    if isinstance(err, str):
        res["error"] = err
    else:
        res.update(
            {
                "stage": stage,
                "command": err.cmd,
                "returncode": err.returncode,
                "stdout": err.stdout,
                "stderr": err.stderr,
            }
        )
    return json.dumps(res, indent=2, ensure_ascii=False)


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


def compute_label_stats(
    res_list: list[dict[int, dict[str, int]]], labels: list[str], as_df: bool = False
):
    gen_stats = {}
    for gen in res_list[0].keys():
        ratios = [
            sum(res[gen].get(label, 0) for label in labels) / max(res[gen]["total"], 1)
            for res in res_list
        ]
        gen_stats[gen] = {"mean": np.mean(ratios), "std": np.std(ratios)}

    if as_df:
        import pandas as pd

        return pd.DataFrame.from_dict(gen_stats, orient="index")

    return gen_stats
