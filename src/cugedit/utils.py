import json
import os
import random
import re
import time
import warnings
from typing import Any

import pynvml
import tiktoken

try:
    import sysconfig

    from torch.utils.cpp_extension import include_paths

    TORCH_INCLUDES = [f"-I{path}" for path in include_paths("cuda")]
    TORCH_INCLUDES.append(sysconfig.get_path("include", scheme="posix_prefix"))
except:
    TORCH_INCLUDES = []


def strip_codeblock(codeblock: str) -> str:
    codeblock = codeblock.strip()
    pattern = r"```(?:[a-zA-Z0-9_+-]+\s*)?\n([\s\S]*?)```"
    match = re.search(pattern, codeblock)
    return match.group(1).strip() if match else codeblock


def render_feedback_md(feedback: dict[str, Any]) -> str:
    sections = []
    for key, value in feedback.items():
        pretty_json = json.dumps(value, indent=2, ensure_ascii=False)
        sections.append(f"### {key}\n\n ```json\n{pretty_json}\n```")
    return "\n\n".join(sections)


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
