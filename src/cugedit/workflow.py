import json
import os
from collections.abc import Callable
from pathlib import Path

from .pipeline import AnalysisPipe, IntervenePipe
from .pipeline.config import AnalysisCfg, AnalysisMask, apply_config_mask
from .toolkit.base import ToolContext


def intervene(
    evaluate_func: Callable,
    input_ckpt_dir: os.PathLike,
    output_root_dir: os.PathLike,
    chat_config_path: os.PathLike,
    config_mask: AnalysisMask,
    num_run: int,
    num_trials: int,
    max_workers: int = 4,
    problem_dir: os.PathLike = None,
    label: str = None,
):
    output_root = Path(output_root_dir)
    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)

    res_list = []
    for i in range(num_run):
        output_dir = output_root / str(config_mask) / f"run-{i}"
        pipe = IntervenePipe(config, evaluate_func, problem_dir)
        print("using llm service: ", pipe.service)
        res = pipe.run(input_ckpt_dir, output_dir, max_workers, num_trials)
        res_list.append(res)

    label = label or str(config_mask)
    (output_root / f"{label}.json").write_text(json.dumps(res_list, ensure_ascii=True))
    return res_list


def planning(config: AnalysisCfg, ctx: ToolContext, error: str = ""):
    return AnalysisPipe(config).run(ctx, error) if config else []
