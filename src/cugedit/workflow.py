import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path

from .pipeline import AnalysisPipe, IntervenePipe, IntervenePipeAsync
from .pipeline.config import AnalysisCfg, AnalysisMask, apply_config_mask
from .toolkit.base import ToolContext


def planning(config: AnalysisCfg, ctx: ToolContext, error: str = ""):
    return AnalysisPipe(config).run(ctx, error) if config else []


def intervene(
    evaluate_func: Callable,
    input_ckpt_dir: os.PathLike,
    output_root_dir: os.PathLike,
    chat_config_path: os.PathLike,
    config_mask: AnalysisMask,
    num_run: int,
    num_trials: int,
    max_workers: int = 4,
    llm_concurrency: int = None,
    problem_dir: os.PathLike = None,
    label: str = None,
    verbose: bool = True,
):
    output_root = Path(output_root_dir)
    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)

    res_list = []
    for i in range(num_run):
        output_dir = output_root / str(config_mask) / f"run-{i}"
        pipe = IntervenePipe(config, evaluate_func, problem_dir)

        if verbose:
            print("using llm service: ", pipe.service)
            print("config: ", pipe.config)

        res = pipe.run(
            input_ckpt_dir, output_dir, num_trials, max_workers, llm_concurrency
        )
        res_list.append(res)

    label = label or str(config_mask)
    (output_root / f"{label}.json").write_text(json.dumps(res_list, ensure_ascii=True))
    return res_list


async def intervene_async(
    evaluate_func: Callable,
    input_ckpt_dir: os.PathLike,
    output_root_dir: os.PathLike,
    chat_config_path: os.PathLike,
    config_mask: AnalysisMask,
    num_run: int,
    num_trials: int,
    max_workers: int = 4,
    llm_concurrency: int = None,
    max_parallel_pipes: int = 2,
    problem_dir: os.PathLike = None,
    label: str = None,
    verbose: bool = True,
):
    output_root = Path(output_root_dir)
    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)
    sem = asyncio.Semaphore(max_parallel_pipes)

    async def run_one(i):
        async with sem:
            output_dir = output_root / str(config_mask) / f"run-{i}"
            pipe = IntervenePipeAsync(config, evaluate_func, problem_dir)
            if verbose:
                print(f"[P{i}] using llm service: {pipe.service}")
                print(f"[P{i}] config: {pipe.config}")
            return await pipe.run(
                input_ckpt_dir, output_dir, num_trials, max_workers, llm_concurrency
            )

    res_list = []
    tasks = [run_one(i) for i in range(num_run)]
    for fut in asyncio.as_completed(tasks):
        res = await fut
        res_list.append(res)

    label = label or str(config_mask)
    (output_root / f"{label}.json").write_text(json.dumps(res_list, ensure_ascii=False))
    return res_list
