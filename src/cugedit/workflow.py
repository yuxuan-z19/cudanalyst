import asyncio
import json
import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from .pipeline import *
from .toolkit.base import ToolContext


def planning(config: AnalysisCfg, ctx: ToolContext, error: str = ""):
    return AnalysisPipe(config).run(ctx, error) if config else []


def intervene_sync(
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
    replay: bool = False,
    label: str = None,
    verbose: bool = True,
):
    output_root = Path(output_root_dir)
    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)
    PipeCls = ReplayPipe if replay else IntervenePipe

    res_list = []
    for i in range(num_run):
        output_dir = output_root / str(config_mask) / f"run-{i}"
        pipe = PipeCls(config, evaluate_func, problem_dir)

        if verbose:
            print("is replay: ", replay)
            print("using llm service: ", pipe.service)
            print("config: ", pipe.config)

        res = pipe.run(
            input_ckpt_dir, output_dir, num_trials, max_workers, llm_concurrency
        )
        res_list.append(res)

    print(res_list)
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
    max_parallel_pipes: int = 1,
    problem_dir: os.PathLike = None,
    replay: bool = False,
    label: str = None,
    verbose: bool = True,
):
    output_root = Path(output_root_dir)
    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)
    PipeCls = ReplayPipeAsync if replay else IntervenePipeAsync

    sem = asyncio.Semaphore(max_parallel_pipes)

    async def run_one(i):
        async with sem:
            output_dir = output_root / str(config_mask) / f"run-{i}"
            pipe = PipeCls(config, evaluate_func, problem_dir)

            if verbose:
                print(f"[P{i}] is replay: {replay}")
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


async def intervene_async_multi(
    evaluate_func: Callable,
    input_ckpt_dir: os.PathLike,
    output_root_dir: os.PathLike,
    chat_config_path: os.PathLike,
    config_masks: list[AnalysisMask],
    num_run_per_mask: int,
    num_trials: int,
    max_workers: int = 4,
    llm_concurrency: int = None,
    max_parallel_pipes: int = 1,
    problem_dir: os.PathLike = None,
    replay: bool = False,
    verbose: bool = True,
):
    output_root = Path(output_root_dir)
    PipeCls = ReplayPipeAsync if replay else IntervenePipeAsync

    sem = asyncio.Semaphore(max_parallel_pipes)

    async def run_one(mask: AnalysisMask, i: int):
        async with sem:
            config = apply_config_mask(AnalysisCfg(chat_config_path), mask)
            output_dir = output_root / str(mask) / f"run-{i}"
            pipe = PipeCls(config, evaluate_func, problem_dir)

            if verbose:
                print(f"[Mask={mask}][P{i}] is replay: {replay}")
                print(f"[Mask={mask}][P{i}] using llm service: {pipe.service}")
                print(f"[Mask={mask}][P{i}] config: {pipe.config}")

            res = await pipe.run(
                input_ckpt_dir, output_dir, num_trials, max_workers, llm_concurrency
            )
            return {"mask": str(mask), "run_id": i, "result": res}

    tasks = []
    for mask in config_masks:
        for i in range(num_run_per_mask):
            tasks.append(run_one(mask, i))

    all_results = defaultdict(list)
    for fut in asyncio.as_completed(tasks):
        r = await fut
        all_results[r["mask"]].append(r["result"])

    for mask, mask_results in all_results.items():
        (output_root / f"{mask}.json").write_text(
            json.dumps(mask_results, ensure_ascii=False, indent=2)
        )

    return all_results
