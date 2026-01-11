import asyncio
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, override

from tqdm.asyncio import tqdm_asyncio
from tqdm.rich import tqdm

from ..helper.ckpt import iter_program_json
from ..helper.exec import Status
from ..helper.text import extract_codeblock
from ..module.chat import ChatSession, Service
from ..module.module import Report
from ..module.prompts import PromptCfg
from .base import Pipeline
from .config import AnalysisCfg

MAX_RETRY = 2
TIMEOUT = 15 * 60
BASE_SLEEP = 0.2
JITTER_SLEEP = 0.2


@dataclass(frozen=True)
class ID:
    gen: int
    pid: str


@dataclass
class CodeProfile:
    code: str
    metrics: dict[str, Any]


@dataclass
class PromptCtx:
    id: ID
    prompt: PromptCfg
    profile: CodeProfile
    reports: list[Report] = field(default_factory=list)


@dataclass
class CodeSample:
    id: ID
    code: str = None
    error: str = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Record:
    id: ID
    num_trials: int
    status_count: defaultdict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    highest_status: str = Status.FAIL.value
    reports: list[Report] = field(default_factory=list)
    plan_decision: str = ""
    prompt: PromptCfg = None
    results: list[CodeProfile] = field(default_factory=list)
    old_profile: CodeProfile = None


class IntervenePipe(Pipeline):
    def __init__(
        self,
        config: AnalysisCfg,
        evaluate_func: Callable,
        problem_dir: os.PathLike = None,
    ):
        super().__init__(config)
        self.evaluator = evaluate_func
        self.problem_dir = problem_dir

    @staticmethod
    def _format_plan_decision(reports: list[dict]) -> str:
        if not reports:
            return ""
        sections = []
        for r in reports:
            decision = r.get("summary") or r.get("feedback")
            sections.append(f"## {r['name']}\nPlan decision: {decision}\n")
        return "# Plan Decision\n\n" + "\n\n".join(sections)

    @staticmethod
    def _inject_decision(user_prompt: str, decision: str) -> str:
        if not decision:
            return user_prompt
        anchor = "# Program Evolution History"
        if anchor in user_prompt:
            before, after = user_prompt.split(anchor, 1)
            return f"{before.rstrip()}\n\n{decision}\n\n{anchor}{after}"
        return f"{user_prompt.rstrip()}\n\n{decision}"

    @staticmethod
    def _eval(
        code: str,
        evaluator: Callable,
        problem_dir: os.PathLike = None,
        config: AnalysisCfg = None,
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cu",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = Path(f.name)

        try:
            args = [tmp_path]
            if problem_dir:
                args.append(problem_dir)
            if config:
                args.append(config)
            return evaluator(*args)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    async def build_prompt_context(
        data: dict[str, Any],
        evaluator: Callable,
        problem_dir: os.PathLike,
        config: AnalysisCfg,
    ) -> PromptCtx:
        loop = asyncio.get_running_loop()
        args = [data["code"], evaluator, problem_dir, config]
        metrics = await loop.run_in_executor(None, IntervenePipe._eval, *args)
        raw_prompts = data["prompts"]["full_rewrite_user"]
        return PromptCtx(
            id=ID(data["generation"], data["id"]),
            prompt=PromptCfg(raw_prompts["system"], raw_prompts["user"]),
            profile=CodeProfile(data["code"], metrics),
            reports=metrics.get("reports", []),
        )

    async def build_all_contexts(
        self, input_ckpt_dir: os.PathLike, concurrency: int = 8
    ):
        sem = asyncio.Semaphore(concurrency)

        async def sem_task(data):
            async with sem:
                attempt = 0
                while True:
                    try:
                        return await self.build_prompt_context(
                            data, self.evaluator, self.problem_dir, self.config
                        )
                    except Exception as e:
                        if attempt >= MAX_RETRY:
                            print(
                                f"Error building context for {data.get('id', 'unknown')}: {e}"
                            )
                            return None
                        attempt += 1
                        await asyncio.sleep(
                            BASE_SLEEP * (2**attempt) + random.random() * JITTER_SLEEP
                        )

        tasks = [sem_task(data) for _, data in iter_program_json(input_ckpt_dir)]
        contexts = []
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="BUILD"):
            ctx = await fut
            if ctx:
                contexts.append(ctx)
        return contexts

    @staticmethod
    async def gen_codes(ctx: PromptCtx, service: Service, num_trials: int):
        plan_decision = IntervenePipe._format_plan_decision(ctx.reports)
        usr_prompt = IntervenePipe._inject_decision(ctx.prompt.usr, plan_decision)

        async def one_trial():
            session = ChatSession(service, ctx.prompt.sys)
            attempt = 0
            while True:
                try:
                    reply = await asyncio.wait_for(
                        session.ask_async(usr_prompt), timeout=TIMEOUT
                    )
                    return CodeSample(ctx.id, extract_codeblock(reply))
                except (asyncio.TimeoutError, Exception) as e:
                    if attempt >= MAX_RETRY:
                        return CodeSample(ctx.id, error=str(e))

                    attempt += 1
                    await asyncio.sleep(
                        BASE_SLEEP * (2**attempt) + random.random() * JITTER_SLEEP
                    )

        tasks = [asyncio.create_task(one_trial()) for _ in range(num_trials)]
        return [await t for t in asyncio.as_completed(tasks)]

    async def run_llm_stage(
        self,
        contexts: list[PromptCtx],
        service: Service,
        num_trials: int,
        concurrency: int = 8,
    ):
        sem = asyncio.Semaphore(concurrency)

        async def sem_gen_codes(ctx):
            async with sem:
                return await self.gen_codes(ctx, service, num_trials)

        tasks = [sem_gen_codes(ctx) for ctx in contexts]
        all_samples = []
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="CGO"):
            res = await fut
            all_samples.extend(res)
        return all_samples

    @staticmethod
    def eval_one_sample(
        sample: CodeSample, evaluator: Callable, problem_dir: os.PathLike
    ) -> CodeSample:
        if not sample.error:
            sample.metrics = IntervenePipe._eval(sample.code, evaluator, problem_dir)
        return sample

    def run(
        self,
        input_ckpt_dir: os.PathLike,
        output_dir: os.PathLike,
        num_trials: int = 1,
        max_workers: int = 4,
        llm_concurrency: int = None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if llm_concurrency is None:
            llm_concurrency = max(1, max_workers // 4)

        contexts: list[PromptCtx] = asyncio.run(
            self.build_all_contexts(input_ckpt_dir, llm_concurrency)
        )
        llm_outputs = asyncio.run(
            self.run_llm_stage(contexts, self.service, num_trials, llm_concurrency)
        )

        ctx_map = {ctx.id.pid: ctx for ctx in contexts}
        records: dict[str, Record] = {}

        with ProcessPoolExecutor(
            max_workers=max_workers, max_tasks_per_child=1
        ) as pool:
            futures = [
                pool.submit(
                    IntervenePipe.eval_one_sample,
                    sample,
                    self.evaluator,
                    self.problem_dir,
                )
                for sample in llm_outputs
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="EVAL"):
                sample: CodeSample = fut.result()
                if sample.error or not sample.metrics:
                    continue

                pid = sample.id.pid
                ctx = ctx_map[pid]

                if pid not in records:
                    records[pid] = Record(
                        id=ctx.id,
                        num_trials=num_trials,
                        reports=ctx.reports,
                        plan_decision=IntervenePipe._format_plan_decision(ctx.reports),
                        prompt=ctx.prompt,
                        old_profile=ctx.profile,
                    )

                record = records[pid]
                record.results.append(CodeProfile(sample.code, sample.metrics))

                status = sample.metrics.get("status", "fail")
                record.status_count[status] += 1
                record.status_count["total"] += 1
                if Status(status) > Status(record.highest_status):
                    record.highest_status = status

        gen_stats = defaultdict(Counter)
        for pid, record in records.items():
            gen = record.id.gen
            status = record.highest_status

            gen_stats[gen]["total"] += 1
            gen_stats[gen][status] += 1

            target = output_dir / str(gen)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{pid}.json").write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {k: dict(v) for k, v in gen_stats.items()}


class IntervenePipeAsync(IntervenePipe):
    def __init__(
        self,
        config: AnalysisCfg,
        evaluate_func: Callable,
        problem_dir: os.PathLike = None,
    ):
        super().__init__(config, evaluate_func, problem_dir)

    async def produce_contexts(
        self,
        input_ckpt_dir: os.PathLike,
        ctx_queue: asyncio.Queue,
        ctx_map: dict[str, PromptCtx],
        llm_concurrency: int = 8,
    ):
        sem = asyncio.Semaphore(llm_concurrency)

        async def sem_task(data):
            async with sem:
                attempt = 0
                while True:
                    try:
                        return await self.build_prompt_context(
                            data, self.evaluator, self.problem_dir, self.config
                        )
                    except Exception as e:
                        if attempt >= MAX_RETRY:
                            print(
                                f"Error building context for {data.get('id', 'unknown')}: {e}"
                            )
                            return None
                        attempt += 1
                        await asyncio.sleep(
                            BASE_SLEEP * (2**attempt) + random.random() * JITTER_SLEEP
                        )

        tasks = [
            asyncio.create_task(sem_task(data))
            for _, data in iter_program_json(input_ckpt_dir)
        ]

        for fut in asyncio.as_completed(tasks):
            ctx: PromptCtx = await fut
            if ctx:
                ctx_map[ctx.id.pid] = ctx
                await ctx_queue.put(ctx)

        for _ in range(llm_concurrency):
            await ctx_queue.put(None)

    async def llm_worker(
        self,
        ctx_queue: asyncio.Queue,
        sample_queue: asyncio.Queue,
        service: Service,
        num_trials: int,
    ):
        while True:
            ctx = await ctx_queue.get()
            if ctx is None:
                await sample_queue.put(None)
                break
            for s in await self.gen_codes(ctx, service, num_trials):
                await sample_queue.put(s)

    async def eval_consumer(
        self,
        sample_queue: asyncio.Queue,
        records: dict[str, Record],
        ctx_map: dict[str, PromptCtx],
        num_trials: int,
        max_workers: int,
        llm_concurrency: int,
    ):
        loop = asyncio.get_running_loop()

        total = len(ctx_map)
        pbar = tqdm(total=total, desc="EVAL")

        with ProcessPoolExecutor(
            max_workers=max_workers, max_tasks_per_child=1
        ) as pool:
            finished = 0

            while True:
                sample = await sample_queue.get()
                if sample is None:
                    finished += 1
                    if finished == llm_concurrency:
                        break
                    continue

                sample = await loop.run_in_executor(
                    pool,
                    self.eval_one_sample,
                    sample,
                    self.evaluator,
                    self.problem_dir,
                )

                if sample.error or not sample.metrics:
                    continue

                pid = sample.id.pid
                ctx = ctx_map[pid]

                if pid not in records:
                    records[pid] = Record(
                        id=ctx.id,
                        num_trials=num_trials,
                        reports=ctx.reports,
                        plan_decision=self._format_plan_decision(ctx.reports),
                        prompt=ctx.prompt,
                        old_profile=ctx.profile,
                    )

                record = records[pid]
                record.results.append(CodeProfile(sample.code, sample.metrics))

                status = sample.metrics.get("status", "fail")
                record.status_count[status] += 1
                record.status_count["total"] += 1
                if Status(status) > Status(record.highest_status):
                    record.highest_status = status
                if record.status_count["total"] == num_trials:
                    pbar.update(1)

            pbar.close()

    @override
    async def run(
        self,
        input_ckpt_dir: os.PathLike,
        output_dir: os.PathLike,
        num_trials: int = 1,
        max_workers: int = 4,
        llm_concurrency: int = None,
    ):
        if llm_concurrency is None:
            llm_concurrency = max(1, max_workers // 4)

        self.llm_concurrency = llm_concurrency
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ctx_queue = asyncio.Queue(maxsize=llm_concurrency * 2)
        sample_queue = asyncio.Queue(
            maxsize=max(llm_concurrency * num_trials, max_workers * 2)
        )

        records: dict[str, Record] = {}
        ctx_map: dict[str, PromptCtx] = {
            data["id"]: None for _, data in iter_program_json(input_ckpt_dir)
        }

        await asyncio.gather(
            self.produce_contexts(input_ckpt_dir, ctx_queue, ctx_map, llm_concurrency),
            *[
                self.llm_worker(ctx_queue, sample_queue, self.service, num_trials)
                for _ in range(llm_concurrency)
            ],
            self.eval_consumer(
                sample_queue,
                records,
                ctx_map,
                num_trials,
                max_workers,
                llm_concurrency,
            ),
        )

        gen_stats = defaultdict(Counter)
        for pid, record in records.items():
            gen = record.id.gen
            status = record.highest_status
            gen_stats[gen]["total"] += 1
            gen_stats[gen][status] += 1

            target = output_dir / str(gen)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{pid}.json").write_text(
                json.dumps(asdict(record), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return {k: dict(v) for k, v in gen_stats.items()}
