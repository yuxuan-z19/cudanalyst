import asyncio
import json
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.rich import tqdm

from ..helper.ckpt import iter_program_json
from ..helper.text import extract_codeblock
from ..module.chat import ChatSession, Service
from ..result import Status
from .base import Pipeline
from .config import AnalysisCfg

MAX_RETRY = 2
BASE_SLEEP = 0.2
JITTER_SLEEP = 0.2


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
        else:
            return f"{user_prompt.rstrip()}\n\n{decision}"

    @staticmethod
    def _eval(
        code: str,
        evaluator: Callable,
        problem_dir: os.PathLike = None,
        config: AnalysisCfg = None,
    ) -> dict[str, Any]:
        tmp = Path(tempfile.mkstemp(suffix=".cu")[1])
        try:
            tmp.write_text(extract_codeblock(code), encoding="utf-8")
            args = [tmp]
            if problem_dir:
                args.append(problem_dir)
            if config:
                args.append(config)
            return evaluator(*args)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    async def _run_trials_async(
        service: Service,
        sys_prompt: str,
        usr_prompt: str,
        evaluator: Callable,
        problem_dir: os.PathLike,
        num_trials: int,
    ):
        async def _ask_with_retry():
            attempt = 0
            while True:
                try:
                    return await ChatSession(service, sys_prompt).ask_async(usr_prompt)
                except Exception:
                    if attempt >= MAX_RETRY:
                        return ""
                    time.sleep(
                        BASE_SLEEP * (2**attempt) + random.random() * JITTER_SLEEP
                    )
                    attempt += 1

        tasks = [asyncio.create_task(_ask_with_retry()) for _ in range(num_trials)]
        results = []

        for fut in asyncio.as_completed(tasks):
            try:
                reply = await fut
                code = extract_codeblock(reply)
                metrics = IntervenePipe._eval(code, evaluator, problem_dir)
                results.append({"code": code, "metrics": metrics})
            except Exception as e:
                results.append(
                    {
                        "code": None,
                        "metrics": {
                            "status": "fail",
                            "error": {
                                "type": type(e).__name__,
                                "msg": str(e),
                            },
                        },
                    }
                )
        return results

    @staticmethod
    def _run_one_sample(
        data: dict,
        config: AnalysisCfg,
        service: Service,
        evaluator: Callable,
        problem_dir: os.PathLike = None,
        num_trials: int = 1,
    ) -> dict:
        gen = data["generation"]
        sample_id = data["id"]
        raw_prompt = data["prompts"]["full_rewrite_user"]

        old_code = data["code"]
        old_metrics = IntervenePipe._eval(old_code, evaluator, problem_dir, config)
        reports = old_metrics.get("reports", [])

        plan_decision = IntervenePipe._format_plan_decision(reports)
        user_prompt = IntervenePipe._inject_decision(raw_prompt["user"], plan_decision)

        results = asyncio.run(
            IntervenePipe._run_trials_async(
                service,
                raw_prompt["system"],
                user_prompt,
                evaluator,
                problem_dir,
                num_trials,
            )
        )

        statuses = [Status(r["metrics"].get("status", "fail")) for r in results]
        status_counter = Counter(s.value for s in statuses)
        highest_status = max(statuses) if statuses else Status("fail")

        return {
            "gen": gen,
            "id": sample_id,
            "num_trials": num_trials,
            "status_count": dict(status_counter),
            "highest_status": highest_status.value,
            "reports": reports,
            "plan_decision": plan_decision,
            "prompts": {"system": raw_prompt["system"], "user": user_prompt},
            "results": results,
            "old_code": old_code,
            "old_metrics": old_metrics,
        }

    def run(
        self,
        input_ckpt_dir: os.PathLike,
        output_dir: os.PathLike,
        max_workers: int = 4,
        num_trials: int = 1,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gen_stats = defaultdict(Counter)
        futures = []

        with ProcessPoolExecutor(
            max_workers=max_workers, max_tasks_per_child=1
        ) as pool:
            for _, data in iter_program_json(input_ckpt_dir):
                futures.append(
                    pool.submit(
                        IntervenePipe._run_one_sample,
                        data,
                        self.config,
                        self.service,
                        self.evaluator,
                        self.problem_dir,
                        num_trials,
                    )
                )

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    record = fut.result()
                    if not record:
                        continue

                    gen = record.get("gen", "unknown")
                    status = record.get("highest_status", "fail")

                    gen_stats[gen]["total"] += 1
                    gen_stats[gen][status] += 1

                    target = output_dir / str(gen)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"{record['id']}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    raise

        return {k: dict(v) for k, v in gen_stats.items()}
