import json
import os
import tempfile
import traceback
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import yaml
from dacite import from_dict
from tqdm.rich import tqdm

from .helper import extract_codeblock, iter_program_json
from .module.chat import ChatConfig
from .module.module import *
from .module.prompts import *
from .result import ResultConstant, Status
from .tools import CodeAnlzTool, LintTool, PerfTool, SanitizeTool, ToolContext


@dataclass
class AnalysisCfg:
    chat_config_path: str
    chat_select: str
    debug_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    anlz_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    perf_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    plan_cfg: ModuleCfg = field(default_factory=ModuleCfg)


def load_analysis_cfg(yaml_path: str) -> AnalysisCfg:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return from_dict(data_class=AnalysisCfg, data=data)


class Pipeline(ABC):
    def __init__(self, config: AnalysisCfg):
        self.config = config
        self.service = ChatConfig(config.chat_config_path, config.chat_select)[0]

    @abstractmethod
    def run(self):
        pass


class AnalysisPipe(Pipeline):
    def __init__(self, config: AnalysisCfg):
        super().__init__(config)

        self.fail_phases = [
            Module(config.debug_cfg, LINT_PROMPT, self.service, LintTool),
            Module(config.debug_cfg, SANITIZE_PROMPT, self.service, SanitizeTool),
        ]

        self.pass_phases = [
            Module(config.anlz_cfg, ANLZ_PROMPT, self.service, CodeAnlzTool),
            Module(config.perf_cfg, PERF_PROMPT, self.service, PerfTool),
        ]

        self.planner = Planner(config.plan_cfg, PLAN_PROMPT, self.service)

    @override
    def run(self, ctx: ToolContext, error: str) -> list[Report]:
        reports: list[Report] = []

        if error:
            reports.append(Report("ErrorPhase", error, error))

        if not (ResultConstant.ERR_MISMATCH in error):
            phases_to_run = self.fail_phases if error else self.pass_phases
            reports.extend(r for phase in phases_to_run if (r := phase.run(ctx)))

        planner_report = self.planner.run(ctx, reports)
        return [planner_report, *reports]


def generate_plan(config: AnalysisCfg, ctx: ToolContext, error: str = ""):
    return AnalysisPipe(config).run(ctx, error) if config else []


class IntervenePipe(Pipeline):
    NEW_SESSION = True
    MODE = "full_rewrite_user"

    def __init__(self, config: AnalysisCfg, evaluate_func: Callable):
        super().__init__(config)
        self.evaluator = evaluate_func

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
    def _run_one_sample(
        data: dict,
        config: AnalysisCfg,
        service,
        evaluator: Callable,
        problem_dir: os.PathLike = None,
        k: int = 1,
    ) -> dict:
        gen = data["generation"]
        sample_id = data["id"]
        raw_prompt = data["prompts"]["full_rewrite_user"]

        def _eval(code: str, use_cfg: bool = True) -> dict[str, Any]:
            tmp = Path(tempfile.mkstemp(suffix=".cu")[1])
            try:
                tmp.write_text(extract_codeblock(code), encoding="utf-8")

                args = [tmp]
                if problem_dir:
                    args.append(problem_dir)
                if use_cfg:
                    args.append(config)

                return evaluator(*args)
            finally:
                tmp.unlink(missing_ok=True)

        old_code = data["code"]
        old_metrics = _eval(old_code)
        reports = old_metrics.get("reports", [])

        plan_decision = IntervenePipe._format_plan_decision(reports)
        user_prompt = IntervenePipe._inject_decision(raw_prompt["user"], plan_decision)

        def _run_trial(_):
            agent = ChatSession(service, raw_prompt["system"])
            reply = agent.ask(user_prompt)

            new_code = extract_codeblock(reply)
            new_metrics = _eval(new_code, False)

            return {"code": new_code, "metrics": new_metrics}

        with ThreadPoolExecutor(max_workers=k) as executor:
            results = list(executor.map(_run_trial, range(k)))

        statuses = [Status(r["metrics"].get("status", "fail")) for r in results]
        status_counter = Counter(s.value for s in statuses)
        highest_status = max(statuses)

        return {
            "gen": gen,
            "id": sample_id,
            "k": k,
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
        problem_dir: os.PathLike | None = None,
        max_workers: int = 4,
        k: int = 1,
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
                        self._run_one_sample,
                        data,
                        self.config,
                        self.service,
                        self.evaluator,
                        Path(problem_dir) if problem_dir else None,
                        k,
                    )
                )

            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    record = fut.result()
                    if not record:
                        continue

                    gen = record.get("gen", "unknown")
                    highest_status = record.get("highest_status", "fail")

                    gen_stats[gen]["total"] += 1
                    gen_stats[gen][highest_status] += 1

                    target_dir = output_dir / str(gen)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    (target_dir / f"{record['id']}.json").write_text(
                        json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception as e:
                    error_info = traceback.format_exc()
                    print(f"Worker error: {e}\n")
                    print(error_info)

        return {k: dict(v) for k, v in gen_stats.items()}
