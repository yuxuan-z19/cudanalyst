from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import override

from ..helper.exec import ExecFailReason
from ..module.module import Module, Planner, Report
from ..module.prompts import *
from ..toolkit import *
from .base import Pipeline
from .config import AnalysisCfg


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

        if not (ExecFailReason.VERIFY_MISMATCH in error):
            phases_to_run = self.fail_phases if error else self.pass_phases

            with ThreadPoolExecutor(max_workers=len(phases_to_run)) as pool:
                futures = [pool.submit(phase.run, ctx) for phase in phases_to_run]
                for fut in as_completed(futures):
                    if r := fut.result():
                        reports.append(r)

        planner_report = self.planner.run(ctx, reports)
        return [planner_report, *reports]
