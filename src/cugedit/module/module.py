from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from ..tools.base import BaseTool, ToolContext
from .chat import ChatSession, Service
from .prompts import SUMMARY_MAP, PromptCfg


@dataclass
class ModuleCfg:
    enabled: bool = False
    formatted: bool = False
    summarized: bool = False


@dataclass
class Report:
    name: str
    feedback: Any = None
    summary: str = None

    def __bool__(self) -> bool:
        return self.feedback is not None


class Module:
    NAME = None

    def __init__(
        self, cfg: ModuleCfg, prompt: PromptCfg, service: Service, tool: BaseTool
    ):
        self.cfg = cfg
        self.prompt = prompt
        self.agent = ChatSession(service, prompt.sys)
        self.tool = tool
        self.name = self.NAME or self.tool.__name__

    def _render_prompt(self, ctx: ToolContext, context: str):
        template = self.prompt.usr or ""
        placeholders = {
            f"<{self.name.upper()}>": context,
            "<RAWCODE>": Path(ctx.code_path).read_text().strip(),
        }
        for k, v in placeholders.items():
            template = template.replace(k, v)
        return template

    def _summarize(self, ctx: ToolContext, context: str):
        prompt = self._render_prompt(ctx, context)
        return self.agent.ask(prompt)

    def run(self, ctx: ToolContext) -> Report:
        if not self.cfg.enabled:
            return Report(self.name)
        feedback = self.tool.run(ctx)
        feedback = self.tool.render(feedback) if self.cfg.formatted else str(feedback)
        summary = self._summarize(ctx, feedback) if self.cfg.summarized else None
        return Report(self.name, feedback, summary)


class Planner(Module):
    NAME = "Planner"

    def __init__(self, cfg: ModuleCfg, prompt: PromptCfg, service: Service):
        super().__init__(cfg, prompt, service, None)

    def _build_context(self, reports: list[Report]) -> str:
        if not reports:
            return ""

        summaries = []
        for report in reports:
            template = SUMMARY_MAP.get(report.name)
            template = template.replace(
                f"<{report.name.upper()}>", report.summary or report.feedback
            )
            summaries.append(template)

        return "\n".join(summaries)

    @override
    def run(self, ctx: ToolContext, reports: list[Report]) -> Report:
        context = self._build_context(reports)
        if not self.cfg.enabled:
            return Report(self.name, context)
        plan = self._summarize(ctx, context) if self.cfg.summarized else context
        return Report(self.name, context, plan)
