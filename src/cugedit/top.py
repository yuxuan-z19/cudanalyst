import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dacite import from_dict

from .module import ChatConfig, Module, ModuleCfg, Planner
from .tools import CodeAnlzTool, LintTool, PerfTool, SanitizeTool, ToolContext


@dataclass
class TopCfg:
    debug_cfg: ModuleCfg
    anlz_cfg: ModuleCfg
    perf_cfg: ModuleCfg
    use_planner: bool

    chat_config_path: os.PathLike
    chat_select: str


def load_top_cfg(yaml_path: str) -> TopCfg:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return from_dict(data_class=TopCfg, data=data)


class Top:
    def __init__(self, cfg: TopCfg):
        service = ChatConfig(cfg.chat_config_path, cfg.chat_select)[0]
        self.static_debugger = Module(cfg.debug_cfg, LintTool, service)
        self.runtime_debugger = Module(cfg.debug_cfg, SanitizeTool, service)
        self.analyzer = Module(cfg.anlz_cfg, CodeAnlzTool, service)
        self.profiler = Module(cfg.perf_cfg, PerfTool, service)

        self.planner = Planner()

    def run(self, ctx: ToolContext) -> str:
        code = Path(ctx.code_path).read_text()

        phases = [
            [self.static_debugger],
            [self.runtime_debugger],
            [self.analyzer, self.profiler],
        ]

        for phase in phases:
            reports = [m.run(ctx) for m in phase]
            active_reports = [r for r in reports if r]
            if active_reports:
                return self.planner.run(code, active_reports)

        return self.planner.run(code)
