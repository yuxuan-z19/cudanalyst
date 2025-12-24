import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, override

import ncu_report

from ..helper import pick_idle_gpu, render_feedback_md
from .base import BaseTool, ToolContext


class PerfTool(BaseTool):
    _NCU_BASIC_SETS = [
        "LaunchStats",
        "Occupancy",
        "SpeedOfLight",
        "WorkloadDistribution",
    ]
    _NVTX_ID = "cugedit" + "/"
    _SECTION_PATTERN = re.compile(r"@section:([^:]+):")

    @classmethod
    def _ncu_filter_rules(cls, raw_rules: list[dict[str, Any]]):
        rules_by_section = defaultdict(list)
        result = []
        added_rule_ids = set()
        included_sections = set(cls._NCU_BASIC_SETS)

        for rule in raw_rules:
            section = rule["section_identifier"]
            rules_by_section[section].append(rule)

            if "speedup_estimation" in rule:
                if id(rule) not in added_rule_ids:
                    result.append(rule)
                    added_rule_ids.add(id(rule))
                included_sections.add(section)

        new_sections = set(included_sections)
        while new_sections:
            found_in_pass = set()
            for section_id in new_sections:
                for rule in rules_by_section.get(section_id, []):
                    msg = rule.get("rule_message", {}).get("message", "")
                    for ref_section in cls._SECTION_PATTERN.findall(msg):
                        if ref_section not in included_sections:
                            included_sections.add(ref_section)
                            found_in_pass.add(ref_section)
            new_sections = found_in_pass

        ordered_sections = list(cls._NCU_BASIC_SETS)
        other_sections = sorted(included_sections - set(cls._NCU_BASIC_SETS))
        ordered_sections.extend(other_sections)

        for section_id in ordered_sections:
            for rule in rules_by_section.get(section_id, []):
                if id(rule) not in added_rule_ids:
                    result.append(rule)
                    added_rule_ids.add(id(rule))

        return result

    @override
    @classmethod
    def extract(cls, path: os.PathLike):
        context: ncu_report.IContext = ncu_report.load_report(path)
        cur_range: ncu_report.IRange = context.range_by_idx(0)
        rule_results = {}
        for i in range(cur_range.num_actions()):
            action: ncu_report.IAction = cur_range.action_by_idx(i)
            rules = action.rule_results_as_dicts()
            rule_results[action.name()] = cls._ncu_filter_rules(rules)
        return rule_results

    @override
    @classmethod
    def render(cls, feedback: dict[str, dict[str, Any]]):
        return render_feedback_md(feedback)

    @override
    @classmethod
    def run(cls, ctx: ToolContext):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(pick_idle_gpu(True))
        cwd = Path(ctx.cwd)

        program_name = cwd.stem
        report_path = cwd / f"{program_name}.ncu-rep"
        ncu_cmd = [
            "ncu",
            "-f",
            "--target-processes",
            "all",
            "--set",
            "full",
            "--nvtx",
            "--nvtx-include",
            cls._NVTX_ID,
            "-o",
            program_name,
        ] + ctx.cmd

        return cls.run_with_report(ncu_cmd, report_path, cwd, env)
