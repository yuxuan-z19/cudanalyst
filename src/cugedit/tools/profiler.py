import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, override

import ncu_report

from cugedit.utils import pick_idle_gpu

from .base import BaseTool

"""
ProfileTool: Grep runtime performance to identify kernel bottleneck
"""


class PerfTool(BaseTool):
    NCU_BASIC_SETS = [
        "LaunchStats",
        "Occupancy",
        "SpeedOfLight",
        "WorkloadDistribution",
    ]
    SANITIZER_TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]
    NVTX_ID = "cugedit" + "/"

    @classmethod
    def _ncu_filter_rules(cls, raw_rules: list[dict[str, Any]]):
        pattern = re.compile(r"@section:([^:]+):")
        rules_by_section = defaultdict(list)
        included_sections = set(cls.NCU_BASIC_SETS)
        new_sections = set(cls.NCU_BASIC_SETS)
        result = []

        for rule in raw_rules:
            section = rule["section_identifier"]
            rules_by_section[section].append(rule)
            if "speedup_estimation" in rule:
                result.append(rule)
                if section not in included_sections:
                    included_sections.add(section)
                    new_sections.add(section)

        while new_sections:
            found_in_pass = set()
            for section_id in new_sections:
                for rule in rules_by_section.get(section_id, []):
                    msg = rule.get("rule_message", {}).get("message", "")
                    for ref_section in pattern.findall(msg):
                        if ref_section not in included_sections:
                            included_sections.add(ref_section)
                            found_in_pass.add(ref_section)
            new_sections = found_in_pass

        ordered_sections = list(cls.NCU_BASIC_SETS)
        ordered_sections.extend(sorted(included_sections - set(cls.NCU_BASIC_SETS)))

        for section_id in ordered_sections:
            for rule in rules_by_section.get(section_id, []):
                if rule not in result:
                    result.append(rule)

        return result

    @override
    @classmethod
    def parse(cls, path: os.PathLike):
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
    def run(cls, cmd: list[str], cwd: os.PathLike):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(pick_idle_gpu(True))
        cwd = Path(cwd)

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
            cls.NVTX_ID,
            "-o",
            program_name,
        ] + cmd

        return cls.run_with_report(ncu_cmd, report_path, cls.parse, env, cwd)
