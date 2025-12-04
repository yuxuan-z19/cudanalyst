import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import ncu_report
import xmltodict

from .utils import pick_idle_gpu


class Analyst:
    NCU_BASIC_SETS = [
        "LaunchStats",
        "Occupancy",
        "SpeedOfLight",
        "WorkloadDistribution",
    ]
    SANITIZER_TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]
    NVTX_ID = "cugedit" + "/"

    @staticmethod
    def run_cmd(
        cmd: list[str],
        env: os._Environ,
        cwd: Path | None = None,
        timeout: float | None = None,
        **kwargs,
    ):
        try:
            return subprocess.run(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return None

    @staticmethod
    def run_with_report(
        cmd: list[str],
        report_path: Path,
        parser: callable,
        env: os._Environ,
        cwd: Path | None = None,
        timeout: float = 300.0,
    ):
        try:
            res = Analyst.run_cmd(cmd, env=env, cwd=cwd, timeout=timeout)
            if res is None:
                return ""
            return parser(report_path) if report_path.exists() else ""
        finally:
            report_path.unlink(missing_ok=True)

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

    @classmethod
    def profile(
        cls, cmd: list[str], cwd: Path, env: os._Environ, use_full_set: bool = False
    ):
        program_name = cwd.stem
        report_path = cwd / f"{program_name}.ncu-rep"

        ncu_cmd = (
            [
                "ncu",
                "-f",
                "--target-processes",
                "all",
                "--nvtx",
                "--nvtx-include",
                cls.NVTX_ID,
                "-o",
                program_name,
            ]
            + (["--set", "full"] if use_full_set else [])
            + cmd
        )

        def parser(path: Path):
            context: ncu_report.IContext = ncu_report.load_report(path)
            cur_range: ncu_report.IRange = context.range_by_idx(0)
            rule_results = {}
            for i in range(cur_range.num_actions()):
                action: ncu_report.IAction = cur_range.action_by_idx(i)
                rules = action.rule_results_as_dicts()
                rule_results[action.name()] = cls._ncu_filter_rules(rules)
            return json.dumps(
                {"tool": "Nsight Compute", "record": rule_results}, indent=2
            )

        return Analyst.run_with_report(ncu_cmd, report_path, parser, env, cwd=cwd)

    @classmethod
    def sanitize(
        cls, cmd: list[str], cwd: Path, env: os._Environ, print_limit: int = 3
    ) -> str:
        for tool in cls.SANITIZER_TOOLS:
            program_name = cwd.stem
            report_path = cwd / f"{program_name}_{tool}.xml"
            cs_cmd = [
                "compute-sanitizer",
                "--tool",
                tool,
                "--show-backtrace",
                "device",
                "--print-limit",
                str(print_limit),
                "--save",
                str(report_path),
                "--xml",
                "yes",
            ] + cmd

            def parser(path: Path):
                report = xmltodict.parse(path.read_text()).get("ComputeSanitizerOutput")
                return (
                    json.dumps(
                        {"tool": f"Compute Sanitizer - {tool}", **report}, indent=2
                    )
                    if report
                    else ""
                )

            if res := Analyst.run_with_report(
                cs_cmd, report_path, parser, env, cwd=cwd
            ):
                return res

        return ""

    @classmethod
    def analyze(cls, cmd: list[str], cwd: Path, valid: bool = True):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(pick_idle_gpu(True))
        if not valid:
            return cls.sanitize(cmd, cwd, env)
        return cls.profile(cmd, cwd, env)
