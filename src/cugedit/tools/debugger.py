import os
from pathlib import Path
from typing import override

import xmltodict
import yaml

from ..config.constants import CUDA_CC_VER
from ..utils import TORCH_INCLUDES, pick_idle_gpu
from .base import BaseTool

"""
LintTool: Check errors (and fix them if possible)
"""


class LintTool(BaseTool):
    capture_output = True
    use_torch = False

    CLANG_CMD = ["clang-tidy", "-fix-errors"]
    CUDA_OPTS = [f"--cuda-gpu-arch={CUDA_CC_VER}"]

    @classmethod
    def _build_cmd(
        cls,
        code_path: Path,
        report_path: Path = None,
    ):
        cmd = cls.CLANG_CMD[:]
        if report_path is not None:
            cmd.append(f"-export-fixes={report_path}")
        cmd += [str(code_path), "--"] + cls.CUDA_OPTS
        if cls.use_torch:
            cmd += TORCH_INCLUDES
        return cmd

    @override
    @classmethod
    def parse(cls, path: Path):
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @override
    @classmethod
    def run(
        cls,
        code_path: os.PathLike,
        cwd: os.PathLike,
        env: os._Environ = None,
        max_retries: int = None,
    ):
        code_path = Path(code_path)
        report_path = code_path.with_suffix(".yml")

        cmd = cls._build_cmd(code_path)
        cmd_with_report = cls._build_cmd(code_path, report_path=report_path)

        last_output = None
        attempts = 0
        while True:
            res = cls.run_cmd(cmd, env=env, cwd=cwd)
            output = (res.stdout, res.stderr)
            if (output == last_output) or (
                max_retries is not None and attempts >= max_retries
            ):
                break
            last_output = output
            attempts += 1

        return cls.run_with_report(cmd_with_report, report_path, cls.parse, env, cwd)


"""
SanitizeTool: Dig out runtime bugs
"""


class SanitizeTool(BaseTool):
    SANITIZER_TOOLS = ["memcheck", "racecheck", "synccheck", "initcheck"]

    @override
    @classmethod
    def parse(cls, path: os.PathLike):
        path = Path(path)
        report = xmltodict.parse(path.read_text()).get("ComputeSanitizerOutput")
        return report

    @override
    @classmethod
    def run(cls, cmd: list[str], cwd: os.PathLike, print_limit: int = 3):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(pick_idle_gpu(True))
        cwd = Path(cwd)

        for tool in cls.SANITIZER_TOOLS:
            report_path = cwd / f"{cwd.stem}_{tool}.xml"
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

            if res := cls.run_with_report(cs_cmd, report_path, cls.parse, env, cwd):
                return res
