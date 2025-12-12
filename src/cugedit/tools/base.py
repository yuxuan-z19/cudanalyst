import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class BaseTool(ABC):
    capture_output: bool = False

    @classmethod
    def run_cmd(
        cls,
        cmd: list[str],
        env: os._Environ = None,
        cwd: os.PathLike = None,
        timeout: float = None,
        **kwargs,
    ):
        stdout = subprocess.PIPE if cls.capture_output else subprocess.DEVNULL
        stderr = subprocess.PIPE if cls.capture_output else subprocess.DEVNULL

        try:
            return subprocess.run(
                cmd,
                env=env,
                cwd=cwd,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout,
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return None

    @classmethod
    def run_with_report(
        cls,
        cmd: list[str],
        report_path: os.PathLike,
        parse: callable,
        env: os._Environ = None,
        cwd: os.PathLike = None,
        timeout: float = 300.0,
    ):
        report_path = Path(report_path)
        try:
            res = cls.run_cmd(cmd, env=env, cwd=cwd, timeout=timeout)
            if res is None:
                return None
            return parse(report_path) if report_path.exists() else None
        finally:
            report_path.unlink(missing_ok=True)

    @classmethod
    def parse(cls):
        raise NotImplementedError(
            "parse() has no default implementation; override in subclass."
        )

    @classmethod
    @abstractmethod
    def run(cls):
        pass


@dataclass
class ToolSummary:
    name: str = None
    profile: str = None
    summary: str = None
