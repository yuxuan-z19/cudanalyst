import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolContext:
    code_path: os.PathLike
    cmd: list[str] = None
    cwd: os.PathLike = None


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
            print(">>>>> Timeout!!!!!")
            return None

    @classmethod
    def run_with_report(
        cls,
        cmd: list[str],
        report_path: os.PathLike,
        cwd: os.PathLike = None,
        env: os._Environ = None,
        timeout: float = 300.0,
    ):
        report_path = Path(report_path)
        try:
            res = cls.run_cmd(cmd, cwd=cwd, env=env, timeout=timeout)
            if res is None:
                return None
            return cls.extract(report_path) if report_path.exists() else None
        finally:
            report_path.unlink(missing_ok=True)

    @classmethod
    def extract(cls):
        raise NotImplementedError()

    @classmethod
    def render(cls, feedback):
        return str(feedback)

    @classmethod
    @abstractmethod
    def run(cls, ctx: ToolContext):
        pass
