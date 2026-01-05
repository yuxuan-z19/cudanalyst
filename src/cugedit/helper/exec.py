import json
import os
import subprocess
from dataclasses import dataclass
from enum import Enum

from .cuda import make_gpu_env


class Stage(str, Enum):
    BUILD = "build"
    RUN = "run"
    VERIFY = "verify"


class Score:
    INVALID_INT: int = 0
    INVALID_FLOAT: float = float("-inf")


class Status(str, Enum):
    FAIL = "fail"
    COMPILE = "compile"
    PASS = "pass"

    # only be set with check_fast()
    FAST = "fast"

    @property
    def rank(self):
        _ranks = {Status.FAIL: 0, Status.COMPILE: 1, Status.PASS: 2, Status.FAST: 3}
        return _ranks.get(self, -1)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.rank < other.rank
        return NotImplemented


class ExecFailReason(str, Enum):
    TIMEOUT = "Execution Timeout"
    FAILED = "Execution Failure: Nonzero Exit"
    SPAWN_FAILED = "Failed to start execution"
    OOM = "Out of Memory"
    VERIFY_MISMATCH: str = "Mismatch output, implementation error"


@dataclass
class ExecError(Exception):
    stage: Stage
    cmd: list[str]
    reason: ExecFailReason
    returncode: int = None
    stdout: str = None
    stderr: str = None

    def __post_init__(self):
        super().__init__(self.reason.value)

    def __str__(self) -> str:
        try:
            info = {
                "stage": self.stage.value if self.stage else None,
                "reason": self.reason.value if self.reason else None,
                "command": self.cmd,
                "returncode": self.returncode,
                "stdout": self.stdout,
                "stderr": self.stderr,
            }
            info = {k: v for k, v in info.items() if v is not None}
            return json.dumps(info, indent=2, ensure_ascii=False)
        except Exception:
            return f"ExecError(stage={self.stage}, reason={self.reason})"


def run_cmd(
    cmd: list[str],
    cwd: os.PathLike,
    stage: Stage,
    gpu_id: int = None,
    timeout: float = None,
):
    env = os.environ.copy()
    if gpu_id is not None:
        env.update(make_gpu_env(gpu_id))
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            check=True,
            timeout=timeout,
        )
    except Exception as e:
        base_kwargs = {"stage": stage, "cmd": cmd}
        match e:
            case subprocess.TimeoutExpired():
                reason = ExecFailReason.TIMEOUT
                base_kwargs["stdout"] = getattr(e, "output", None)
                base_kwargs["stderr"] = getattr(e, "stderr", None)

            case subprocess.CalledProcessError():
                reason = ExecFailReason.FAILED
                base_kwargs["returncode"] = e.returncode
                base_kwargs["cmd"] = e.cmd
                base_kwargs["stdout"] = e.stdout
                base_kwargs["stderr"] = e.stderr

            case OSError():
                reason = ExecFailReason.SPAWN_FAILED
                base_kwargs["stderr"] = str(e)

            case _:
                raise

        raise ExecError(reason=reason, **base_kwargs) from e
