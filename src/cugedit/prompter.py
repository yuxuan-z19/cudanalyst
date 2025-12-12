import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import yaml
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    LLM = "assistant"


@dataclass(frozen=True)
class Interface:
    url: str
    key: str
    timeout: int
    max_tokens: int

    def __post_init__(self):
        if not self.url or not self.key:
            raise ValueError("API url and key must be provided")


@dataclass(frozen=True)
class Service:
    name: str
    api: Interface


@dataclass
class Message:
    role: Role
    content: str


class ChatConfig:
    def __init__(self, config_pth: os.PathLike, select: str):
        config_file = Path(config_pth)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_pth}")

        with config_file.open(encoding="utf-8") as f:
            try:
                root: dict = yaml.safe_load(f) or {}
            except Exception as e:
                raise ValueError(f"Invalid YAML format: {e}")

        config = root.get(select)
        if not isinstance(config, dict):
            raise ValueError(f"Config for '{select}' not found")

        api = Interface(
            url=config.get("api_base"),
            key=config.get("api_key"),
            timeout=config.get("timeout", 30),
            max_tokens=config.get("max_tokens", 1024),
        )

        candidates = config.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")

        self.candidates = [Service(name=name, api=api) for name in candidates]

    @property
    def all_services(self) -> list[str]:
        return [s.name for s in self.candidates]


class ChatSession:
    def __init__(self, service: Service, sys_prompt: str | None = None):
        self.service = service
        self.client = OpenAI(
            base_url=service.api.url,
            api_key=service.api.key,
            timeout=service.api.timeout,
            max_retries=0,  # retry 交给 tenacity
        )

        self._system: Message | None = None
        self._history: list[Message] = []

        if sys_prompt and sys_prompt.strip():
            self.set_sys_prompt(sys_prompt)

    @property
    def name(self) -> str:
        return self.service.name

    def set_sys_prompt(self, prompt: str):
        self._system = Message(Role.SYSTEM, prompt)

    def reset_history(self):
        self._history.clear()

    def _build_messages(self, user_msg: Message, new_session: bool) -> list[dict]:
        msgs: list[Message] = []
        if self._system:
            msgs.append(self._system)
        if not new_session:
            msgs.extend(self._history)
        msgs.append(user_msg)
        return [asdict(m) for m in msgs]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        reraise=True,
    )
    def _ask_impl(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.service.name,
            messages=messages,
            max_tokens=self.service.api.max_tokens,
        )
        for choice in response.choices:
            if content := ((choice.message.content or "").strip()):
                return content

        raise RuntimeError("Empty completion from model")

    def ask(self, prompt: str, new_session: bool = False) -> str:
        user_msg = Message(Role.USER, prompt)
        messages = self._build_messages(user_msg, new_session)

        reply = self._ask_impl(messages)

        self._history.append(user_msg)
        self._history.append(Message(Role.LLM, reply))
        return reply
