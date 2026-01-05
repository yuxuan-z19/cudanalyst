import asyncio
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import yaml
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as CompleteChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
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
    stream: bool

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
    def __init__(self, config_pth: os.PathLike):
        config_file = Path(config_pth)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found at {config_pth}")

        with config_file.open(encoding="utf-8") as f:
            try:
                config: dict = yaml.safe_load(f) or {}
            except Exception as e:
                raise ValueError(f"Invalid YAML format: {e}")

        api = Interface(
            url=config.get("api_base"),
            key=config.get("api_key"),
            timeout=config.get("timeout", 3 * 60),
            max_tokens=config.get("max_tokens", None),
            stream=config.get("stream", False),
        )

        candidates = config.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError("candidates must be a list")

        self.candidates = [Service(name=name, api=api) for name in candidates]

    @property
    def all_services(self) -> list[str]:
        return [s.name for s in self.candidates]

    def __getitem__(self, index: int | slice) -> Service:
        try:
            return self.candidates[index]
        except IndexError:
            raise IndexError(
                f"Service index {index} out of range (total {len(self.candidates)})"
            )

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)


class ChatState:
    def __init__(self, sys_msg: Message | None = None):
        self.sys_msg = sys_msg
        self.history: list[Message] = []

    def reset(self):
        self.history.clear()

    def clone(self) -> "ChatState":
        return deepcopy(self)

    def build_messages(self, user_msg: Message, use_history: bool) -> list[dict]:
        msgs: list[Message] = []
        if self.sys_msg:
            msgs.append(self.sys_msg)
        if use_history:
            msgs.extend(self.history)
        msgs.append(user_msg)
        return [asdict(m) for m in msgs]

    def commit(self, user_msg: Message, reply: str):
        self.history.extend([user_msg, Message(Role.LLM, reply)])


class ChatTransport:
    def __init__(self, service: Service):
        kwargs = {
            "base_url": service.api.url,
            "api_key": service.api.key,
            "timeout": service.api.timeout,
        }
        self.service = service
        self.use_stream = self.service.api.stream

        self.client = OpenAI(**kwargs)
        self.async_client = AsyncOpenAI(**kwargs)

    def _extract_reply(
        self, response: ChatCompletion | Iterable[ChatCompletionChunk]
    ) -> str:
        answer_content = ""
        if self.use_stream:
            for chunk in response:
                choices: list[ChunkChoice] = getattr(chunk, "choices", [])
                if not choices:
                    continue
                delta = choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    answer_content += content
        else:
            choices: list[CompleteChoice] = getattr(response, "choices", [])
            for choice in choices:
                content = getattr(choice.message, "content", None)
                if content:
                    answer_content += content.strip()

        if not answer_content:
            raise RuntimeError("Empty completion from model")

        return answer_content

    def _build_request(self, messages: list[dict]) -> dict:
        kwargs = {"model": self.service.name, "messages": messages}
        if self.service.api.max_tokens is not None:
            kwargs["max_tokens"] = self.service.api.max_tokens
        if self.use_stream:
            kwargs["stream"] = True
        return kwargs

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    def complete(self, messages: list[dict]) -> str:
        req_kwargs = self._build_request(messages)
        resp = self.client.chat.completions.create(**req_kwargs)
        return self._extract_reply(resp)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), reraise=True)
    async def complete_async(self, messages: list[dict]) -> str:
        req_kwargs = self._build_request(messages)
        resp = await self.async_client.chat.completions.create(**req_kwargs)
        return self._extract_reply(resp)


class ChatSession:
    def __init__(self, service: Service, sys_prompt: str = None):
        sys_msg = Message(Role.SYSTEM, sys_prompt) if sys_prompt else None

        self.state = ChatState(sys_msg)
        self.transport = ChatTransport(service)

        self._async_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self.transport.model

    def reset(self):
        self.state.reset()

    def ask(self, prompt: str, use_history: bool = False) -> str:
        user_msg = Message(Role.USER, prompt)
        messages = self.state.build_messages(user_msg, use_history)

        reply = self.transport.complete(messages)
        self.state.commit(user_msg, reply)
        return reply

    async def ask_async(self, prompt: str, use_history: bool = False) -> str:
        async with self._async_lock:
            user_msg = Message(Role.USER, prompt)
            messages = self.state.build_messages(user_msg, use_history)

            reply = await self.transport.complete_async(messages)
            self.state.commit(user_msg, reply)
            return reply
