# API Config

The configuration file uses YAML format and supports multiple namespaces (e.g., analyst and coder). Example:

```yaml
# keyset template

main:
  candidates:
    - xxxx
  api_base: "http://xxx.yyy.zzz"
  api_key: "zzzzzzzz"
  timeout: 65536
  max_tokens: 32768
```

- `candidates`: List of candidate model IDs (e.g., gpt-4, qwen-max).
- `api_base`: Base URL of the model service.
- `api_key`: Authorization key for accessing the service.
- `timeout`: Request timeout (in seconds).
- `max_tokens`: Maximum number of tokens to generate.

`main` is an example service partition. You can freely define other names depending on your use case.

```python
@dataclass
class Interface:
    url: str        # api_base
    key: str        # api_key
    timeout: int
    max_tokens: int


@dataclass
class Service:
    name: str       # model ID
    api: Interface
```

## Usage

```python
import asyncio
from pathlib import Path

from cudanalyst.module.chat import ChatConfig, ChatSession

keyset_pth = Path(__file__).parent.parent / "config/keyset.yml"
service = ChatConfig(keyset_pth)[0]
print(service)

# sync
session = ChatSession(service, "You are a helpful assistant.")
ans = session.ask("What is your name?")
print(ans)

# async
async def test_async(session: ChatSession):
    ans = await session.ask_async("What is your name?")
    print(ans)

asyncio.run(test_async(session))
```