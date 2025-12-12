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
from src.prompter import Configurer

# Load the "analyst" configuration
config = Configurer("./config/keyset.yml", "analyst")

# Assess the first "analyst" candidate
service = config.candidates[0]
print(service.api.url)
print(service.api.timeout)
print(service.api.run_once)

# Ask in batch
client = Prompter(config.candidates[0], "You are a helpful assistant.")
replies = await client.ask("What is 6 * 7?", n_attempt=3, n_sol=4)
```