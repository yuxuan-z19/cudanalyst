# Reference: https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/utils/code_utils.py#L95
import json
import re
from typing import Any

import tiktoken


def extract_codeblock(text: str) -> str:
    text = text.strip()
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return text
    return max(matches, key=len).strip()


def render_feedback_md(feedback: dict[str, Any]) -> str:
    sections = []
    for key, value in feedback.items():
        pretty_json = json.dumps(value, indent=2, ensure_ascii=False)
        sections.append(f"### {key}\n\n ```json\n{pretty_json}\n```")
    return "\n\n".join(sections)


# Reference: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
def ntoken(message: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(message)
    return len(tokens)


def is_valid_str(code: str) -> bool:
    return bool(code and code.strip())
