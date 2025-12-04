from pprint import pp
from typing import Any

ERROR_FEEDBACK = r"""# Compilation or Runtime Errors
You are given CUDA-related error logs produced by tools such as NVCC, NVRTC, driver JIT, or runtime modules. 
Interpret the log strictly as signals about compilation, linking, or runtime correctness. 
Propose minimal, concrete code-level fixes for CUDA C++ and PTX. Avoid hypothetical changes not supported by the error details.

1. Locate the exact failure indicated in the log.  
2. State the cause using only information present in the log.  
3. Suggest minimal code-level changes that resolve it.

```log
<error>
```
"""

ANALYST_FEEDBACK = r"""# Performance Diagnostics
You are given a JSON diagnostics report produced by NVIDIA tools such as Nsight Compute or Compute Sanitizer. 
Interpret the report strictly as performance or correctness signals and propose minimal, concrete code-level transformations for CUDA C++ and PTX.

1. Identify the bottleneck or hazard in the report.  
2. Explain its effect using only the provided metrics.  
3. Suggest minimal code-level transformations targeting it.

```json
<report>
```
"""


def parse_feedback(metrics: dict[str, Any]) -> str:
    feedback_parts = []
    replacements = {
        "error": ("<error>", ERROR_FEEDBACK),
        "report": ("<report>", ANALYST_FEEDBACK),
    }
    for key, (placeholder, template) in replacements.items():
        if value := metrics.get(key):
            feedback_parts.append(template.replace(placeholder, value))
    return "\n".join(feedback_parts)


if __name__ == "__main__":
    metrics = {}
    print(">>> Raw")
    pp(parse_feedback(metrics))
    print("=" * 16)

    print(">>> Error")
    metrics["error"] = "Error: xxxx"
    pp(parse_feedback(metrics))
    print("=" * 16)

    print(">>> Error + Empty Report")
    metrics["report"] = ""
    pp(parse_feedback(metrics))
    print("=" * 16)

    print(">>> Error + Report")
    metrics["report"] = "mock a report"
    pp(parse_feedback(metrics))
    print("=" * 16)

    print(">>> Report")
    metrics.pop("error")
    pp(parse_feedback(metrics))
    print("=" * 16)
