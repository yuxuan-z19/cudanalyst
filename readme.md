# CUDAnalyst

**CUDAnalyst** (CUDA + Analyst) is a unified analysis layer designed for controlled, generation-level attribution of planning decisions in self-evolving LLM agents for CUDA kernel generation. It decouples feedback from planning and enables principled evaluation of how heterogeneous feedback signals shape planning trajectories.

## Motivation

Large language models (LLMs) can act as self-evolving agents for CUDA kernel generation, guided by feedback-conditioned planning. However, the internal mechanisms by which feedback signals are combined and propagated across planning steps remain opaque. Standard end-to-end ablations often fail to disambiguate these effects. CUDAnalyst addresses this gap by providing an intervention-based, interpretable analysis framework.

## Installation

```bash
git clone https://github.com/yuxuan-z19/cudanalyst.git
cd cudanalyst
# for simple usage
uv pip install -e .
# for development
uv sync --dev
```

## Usage

<!-- TODO: add guidance -->
