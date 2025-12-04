# LLM Code Generation Benchmarks


## KernelBench (kbench)

[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) is the first benchmark for evaluating LLMs' ability to generate efficient GPU kernels in either CUDA kernels or DSL directives (in progress).

```bash
cd ./kbench/kernelbench
uv pip install -e .
```

## Robust-Kbench (rkbench)

[SakanaAI/robust-kbench](https://github.com/SakanaAI/robust-kbench) is a new benchmark for more rigorous evaluation of kernel performance and correctness across varied scenarios.

```bash
cd ./rkbench/robust_kbench
uv pip install -e .
```

## compute-eval (cueval)

[NVIDIA/compute-eval](https://github.com/NVIDIA/compute-eval) is a *[correctness-first](https://github.com/NVIDIA/compute-eval/issues/7)* benchmark harnessing the compilability of LLM-generated CUDA kernels.