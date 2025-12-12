# LLM Code Generation Benchmarks

## KernelBench (kbench)

[ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench) is the first benchmark for evaluating LLMs' ability to generate efficient GPU kernels in either CUDA kernels or DSL directives (in progress).

```bibtex
@inproceedings{kbench,
    title        = {KernelBench: Can {LLM}s Write Efficient {GPU} Kernels?},
    author       = {
        Anne Ouyang and Simon Guo and Simran Arora and Alex L Zhang and William
        Hu and Christopher Re and Azalia Mirhoseini
    },
    year         = 2025,
    booktitle    = {Forty-second International Conference on Machine Learning},
    url          = {https://openreview.net/forum?id=yeoN1iQT1x}
}
```

## Robust-Kbench (rkbench)

[SakanaAI/robust-kbench](https://github.com/SakanaAI/robust-kbench) is a new benchmark for more rigorous evaluation of kernel performance and correctness across varied scenarios.

```bibtex
@misc{rkbench,
    title        = {
        Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and
        Optimization
    },
    author       = {
        Robert Tjarko Lange and Qi Sun and Aaditya Prasad and Maxence Faldor
        and Yujin Tang and David Ha
    },
    year         = 2025,
    url          = {https://arxiv.org/abs/2509.14279},
    eprint       = {2509.14279},
    archiveprefix = {arXiv},
    primaryclass = {cs.SE}
}
```

## compute-eval (cueval)

[NVIDIA/compute-eval](https://github.com/NVIDIA/compute-eval) is a *[correctness-first](https://github.com/NVIDIA/compute-eval/issues/7)* benchmark harnessing the compilability of LLM-generated CUDA kernels.