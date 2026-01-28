# Evaluation

We conduct a generalization study on the following open-source self-evolving agents. Our fork's default branch (`cudanalyst`) is continuously synchronized with the upstream `main` branch. All fork-specific changes (listed in reverse chronological order) that may affect agent behavior are explicitly documented below for reproducibility.

Scripts and configs are provided at `./config` and `./scripts` respectively

## OpenEvolve (openevolve)

[algorithmicsuperintelligence/OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) is the open-source implementation of [AlphaEvolve](https://arxiv.org/abs/2506.13131), a Google DeepMind's self-evolving agent.

```bibtex
@software{openevolve,
	title        = {OpenEvolve: an open-source evolutionary coding agent},
	author       = {Asankhaya Sharma},
	year         = 2025,
	publisher    = {GitHub},
	url          = {https://github.com/algorithmicsuperintelligence/openevolve}
}

@misc{alphaevolve,
    title        = {AlphaEvolve: A coding agent for scientific and algorithmic discovery},
    author       = {
        Alexander Novikov and Ngân Vũ and Marvin Eisenberger and Emilien Dupont
        and Po-Sen Huang and Adam Zsolt Wagner and Sergey Shirobokov and
        Borislav Kozlovskii and Francisco J. R. Ruiz and Abbas Mehrabian and M.
        Pawan Kumar and Abigail See and Swarat Chaudhuri and George Holland and
        Alex Davies and Sebastian Nowozin and Pushmeet Kohli and Matej Balog
    },
    year         = 2025,
    url          = {https://arxiv.org/abs/2506.13131},
    eprint       = {2506.13131},
    archiveprefix = {arXiv},
    primaryclass = {cs.AI}
}
```

- Commit [d609ff8](https://github.com/yuxuan-z19/openevolve/commit/d609ff8): Render plan decisions as artifacts
- Commit [df9fb3c](https://github.com/yuxuan-z19/openevolve/commit/df9fb3c): Add `analyst_config` in `EvaluatorConfig`
- Commit [d09052b](https://github.com/yuxuan-z19/openevolve/commit/d09052b): Add `problem_dir` in `EvaluatorConfig`

## LLM4AD (llm4ad)

[Optima-CityU/LLM4AD](https://github.com/Optima-CityU/LLM4AD) is an open-source Python-based Platform leveraging Large Language Models (LLMs) for Automatic Algorithm Design (AD).

```bibtex
@misc{llm4ad,
    title        = {LLM4AD: A Platform for Algorithm Design with Large Language Model},
    author       = {
        Fei Liu and Rui Zhang and Zhuoliang Xie and Rui Sun and Kai Li and Xi
        Lin and Zhenkun Wang and Zhichao Lu and Qingfu Zhang
    },
    year         = 2024,
    url          = {https://arxiv.org/abs/2412.17287},
    eprint       = {2412.17287},
    archiveprefix = {arXiv},
    primaryclass = {cs.AI}
}
```

- Commit [d444db5](https://github.com/yuxuan-z19/LLM4AD/commit/d444db5): Record generation and prompt
