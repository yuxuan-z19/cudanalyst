# CUDAnalyst

**CUDAnalyst** (CUDA + Analyst) is a unified analysis layer designed for controlled, generation-level attribution of planning decisions in self-evolving LLM agents for CUDA kernel generation. It decouples feedback from planning and enables principled evaluation of how heterogeneous feedback signals shape planning trajectories.

## Motivation

Large language models (LLMs) can act as self-evolving agents for CUDA kernel generation, guided by feedback-conditioned planning. However, the internal mechanisms by which feedback signals are combined and propagated across planning steps remain opaque. Standard end-to-end ablations often fail to disambiguate these effects. CUDAnalyst addresses this gap by providing an intervention-based, interpretable analysis framework.

## Installation

```bash
git clone https://github.com/yuxuan-z19/cudanalyst.git
cd cudanalyst
# install PyTorch with appropriate CUDA backend
uv pip install torch --torch-backend=cu124
# for simple usage
uv pip install -e .
# for development
uv sync --dev
```

Ensure you have installed **Nsight Compute >= 2025.2.1** and added the absolute path of its `extras/python` directory to the `$PYTHONPATH` environment variable.

```bash
❯ export PYTHONPATH="/data/zyx/local/nsight-compute-2025.2.1/extras/python:$PYTHONPATH"
❯ python -c "import ncu_report;"
❯ echo $?
0
```

## Usage

This project supports three main workflows:

1. [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve)-based evolution
2. [LLM4AD](https://github.com/Optima-CityU/LLM4AD)-based evolution
3. Generation-level Intervention (multi-rollout intervention on evolution trajectories)

The recommended order is: Run evolution $\rightarrow$ Group by generation $\rightarrow$ Perform intervention $\rightarrow$ Compute statistics

### OpenEvolve evolution

Install the modified OpenEvolve fork:

```bash
git submodule update --init --recursive
cd ./eval/openevolve && uv pip install -e .
```

Example using the [NPB-GPU CG](./benchmark/hpc/npb/src/CG) workload is provided in `./eval/demo/oe-npb.bash`:

```bash
SUITE_ROOT="benchmark/hpc/npb"
WORKLOAD="CG"
WORKLOAD_DIR=$SUITE_ROOT/src/$WORKLOAD

python eval/openevolve/openevolve-run.py \
    $WORKLOAD_DIR/sol.init.cu \
    $SUITE_ROOT/eval.py \
    -c eval/config/openevolve.yml \
    -o out/npb-$WORKLOAD \
    -p $WORKLOAD_DIR \
    -a config/cudanalyst_template.yml \
```

Compared to the original OpenEvolve, two new options are introduced:

- `-p`: Specifies the subtask evaluation directory.

- `-a`: Specifies the CUDAnalyst config file path. Default: None

These two options are passed to the task-specific `evaluate()` as arguments.

After execution, the output directory will look like:

```bash
out/npb-CG/
├── best/
├── checkpoints/
└── logs/
```

The `checkpoints/` directory will be used in the generation-level intervention stage.

### LLM4AD evolution

Install the modified LLM4AD fork:

```bash
git submodule update --init --recursive
cd ./eval/llm4ad
uv pip install -r requirements.txt
uv pip install -e .
```

For usage, please refer to [LLM4AD docs](https://llm4ad-doc.readthedocs.io/en/latest/index.html). We provide a example on [PolyBench-ACC 3MM](./benchmark/cgo/poly/src/3MM/) with [EoH](https://llm4ad-doc.readthedocs.io/en/latest/method/eoh.html) algorithm in `./eval/demo/llm4ad_eoh.py`.

After execution, the output directory will look like:

```bash
out/eoh-3MM/
├── population/
├── run_log.txt
└── samples/
    ├── samples_1~200.json
    └── samples_best.json
```

The `samples_1~200.json` file which keeps all the kernels generated will be used in the generation-level stage.

### Generation-level Intervention

This workflow enables:

1. Grouping evolution results by generation
2. Performing multi-rollout intervention on each generation
3. Computing pass@$k$-like statistics

See `demo.py` for a minimal example.

****

1. Group evolution outputs by generation:

    - For OpenEvolve output:

        ```python
        from cudanalyst.helper.ckpt import group_oe_by_gen

        # ? path to the OpenEvolve output checkpoints
        SRC_CKPT_DIR = Path("./out/npb-CG/checkpoints")
        # ? path to keep the generation-level samples
        DST_GEN_DIR = Path("./tmp/npb-CG")

        group_oe_by_gen(SRC_CKPT_DIR, DST_GEN_DIR)
        ```

    - For LLM4AD output:

        ```python
        from cudanalyst.helper.ckpt import group_llm4ad_by_gen

        SRC_SAMPLE_RECORD = Path("./out/eoh-3MM/samples/samples_1~200.json")
        DST_GEN_DIR = Path("./tmp/eoh-3MM")

        group_llm4ad_by_gen(SRC_SAMPLE_RECORD, DST_GEN_DIR)
        ```
    
    Resulting sturcture looks like:

    ```bash
    tmp/npb-CG/
    ├── gen0/
    ├── gen1/
        ...
    ```

2. Use the template `./config/keyset_template.yml` to configure your LLM API key, endpoint, and model. Make sure the `chat_config_path` variable points to this file.

3. Define the `AnalysisMask` to control the feedback granularity of each module.

    ```python
    from cudanalyst.module.config import ModuleBits
    from cudanalyst.pipeline.config import AnalysisMask

    mask = AnalysisMask(
        debug=ModuleBits.MODE_FULL,
        anlz=ModuleBits.MODE_FULL,
        perf=ModuleBits.MODE_FULL,
        plan=ModuleBits.MODE_FULL,
    )
    ```

    - `MODE_FULL`: Structured summarized feedback (recommended)
    - `MODE_RAW`: Raw, unprocessed feedback
    - `MODE_NONE`: Disable the module

    This allows controlled ablation experiments.

4. Launch async multi-rollout intervention.

    ```python
    from cudanalyst.workflow import intervene_async

    problem_dir = Path("benchmark/hpc/npb/src/CG")
    output_root = Path("gen/npb-CG")
    res_list = asyncio.run(
        intervene_async(
            evaluate_func=evaluate,
            input_ckpt_dir=DST_GEN_DIR,
            output_root_dir=output_root,
            chat_config_path=CONFIG_FILE,
            config_mask=mask,
            num_run=3,    # * rollout
            num_trials=5,  # * k
            max_workers=32,
            llm_concurrency=16,
            problem_dir=problem_dir,
        )
    )
    ```

    - `num_run`: Number of rollouts per evaluation
    - `num_trials`: Used for pass@k computation
    - `max_workers`: Local parallel workers
    - `llm_concurrency`: Concurrent LLM API calls

    Output structure:

    ```bash
    gen/npb-CG
    ├── p7-d7-a7-p7
    │   ├── run-0
    │   ├── run-1
    │   └── run-2
    └── p7-d7-a7-p7.json
    ```

    You may use `intervene_async_multi()` to launch concurrent evaluation on different feedback granularity when `num_run` is small.

5. Compute statistics.

    ```python
    from cudanalyst.helper.stat import compute_label_stats

    res_stat = compute_label_stats(res_list, ["fast", "pass"], as_df=True)
    ```

    It returns a dictionary or a `pd.DataFrame` when `as_df=True`. Supported labels are as defined in `class Status(str, Enum)`:

    ```python
    # src/cudanalyst/helper/exec.py

    class Status(str, Enum):
        FAIL = "fail"
        COMPILE = "compile"
        PASS = "pass"

        # only be set with check_fast()
        FAST = "fast"

        @property
        def rank(self):
            _ranks = {Status.FAIL: 0, Status.COMPILE: 1, Status.PASS: 2, Status.FAST: 3}
            return _ranks.get(self, -1)

        def __lt__(self, other):
            if self.__class__ is other.__class__:
                return self.rank < other.rank
            return NotImplemented
    ```

    Can be used for calculating pass@$k$ and plotting generation-level curves.

#### Known Issues

- Grouping may overwrite early attempts if a program sample is invoked multiple times during the evolution.

### DIY Evaluator

We adopt an OpenEvolve-style evaluator for convenience (see [docs](https://github.com/algorithmicsuperintelligence/openevolve/tree/main/examples#2-evaluator-evaluatorpy)).

#### 1. Create your own evaluator

Each benchmark suite under the `./benchmark` directory provides a reference eval.py. An evaluator typically defines:

```python
from cudanalyst.result import *

@dataclass
class YourMeta(ResultMeta):
    # add custom metadata fields here

@dataclass
class YourResult(Result):
    base_result: YourMeta = None
    custom_result: YourMeta = None
    # add additional statistics if needed

def _exec(...) -> YourMeta:
    # implement execution logic here
    ...

@return_asdict
def evaluate(program_path: os.PathLike, problem_dir: os.PathLike, config: AnalysisCfg = None):
    program_path = Path(program_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        dst_path = tmp_path / "src"
        shutil.copytree(SRC_DIR, dst_path)

        code_path = dst_path / "Solution.cu"
        code_path.write_text(extract_codeblock(program_path.read_text()))

        ctx = ToolContext(code_path=code_path, cwd=dst_path)

        gpu_id = pick_idle_gpu()

        base_result = _exec(...)
        custom_result = _exec(...)
        
        score = ... # compute combined score
        return YourResult(
            combined_score=score,
            base_result=base_result,
            custom_result=custom_result
        )
```

#### 2. Integrate CUDAnalyst

You can plug CUDAnalyst into your evaluator via planning():

```python
from cudanalyst import AnalysisCfg, ToolContext, planning

def _exec(...):
    config = AnalysisCfg()
    ctx = ToolContext(
        code_path=(task_dir / task_name / "sol.cu"), 
        cwd=task_dir,
    )
    ctx.cmd = ["make", task_name, "CLASS=S"]  # same format as subprocess.run
    return YourResult(reports=planning(config, ctx))
```

##### Configuring `AnalysisCfg`

Two options:

- Load from YAML (template: `./config/cudanalyst_template.yml`)

    ```yaml
    chat_config_path: "./config/keyset.yml"

    debug_cfg:
        enabled: true
        formatted: true
        summarized: true

    anlz_cfg:
        enabled: true
        formatted: true
        summarized: true

    perf_cfg:
        enabled: true
        formatted: true
        summarized: true

    plan_cfg:
        enabled: true
        summarized: true
    ```

    ```python
    from cudanalyst import load_analysis_cfg, AnalysisCfg

    config: AnalysisCfg = load_analysis_cfg("./config/cudanalyst_template.yml")
    ```

- Apply a bitmask

    ```python
    from cudanalyst import load_analysis_cfg, AnalysisCfg, AnalysisMask
    from cudanalyst.module.config import ModuleBits

    config_mask = mask = AnalysisMask(
        debug=ModuleBits.MODE_RAW,   # raw feedback, no summary
        anlz=ModuleBits.MODE_FULL,   # summarized feedback by an agent
        perf=ModuleBits.MODE_NONE    # disabled
        plan=ModuleBits.MODE_FULL,   # explicit planning
    )

    config = apply_config_mask(AnalysisCfg(chat_config_path), config_mask)
    ```

##### Setting `ToolContext`

- `code_path`: path to the source code to analyze
- `cmd`: command to be executed (list of strings, same format as `subprocess.run`)
- `cwd`: working directory for the analysis

## Contributing

We welcome contributions, especially for porting existing CUDA benchmarks or adding new agentic frameworks and analysis pipelines.

### Development Setup

This repository uses [`uv`](https://github.com/algorithmicsuperintelligence/uv) to manage dependencies and development environment, and [`pre-commit`](https://pre-commit.com/) to enforce code style and formatting rules.

To set up your environment and enable pre-commit hooks:

```bash
uv sync --dev      # install/update development dependencies
pre-commit install # enable pre-commit hooks for code style
```

After this, any committed code will automatically be checked and formatted according to the project's standards.
