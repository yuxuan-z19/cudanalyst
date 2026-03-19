import asyncio
import time
from pathlib import Path
from pprint import pprint

from benchmark.hpc.npb.eval import evaluate
from cudanalyst.helper.ckpt import group_llm4ad_by_gen, group_oe_by_gen
from cudanalyst.helper.stat import compute_label_stats
from cudanalyst.module.config import ModuleBits
from cudanalyst.pipeline.config import AnalysisMask
from cudanalyst.workflow import intervene_async

# TODO: change to your own keyset
CONFIG_FILE = Path("./config/keyset_template.yml")

# ^ OpenEvolve-based trajectory
# ? path to the OpenEvolve output checkpoints
SRC_CKPT_DIR = Path("./out/npb-CG/checkpoints")
# ? path to keep the generation-level samples
DST_GEN_DIR = Path("./tmp/npb-CG")
group_oe_by_gen(SRC_CKPT_DIR, DST_GEN_DIR)

# ^ LLM4AD-based trajectory
# SRC_SAMPLE_RECORD = Path("./out/eoh-3MM/samples/samples_1~200.json")
# DST_GEN_DIR = Path("./tmp/eoh-3MM")
# group_llm4ad_by_gen(SRC_SAMPLE_RECORD, DST_GEN_DIR)

MASK_LIST = {
    # ? Plan only (P)
    "plan-only": AnalysisMask(plan=ModuleBits.MODE_FULL),
    # ? Plan + Feedback (P+F)
    "plan-feedback": AnalysisMask(
        ModuleBits.MODE_RAW,
        ModuleBits.MODE_RAW,
        ModuleBits.MODE_RAW,
        ModuleBits.MODE_FULL,
    ),
    # ? Plan + Summary (P+S)
    "plan-summary": AnalysisMask(
        ModuleBits.MODE_FULL,
        ModuleBits.MODE_FULL,
        ModuleBits.MODE_FULL,
        ModuleBits.MODE_FULL,
    ),
}


def work():
    problem_dir = Path("benchmark/hpc/npb/src/CG")
    output_root = Path("gen/npb-CG")
    for name, mask in MASK_LIST.items():
        print(f"Running analysis mode: {name}")
        res_list = asyncio.run(
            intervene_async(
                evaluate_func=evaluate,
                input_ckpt_dir=DST_GEN_DIR,
                output_root_dir=output_root,
                chat_config_path=CONFIG_FILE,
                config_mask=mask,
                num_run=3,  # * rollout
                num_trials=5,  # * k
                max_workers=32,
                llm_concurrency=16,
                problem_dir=problem_dir,
            )
        )
        res_stat = compute_label_stats(res_list, ["fast", "pass"], as_df=True)
        pprint(res_stat)
        time.sleep(10)


if __name__ == "__main__":
    work()
