import json
import statistics as stats
from collections import defaultdict
from math import factorial
from pathlib import Path

import numpy as np


def stats_med(result: list):
    return stats.median(result)


def elemwise_equal(
    a_list: list | np.ndarray, b_list: list | np.ndarray, rtol=1e-9, atol=0.0
) -> bool:
    a = np.asarray(a_list)
    b = np.asarray(b_list)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        return np.array_equal(a, b)


def compute_label_stats(
    res_list: list[dict[int, dict[str, int]]], labels: list[str], as_df: bool = False
):
    gen_stats = {}
    for gen in res_list[0].keys():
        ratios = [
            sum(res[gen].get(label, 0) for label in labels) / max(res[gen]["total"], 1)
            for res in res_list
        ]
        gen_stats[gen] = {"mean": np.mean(ratios), "std": np.std(ratios)}

    if as_df:
        import pandas as pd

        return pd.DataFrame.from_dict(gen_stats, orient="index")

    return gen_stats


class ToolContribAnlz:
    TOOLS = ["d", "a", "p"]
    TOOL_IDX = {t: i for i, t in enumerate(TOOLS)}

    def __init__(self, base_path=None, final_json=None):
        if final_json:
            self.data = json.loads(Path(final_json).read_text())
        elif base_path:
            self.data = self._preprocess(base_path)
        else:
            raise ValueError(
                "Please provide base path to the intervention results or the final json file."
            )

    @staticmethod
    def mask_to_suffix(mask):
        d = 3 if (mask & 0b001) else 0
        a = 3 if (mask & 0b010) else 0
        p = 3 if (mask & 0b100) else 0
        return f"d{d}-a{a}-p{p}"

    @staticmethod
    def _load_stats(json_path):
        data = json.loads(Path(json_path).read_text())

        compile_rates = defaultdict(list)
        pass_rates = defaultdict(list)
        fast_rates = defaultdict(list)

        for run in data:
            for gen, stats in run.items():
                total = stats.get("total", 1)
                compile_rates[gen].append(stats.get("compile", 0) / total)
                pass_rates[gen].append(stats.get("pass", 0) / total)
                fast_rates[gen].append(stats.get("fast", 0) / total)

        def mean_std(rates):
            return {
                int(gen): (
                    float(np.mean(v)),
                    float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                )
                for gen, v in rates.items()
            }

        return {
            "compile": mean_std(compile_rates),
            "pass": mean_std(pass_rates),
            "fast": mean_std(fast_rates),
        }

    def _preprocess(self, base_path):
        base_path = Path(base_path)
        all_stats = {}
        for mask in range(8):
            suffix = self.mask_to_suffix(mask)
            json_file = base_path / f"p7-{suffix}.json"
            if not json_file.exists():
                raise FileNotFoundError(json_file)
            all_stats[suffix] = self._load_stats(json_file)
        (base_path / "final.json").write_text(json.dumps(all_stats))
        return all_stats

    @staticmethod
    def mask_to_key(mask):
        vals = []
        for i, t in enumerate(ToolContribAnlz.TOOLS):
            vals.append(f"{t}{3 if (mask & (1 << i)) else 0}")
        return "-".join(vals)

    def v(self, mask, metric, gen):
        key = self.mask_to_key(mask)
        return self.data[key][metric][str(gen)][0]

    def shapley(self, metric, gen):
        n = len(self.TOOLS)
        phi = {t: 0.0 for t in self.TOOLS}

        for t in self.TOOLS:
            i = self.TOOL_IDX[t]
            for mask in range(1 << n):
                if mask & (1 << i):
                    continue
                S = mask
                S_with_i = mask | (1 << i)
                s = bin(S).count("1")
                weight = factorial(s) * factorial(n - s - 1) / factorial(n)
                marginal = self.v(S_with_i, metric, gen) - self.v(S, metric, gen)
                phi[t] += weight * marginal

        return phi

    def full_synergy(self, metric, gen):
        V = lambda mask: self.v(mask, metric, gen)

        v0 = V(0b000)
        vd = V(0b001)
        va = V(0b010)
        vp = V(0b100)
        vda = V(0b011)
        vdp = V(0b101)
        vap = V(0b110)
        vdap = V(0b111)

        syn_3 = vdap - vda - vdp - vap + vd + va + vp - v0
        syn_da = vda - vd - va + v0 - syn_3 / 3
        syn_dp = vdp - vd - vp + v0 - syn_3 / 3
        syn_ap = vap - va - vp + v0 - syn_3 / 3

        return {
            "pairwise": {("d", "a"): syn_da, ("d", "p"): syn_dp, ("a", "p"): syn_ap},
            "triple": syn_3,
        }

    def analyze_metric(self, metric, gen):
        return {
            "shapley": self.shapley(metric, gen),
            "synergy": self.full_synergy(metric, gen),
        }

    def analyze_metric_all_gens(self, metric):
        first_key = next(iter(self.data))
        gens = sorted(int(g) for g in self.data[first_key][metric].keys())
        return {gen: self.analyze_metric(metric, gen) for gen in gens}

    def coalitions_to_ndarray(self, metric):
        all_res = self.analyze_metric_all_gens(metric)
        first_key = next(iter(self.data))
        gens = sorted(int(g) for g in self.data[first_key][metric].keys())

        coalitions = ["d", "a", "p", "d+a", "d+p", "a+p", "d+a+p"]
        out = {c: np.zeros(len(gens)) for c in coalitions}

        for i, gen in enumerate(gens):
            res = all_res[gen]
            # Shapley
            out["d"][i] = res["shapley"]["d"]
            out["a"][i] = res["shapley"]["a"]
            out["p"][i] = res["shapley"]["p"]
            # Pairwise synergy
            out["d+a"][i] = res["synergy"]["pairwise"][("d", "a")]
            out["d+p"][i] = res["synergy"]["pairwise"][("d", "p")]
            out["a+p"][i] = res["synergy"]["pairwise"][("a", "p")]
            # Triple synergy
            out["d+a+p"][i] = res["synergy"]["triple"]

        return gens, out

    @staticmethod
    def print_results(res):
        print("Shapley:")
        for k, v in res["shapley"].items():
            print(f"  {k}: {v:.4f}")

        print("\nPairwise synergy:")
        for (t1, t2), v in res["synergy"]["pairwise"].items():
            print(f"  {t1}+{t2}: {v:.4f}")

        print("\nTriple synergy:")
        print(f"  d+a+p: {res['synergy']['triple']:.4f}")
