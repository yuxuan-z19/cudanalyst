import statistics as stats

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
