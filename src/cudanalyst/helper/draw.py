import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import t


def plot_dist_per_run(runs):
    OUTCOMES = ["fail", "compile", "pass", "fast"]
    COLORS = {
        "fail": "#A2090C",
        "compile": "#E88D2F",
        "pass": "#1159AA",
        "fast": "#3BA595",
    }

    num_runs = len(runs)

    fig, axes = plt.subplots(
        num_runs,
        1,
        figsize=(3.5, 0.9 * num_runs),
        sharex=True,
        sharey=True,
    )

    if num_runs == 1:
        axes = [axes]

    for idx, (ax, run_stats) in enumerate(zip(axes, runs)):
        generations = sorted(run_stats.keys())
        fractions = {k: [] for k in OUTCOMES}

        for g in generations:
            total = run_stats[g]["total"]
            for k in OUTCOMES:
                fractions[k].append(
                    run_stats[g].get(k, 0) / total if total > 0 else 0.0
                )

        bottom = [0.0] * len(generations)
        for k in OUTCOMES:
            top = [b + f for b, f in zip(bottom, fractions[k])]
            ax.fill_between(
                generations,
                bottom,
                top,
                color=COLORS[k],
                alpha=0.8,
                linewidth=0,
            )
            bottom = top

        ax.set_ylim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.margins(y=0.02)
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

        # clean spines
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        # legend only once, inside first subplot
        if idx == 0:
            handles = [plt.Line2D([0], [0], color=COLORS[k], lw=4) for k in OUTCOMES]
            ax.legend(
                handles,
                OUTCOMES,
                ncol=4,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.54, 1.6),
                handlelength=1.2,
                columnspacing=0.8,
            )

    axes[-1].set_xlabel("Generation")

    fig.subplots_adjust(
        hspace=0.08,
        top=0.90,
        bottom=0.18,
        left=0.12,
        right=0.98,
    )

    fig.tight_layout()
    return fig


def plot_gen_trajectory(
    means_list: list[list[list[float]]],
    config_labels: list[str],
    model_names: list[str],
    stds_list: list[list[list[float]]] = None,
    n_runs: int = 1,
    cmap: str = "Set1",
    markers: list[str] = None,
    base_height: float = 1.5,
    label="Execution Success Rate",
):
    num_models = len(model_names)
    num_cols = 2
    num_rows = (num_models + num_cols - 1) // num_cols

    num_configs = len(config_labels)

    colors = plt.get_cmap(cmap).colors[:num_configs]
    if markers is None:
        markers = ["o", "s", "^", "D", "*"][:num_configs]

    if stds_list is not None:
        t_crit = t.ppf(0.975, df=n_runs - 1)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.3, base_height * num_rows + 0.5),
        sharex=True,
        sharey=True,
    )

    if num_rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for m_idx in range(num_models):
        ax = axes_flat[m_idx]
        means = np.asarray(means_list[m_idx])

        if stds_list is not None:
            stds = np.asarray(stds_list[m_idx])
            ci_half = t_crit * stds / np.sqrt(n_runs)

        for c_idx in range(num_configs):
            gens = np.arange(len(means[c_idx]))
            ax.plot(
                gens,
                means[c_idx],
                color=colors[c_idx],
                linewidth=1.0,
                marker=markers[c_idx],
                markersize=2.5,
                markevery=2,
            )

            if stds_list is not None:
                ax.fill_between(
                    gens,
                    np.clip(means[c_idx] - ci_half[c_idx], 0, 1),
                    np.clip(means[c_idx] + ci_half[c_idx], 0, 1),
                    color=colors[c_idx],
                    alpha=0.22,
                    linewidth=0,
                )

        ax.set_title(model_names[m_idx], fontsize=6, pad=5)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=6)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for i in range(num_models, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.text(
        0.02,
        0.5,
        label,
        va="center",
        rotation="vertical",
        fontsize=6,
    )
    fig.supxlabel("Generation", y=0.02, fontsize=6)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            lw=1.2,
            marker=markers[i],
            markersize=3,
            markerfacecolor=colors[i],
            markeredgewidth=0.0,
        )
        for i in range(num_configs)
    ]

    fig.legend(
        handles,
        config_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=num_configs,
        frameon=False,
        fontsize=6,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    fig.subplots_adjust(
        left=0.15, right=0.95, top=0.82, bottom=0.15, hspace=0.6, wspace=0.15
    )

    return fig


def plot_pairwise_synergy_heatmap(
    pairwise_data,
    metrics,
    pair_names,
    gens=None,
    cmap=None,
    row_height=1.1,
    width=5,
    cbar_label=r"Pairwise Synergy",
):
    n_rows = len(metrics)
    figsize = (width, row_height * n_rows)

    if gens is None:
        gens = np.arange(
            len(pairwise_data[metrics[0]][list(pairwise_data[metrics[0]].keys())[0]])
        )
    if cmap is None:
        jingliu = ["#1D2652", "#1159AA", "#CEDDF3", "#A2090C", "#4F0407"]
        cmap = LinearSegmentedColormap.from_list("jingliu", jingliu)

    all_values = []
    for m in metrics:
        for k in pair_names:

            key = k.replace(r"$\sigma_{", "sigma_").replace("}$", "")
            all_values.extend(pairwise_data[m][key])
    vmin, vmax = min(all_values), max(all_values)

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=figsize, sharex=True, gridspec_kw={"right": 0.85}
    )

    for i, m in enumerate(metrics):
        ax = axes[i]
        data = np.array(
            [
                pairwise_data[m][k.replace(r"$\sigma_{", "sigma_").replace("}$", "")]
                for k in pair_names
            ]
        )
        sns.heatmap(
            data,
            ax=ax,
            center=0,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=gens,
            yticklabels=pair_names,
        )
        ax.set_ylabel(m)
        if i == len(metrics) - 1:
            ax.set_xlabel("Generation")
        else:
            ax.set_xlabel("")

    cbar_ax = fig.add_axes([0.87, 0.15, 0.015, 0.7])
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    fig.subplots_adjust(left=0.12, right=0.82, top=0.95, bottom=0.12, hspace=0.25)

    return fig
