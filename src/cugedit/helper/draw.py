import matplotlib.pyplot as plt
import numpy as np


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
    stds_list: list[list[list[float]]],
    config_labels: list[str],
    model_names: list[str],
    cmap: str = "Set1",
    markers: list[str] = None,
):
    assert len(means_list) == 4
    assert len(stds_list) == 4
    assert len(model_names) == 4

    num_configs = len(config_labels)
    num_gens = len(means_list[0][0])
    gens = np.arange(num_gens)

    colors = plt.get_cmap(cmap).colors[:num_configs]
    if markers is None:
        markers = ["o", "s", "^", "D", "*"][:num_configs]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True, sharey=True)

    for m_idx, ax in enumerate(axes):
        means = np.asarray(means_list[m_idx])
        stds = np.asarray(stds_list[m_idx])

        for c_idx in range(num_configs):
            ax.plot(
                gens,
                means[c_idx],
                color=colors[c_idx],
                linewidth=1.3,
                marker=markers[c_idx],
                markersize=3.5,
                markevery=2,
            )
            ax.fill_between(
                gens,
                np.clip(means[c_idx] - stds[c_idx], 0, 1),
                np.clip(means[c_idx] + stds[c_idx], 0, 1),
                color=colors[c_idx],
                alpha=0.22,
                linewidth=0,
            )

        ax.set_title(model_names[m_idx], fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=7)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        if m_idx == 0:
            ax.set_ylabel("Execution Success Rate")
        else:
            ax.set_ylabel("")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            lw=1.5,
            marker=markers[i],
            markersize=4,
            markerfacecolor=colors[i],
            markeredgewidth=0.0,
        )
        for i in range(num_configs)
    ]
    fig.supxlabel("Generation", y=0.01)
    fig.legend(
        handles,
        config_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=num_configs,
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.88,
        bottom=0.18,
        wspace=0.15,
    )
    return fig
