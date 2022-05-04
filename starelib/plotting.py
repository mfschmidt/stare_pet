import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def centroids_to_plottable_tacs(all_centroids, times):
    """ Format dictionaries into a plottable dataframe

        From centroid tacs data and corresponding timepoints,
        create a dataframe that makes seaborn plotting seamless.

        :param list all_centroids: list of all centroids as dicts
        :param DataFrame times: time points corresponding to each volume

        :returns DataFrame: long-format plottable TACs data
    """

    tacs = []
    for centroid in all_centroids:
        for i, activity in enumerate(centroid["centroid"]):
            tacs.append({
                "t": times.loc[i, 't'],
                "k": centroid["k"],
                "label": centroid["label"],
                "best_overall": centroid["best_overall"],
                "best_in_k": centroid["best_in_k"],
                "activity": activity,
            })

    return pd.DataFrame(tacs)


def plot_tacs(data):
    """ Plot a time activity curve (TAC)

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :returns Figure:
    """

    fig, axes = plt.subplots(figsize=(10, 6))

    # Force seaborn to treat K as categorical rather than continuous
    data['K'] = data['k'].astype(str)

    sns.lineplot(data=data[data['best_overall']], x="t", y="activity",
                 color="gray", linestyle=":", linewidth=5, alpha=0.5, ax=axes)
    sns.lineplot(data=data[data['best_in_k']], x="t", y="activity",
                 hue="K", ax=axes)
    axes.set_xlabel("Minutes")  # ranges 0 to 60
    axes.set_ylabel("Activity/cc")  # ranges -0.05 to +0.30
    axes.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    fig.suptitle(f"Optimal vascular TACs: {data['k'].min()}-{data['k'].max()}"
                 " k-means clusters")
    fig.tight_layout()

    return fig
