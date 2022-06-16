import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def centroids_to_plottable_tacs(all_centroids, times):
    """ Format dictionaries into a plottable dataframe

        From centroid tacs data and corresponding timepoints,
        create a dataframe that makes seaborn plotting seamless.

        If any frames are being ignored, each centroid's time points
        and the times time points should both have already removed
        the same frames, so they are synchronized.

        :param list all_centroids: list of all centroids as dicts
        :param DataFrame times: time points corresponding to each volume

        :returns DataFrame: long-format plottable TACs data
    """

    tacs = []
    for centroid in all_centroids:
        for i, activity in enumerate(centroid["centroid"]):
            tacs.append({
                "t": times.iloc[i]['t'],  # Ignore index values, just take i'th t value
                "k": centroid["k"],
                "label": centroid["label"],
                "best_overall": centroid["best_overall"],
                "best_in_k": centroid["best_in_k"],
                "vascular": centroid["vascular"],
                "activity": activity,
            })

    return pd.DataFrame(tacs)


def plot_tacs(data, vascular_color='black', highlight_color='red', ax=None):
    """ Plot a time activity curve (TAC)

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :param vascular_color: The color of lines representing the best vascular TACs for each value of k
        :param highlight_color: The color of highlight laid over the best of all vascular TACs
        :param ax: Optionally draw on your own axes
        :returns Figure:
    """

    if ax is None:
        fig, axes = plt.subplots(figsize=(10, 6))
    else:
        fig, axes = ax.get_figure(), ax

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(lambda k: f"{k:02d}")

    # Create a unique id for each k/label combination,
    # allows seaborn to plot lines individually rather than estimate their mean/ci
    data.loc[:, 'run'] = data.apply(lambda row: f"k-{row['k']:02d}_label-{row['label']}", axis=1)

    # Create color palettes that make all hues identical
    num_gray_lines = len(data[data['vascular']]['run'].unique())
    grays = ['gray', ] * num_gray_lines
    num_vasc_lines = len(data[data['best_in_k']]['run'].unique())
    vascs = [vascular_color, ] * num_vasc_lines

    # Plot every single centroid as light gray to provide context.
    # These are plotted first to set them as background.
    sns.lineplot(data=data[data['vascular']], x='t', y='activity', hue='run',
                 palette=grays, alpha=0.5, linewidth=1, legend=False, ax=axes)

    # Next, plot the centroids that are the best for their k-means group
    sns.lineplot(data=data[data['best_in_k']], x="t", y="activity", hue='run',
                 palette=vascs, alpha=0.5, linewidth=1, label="", ax=axes)

    # Finally, plot the very best centroid of the whole batch.
    best_k = data[data['best_overall']]['k'].unique()[0]
    best_label = data[data['best_overall']]['label'].unique()[0]
    sns.lineplot(data=data[data['best_overall']], x="t", y="activity",
                 color=highlight_color, linestyle=":", linewidth=6, alpha=0.5,
                 label=f"Best (k-{best_k:02d}-{best_label:02d})", ax=axes)

    # Finish off the details so the plot is readable.
    axes.set_xlabel("Minutes")  # ranges 0 to 60
    axes.set_ylabel("Activity/cc")  # ranges -0.05 to +0.30
    axes.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    fig.suptitle(f"Optimal vascular TACs: {data['k'].min()}-{data['k'].max()}"
                 " k-means clusters")
    fig.tight_layout()

    return fig
