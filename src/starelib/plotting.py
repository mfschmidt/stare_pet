import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def tacs_to_plottable_dataframe(tacs):
    """ Format dictionaries into a plottable dataframe

        From centroid tacs data and corresponding timepoints,
        create a dataframe that makes seaborn plotting seamless.

        If any frames are being ignored, each centroid's time points
        and the times time points should both have already removed
        the same frames, so they are synchronized.

        :param list tacs: list of all centroids as dicts

        :returns DataFrame: long-format plottable TACs data
    """

    rows = []
    for tac in tacs:
        if tac is not None:
            run = tac.source
            if hasattr(tac, 'best_overall') and tac.best_overall:
                if tac.k > 4:
                    run = "step 1"
                else:
                    run = "step 2"
            for i, activity in enumerate(tac.activity):
                row = {
                    "t": tac.timepoints[i],  # Ignore index values, just take i'th t value
                    "activity": activity,  # The y-axis plotted value, in mCis
                    "k": tac.k if hasattr(tac, 'k') else 0,
                    "label": tac.label if hasattr(tac, 'label') else 0,
                    "source": tac.source,
                    "best_overall": tac.best_overall if hasattr(tac, 'best_overall') else False,
                    "best_in_k": tac.best_in_k if hasattr(tac, 'best_in_k') else False,
                    "vascular": tac.vascular if hasattr(tac, 'vascular') else False,
                    "run": run
                }
                rows.append(row)

    return pd.DataFrame(rows)


def plot_simple_tacs(data, vascular_color='black', highlight_color='red', ax=None):
    """ Plot a time activity curve (TAC), in one panel

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
    if 'run' not in data.columns:
        data.loc[:, 'run'] = data.apply(
            lambda row: f"k-{row['k']:02d}_label-{row['label']}", axis=1
        )

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


def plot_detailed_tacs(data, title=None, palette=None):
    """ Plot a time activity curve (TAC), in three panels

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :param str title: The title of the figure
        :param palette: The colors of lines representing the TACs
        :returns Figure:
    """

    # Create the figure and lay out axes for three panels
    fig = plt.figure(figsize=(11, 11))
    gs = gridspec.GridSpec(nrows=2, ncols=5)

    ax_full = fig.add_subplot(gs[0, :])
    ax_early = fig.add_subplot(gs[1, 0:2])
    ax_late = fig.add_subplot(gs[1, 3:5])
    axes = [ax_full, ax_early, ax_late, ]

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(
        lambda k: k if isinstance(k, str) else f"{k:02d}"
    )

    # Create a unique id for each k/label combination,
    # allows seaborn to plot lines individually rather than estimate their mean/ci
    if 'run' not in data.columns:
        data.loc[:, 'run'] = data.apply(
            lambda row: f"k-{row['k']:02d}_label-{row['label']}", axis=1
        )

    # Create color palettes that make all hues identical
    num_gray_lines = len(data[data['vascular']]['run'].unique())
    grays = ['gray', ] * num_gray_lines
    if palette is None:
        num_vasc_lines = len(data[data['best_in_k']]['run'].unique())
        palette = ['black', ] * num_vasc_lines

    for i, ax in enumerate(axes):
        # Determine which time ranges are included in each axes
        if i == 2:
            t_filter = data['t'] >= 4.0
            do_legend = False
        elif i == 1:
            t_filter = data['t'] < 4.0
            do_legend = False
        else:
            t_filter = [True, ] * len(data)
            do_legend = True
        # Plot every single centroid as light gray to provide context.
        # These are plotted first to set them as background.
        # sns.lineplot(data=data[t_filter],
        #              x='t', y='activity', hue='run',
        #              palette=grays, alpha=0.5, linewidth=1, legend=False,
        #              ax=ax)

        # Next, plot the centroids that are the best for their k-means group
        sns.lineplot(data=data[t_filter],
                     x="t", y="activity", hue='run',
                     palette=palette, alpha=0.5, linewidth=1, legend=do_legend,
                     ax=ax)

        # Finally, plot the very best centroid of the whole batch.
        # best_k = data[data['best_overall'] & t_filter]['k'].unique()[0]
        # best_label = data[data['best_overall'] & t_filter]['label'].unique()[0]
        # sns.lineplot(data=data[data['best_overall'] & t_filter],
        #              x="t", y="activity",
        #              color=highlight_color, linestyle=":", linewidth=6, alpha=0.5,
        #              label=f"Best (k-{best_k:02d}-{best_label:02d})", ax=ax)

        # Finish off the details so the plot is readable.
        ax.set_xlabel("Minutes")  # ranges 0 to 60
        ax.set_ylabel("Activity in mCi/cc")  # ranges -0.05 to +0.30

    ax_full.legend(bbox_to_anchor=(0.50, -0.25), loc="upper center", borderaxespad=0)

    ax_full.set_title("Full scan")
    ax_early.set_title("Early")
    ax_late.set_title("Late")

    fig.suptitle(f"Optimal vascular TACs" if title is None else title)
    # Do NOT use tight_layout; it carves out extra space for the legend
    # fig.tight_layout()

    return fig
