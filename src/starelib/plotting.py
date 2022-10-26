import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path

from .timeactivitycurve import TimeActivityCurve


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
            for i, activity in enumerate(tac.activity):
                row = {
                    "t": tac.timepoints[i],  # Ignore index values, just take i'th t value
                    "activity": activity,  # The y-axis plotted value, in mCis
                    "k": tac.k if hasattr(tac, 'k') else 0,
                    "label": tac.label if hasattr(tac, 'label') else 0,
                    "source": tac.source,
                    "best_overall": tac.best_overall if hasattr(tac, 'best_overall') else False,
                    "best_in_k": tac.best_in_k if hasattr(tac, 'best_in_k') else False,
                    "name": "n/a" if tac.name is None else tac.name,
                }
                for feature, label in tac.features.items():
                    row[feature] = label
                rows.append(row)

    return pd.DataFrame(rows)


def plot_vascular_tacs(data, vascular_color='blue', highlight_color='red', ax=None):
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

    # Ensure data are properly formatted for our use
    data = prep_data(data)

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(lambda k: f"{k:02d}")

    # Create color palettes that make all hues identical
    num_gray_lines = len(data[data['likely_vascular']]['name'].unique())
    grays = ['gray', ] * num_gray_lines
    num_vasc_lines = len(data[data['best_in_k']]['name'].unique())
    vascs = [vascular_color, ] * num_vasc_lines

    # Plot every vascular centroid as light gray to provide context.
    # These are plotted first to set them as background.
    sns.lineplot(data=data[data['likely_vascular']], x='t', y='activity', hue='name',
                 palette=grays, alpha=0.5, linewidth=1, legend=False, ax=axes)

    # Next, plot the centroids that are the best for their k-means group
    sns.lineplot(data=data[data['best_in_k']], x="t", y="activity", hue='name',
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


def prep_data(data):
    """ Attempt to turn any data into a plottable dataframe.

        :param Any data: The data to identify and manipulate
        :returns: A plottable dataframe
    """

    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], TimeActivityCurve):
            return tacs_to_plottable_dataframe(data)
    else:
        raise TypeError("prep_data can handle DataFrame objects or lists "
                        f"of TimeActivityCurve objects, but not {type(data)}.")


def plot_detailed_tacs(data, title=None, palette=None, color_filter=None):
    """ Plot a time activity curve (TAC), in three panels

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :param str title: The title of the figure
        :param palette: The colors of lines representing the TACs
        :param str color_filter: if set, TACs must have that property
                                  set to True for color lines, otherwise
                                  they'll be plotted gray
        :returns Figure:
    """

    # Create the figure and lay out axes for three panels
    fig = plt.figure(figsize=(11, 11))
    gs = gridspec.GridSpec(nrows=6, ncols=4)

    ax_full = fig.add_subplot(gs[0:2, :])
    ax_early = fig.add_subplot(gs[3:, :2])
    ax_late = fig.add_subplot(gs[3:, 2:])
    axes = [ax_full, ax_early, ax_late, ]

    # Handle different types of data we may receive
    data = prep_data(data)

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(
        lambda k: k if isinstance(k, str) else f"{k:02d}"
    )

    # Create color palettes that make all hues identical
    if palette is None or len(palette) == 0:
        palette = None
        # num_vasc_lines = len(data[data['best_in_k']]['name'].unique())
        # palette = ['black', ] * num_vasc_lines

    for i, ax in enumerate(axes):
        # Determine which time ranges are included in each axes
        if i == 2:
            t_filter = data['t'] >= 5.0
            do_legend = False
        elif i == 1:
            t_filter = data['t'] <= 5.0
            do_legend = False
        else:
            t_filter = [True, ] * len(data)
            do_legend = True
        # Determine which TACs get plotted in color
        if color_filter is None or color_filter not in data.columns:
            c_filter = [True, ] * len(data)
        else:
            c_filter = data[color_filter]

        # Plot every single centroid as light gray to provide context.
        # These are plotted first to set them as background.
        grays = ['gray', ] * len(data[t_filter]['name'].unique())
        sns.lineplot(data=data[t_filter],
                     x="t", y="activity", hue='name',
                     palette=grays, alpha=0.5, linewidth=1, legend=False,
                     ax=ax)

        # Plot circles on the next layer to demonstrate centroid data
        # underlying the model fits
        combined_filter = [
            t and pvc for t, pvc in zip(t_filter, data['name'] == "pvc")
        ]
        sns.scatterplot(data=data[combined_filter],
                        x='t', y='activity', hue='name',
                        palette=palette, alpha=0.5, s=25, legend=False,
                        ax=ax)

        # Next, plot the centroids that are the best for their k-means group
        combined_filter = [t and c for t, c in zip(t_filter, c_filter)]
        sns.lineplot(data=data[combined_filter],
                     x="t", y="activity", hue='name',
                     palette=palette, alpha=0.5, linewidth=3, legend=do_legend,
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

    ax_early.set_ylabel("Activity in mCi/cc")  # ranges -0.05 to +2.00
    ax_late.set_ylabel("Activity in mCi/cc")  # ranges typically -0.05 to +0.30
    ax_late.yaxis.set_label_position("right")
    ax_late.get_yaxis().tick_right()

    ax_full.legend(bbox_to_anchor=(0.50, -0.25), loc="upper center", borderaxespad=0)

    ax_full.set_title("Full scan")
    ax_early.set_title("Early")
    ax_late.set_title("Late")

    fig.suptitle(f"Optimal vascular TACs" if title is None else title)
    # Do NOT use tight_layout; it carves out extra space for the legend
    # fig.tight_layout()

    return fig


def plot_tac_fits(fit_data, param_data, title="", figsize=(8, 5), save_to=None):
    """ Plot the fits contained in the dict. """

    # Plot results for analysis
    fig, axes = plt.subplots(figsize=figsize)
    if "fit_1" in fit_data and "params_1" in param_data:
        sse_1 = np.sum(np.square(fit_data['y'] - fit_data['fit_1']))
        eq_1 = r"$" + f"{param_data['params_1'][0]:0.3f}" + \
               r"e^{-" + f"{param_data['params_1'][1]:0.3f}" + \
               r"}$"
        label_1 = f"P1. {eq_1}  (sse = {sse_1:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_1", label=label_1, ax=axes)
    if "fit_2" in fit_data and "params_2" in param_data:
        sse_2 = np.sum(np.square(fit_data['y'] - fit_data['fit_2']))
        eq_2 = r"$" + f"{param_data['params_2'][0]:0.3f}" + r"e^{-" + \
               f"{param_data['params_2'][1]:0.3f}" + \
               r"} + " + f"{param_data['params_2'][2]:0.3f}" + r"e^{-" + \
               f"{param_data['params_2'][3]:0.3f}" + \
               r"}$"
        label_2 = f"P2. {eq_2}  (sse = {sse_2:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_2", label=label_2, ax=axes)
    if "fit_3" in fit_data and "params_3" in param_data:
        sse_3 = np.sum(np.square(fit_data['y'] - fit_data['fit_3']))
        eq_3 = r"$" + f"{param_data['params_3'][0]:0.3f}" + r"e^{-" + \
               f"{param_data['params_3'][1]:0.3f}" + \
               r"} + " + f"{param_data['params_3'][2]:0.3f}" + r"e^{-" + \
               f"{param_data['params_3'][3]:0.3f}" + \
               r"} + " + f"{param_data['params_3'][4]:0.3f}" + r"e^{-" + \
               f"{param_data['params_3'][5]:0.3f}" + \
               r"}$"
        label_3 = f"P3. {eq_3}  (sse = {sse_3:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_3", label=label_3, ax=axes)
    if "fit_original" in fit_data and "params_original" in param_data:
        sse_o = np.sum(np.square(fit_data['y'] - fit_data['fit_original']))
        eq_o = r"$" + f"{param_data['params_original'][0]:0.3f}" + r"e^{-" + \
               f"{param_data['params_original'][1]:0.3f}" + \
               r"} + " + f"{param_data['params_original'][2]:0.3f}" + r"e^{-" + \
               f"{param_data['params_original'][3]:0.3f}" + \
               r"} + " + f"{param_data['params_original'][4]:0.3f}" + r"e^{-" + \
               f"{param_data['params_original'][5]:0.3f}" + \
               r"}$"
        label_o = f"ML. {eq_o}  (sse = {sse_o:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_original",
                     label=label_o, linestyle=":", ax=axes)
    sns.scatterplot(data=fit_data, x="t", y="y", label="data", color='black', ax=axes)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    axes.set_title(title)
    fig.tight_layout()

    if save_to is not None and Path(save_to).parent.exists():
        fig.savefig(Path(save_to))

    return fig


def plot_before_and_after_tacs(
        before_tacs, after_tacs, mid_times,
        title="", figsize=(8, 5), save_to=None
):
    """ Plot two versions of each TAC.

        This function has never been completed or used, but was left as a stub
        if it needs implementation in the future.

        :param DataFrame before_tacs: TACs before modification, plotted dotted
        :param DataFrame after_tacs: TACs after modification, plotted solid
        :param iterable mid_times: Time indices for rows in TACs
        :param str title: The title of the figure
        :param tuple figsize: The figure's height and width, in inches
        :param Path save_to: If provided, figure will be saved to this path
        :return: matplotlib figure
    """

    # Plot results for analysis
    fig, axes = plt.subplots(figsize=figsize)

    df_mid_times = pd.DataFrame({"t": mid_times})
    df_before = pd.concat([df_mid_times, before_tacs, ], axis=1).melt(
        id_vars="t"
    ).rename(columns={"variable": "region", "value": "activity"})
    df_after = pd.concat([df_mid_times, after_tacs, ], axis=1).melt(
        id_vars="t"
    ).rename(columns={"variable": "region", "value": "activity"})

    sns.lineplot(data=df_before, x="t", y="activity", hue="region", legend=None, linestyle=":")
    sns.lineplot(data=df_after, x="t", y="activity", hue="region")

    axes.set_title(title)
    fig.tight_layout()

    if save_to is not None and Path(save_to).parent.exists():
        fig.savefig(Path(save_to))

    return fig
