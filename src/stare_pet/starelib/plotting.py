import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from pathlib import Path
import nibabel as nib
from nilearn import image
from nilearn.plotting import plot_roi
from math import ceil

from .timeactivitycurve import TimeActivityCurve
from .util import get_kde_fwhm_points
from .colors import bat_palette, freesurfer_palette
from .centroid_heuristics import (
    likely_noise, likely_irreversible, likely_vascular, likely_peripheral
)


# One global random number generator should be sufficient
rng = np.random.default_rng()

# Make a color palette with FS colors, and matching Betsy's in matplotlib
stare_palette = freesurfer_palette.copy()
stare_palette.update(bat_palette)


def palette_from_tac_regions(data):
    """ Generate a palette to cover all regions in data.
    """

    palette = dict()
    # Every region in the data must have a color in the palette, so
    # check to ensure we can draw something rather than error out.
    default_color = 'gray'
    for region in data.columns:
        color = None
        if region in stare_palette.keys():
            # Great! use the same color for the region's fit line
            color = stare_palette[region]
        else:
            alternatives = [
                f"Left-{region}", f"Right-{region}",
                f"{region[:3]}-lh-{region[4:]}", f"{region[:3]}-rh-{region[4:]}"
            ]
            for alt in alternatives:
                if alt in stare_palette.keys():
                    color = stare_palette[alt]
                    break
        if color is None:  # still, after all that
            color = default_color
        # Add this region into the palette with whatever color we found for it.
        palette[region] = color
        palette[f"fit_{region}"] = color

    palette.update(stare_palette)

    return palette


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
            tac_dict = tac.to_dict()
            for i, activity in enumerate(tac.activity):
                row = {
                    "t": tac.timepoints[i],  # Ignore idx values, take i'th t
                    "activity": activity,  # The y-axis plotted value, in mCis
                    "k": tac_dict.get('k', 0),
                    "label": tac_dict.get('label', 0),
                    "source": tac_dict.get('source', "n/a"),
                    "best_overall": tac_dict.get('best_overall', False),
                    "best_in_k": tac_dict.get('best_in_k', False),
                    "name": tac_dict.get('name', "n/a"),
                }
                for k, v in tac_dict.items():
                    if k.startswith('feature_'):
                        row[k[8:]] = v
                # for feature, label in tac.features.items():
                #     row[feature] = label

                rows.append(row)

    return pd.DataFrame(rows)


def plot_vascular_tacs(
        data, vascular_color='blue', highlight_color='red',
        tall=False, large=False, draw_non_vascular=False, ax=None
):
    """ Plot a time activity curve (TAC), in one panel

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :param vascular_color: The color of lines representing the best
                               vascular TACs for each value of k
        :param highlight_color: The color of highlight laid over the best of
                                all vascular TACs
        :param tall: Shrink the legend and move it to the bottom
        :param large: Give a bit more resolution and size for 16x9 aspect
        :param draw_non_vascular: Draw even TACs that aren't considered
        :param ax: Optionally draw on your own axes
        :returns Figure:
    """

    if tall:
        figsize = (6, 6)
    elif large:
        figsize = (16, 9)
    else:
        figsize = (10, 6)
    if ax is None:
        fig, axes = plt.subplots(figsize=figsize, layout='tight')
    else:
        fig, axes = ax.get_figure(), ax

    # Ensure data are properly formatted for our use
    data = prep_data(data)

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(lambda k: f"{k:02d}")

    # Create color palettes that make all hues identical
    num_gray_lines = len(data[data['likely_vascular'].astype(bool)]['name'].unique())
    grays = ['gray', ] * num_gray_lines
    num_vasc_lines = len(data[data['best_in_k'].astype(bool)]['name'].unique())
    vascs = [vascular_color, ] * num_vasc_lines

    # Plot every non-vascular centroid as light gray to provide context.
    # These are plotted first to set them as background.
    if draw_non_vascular and len(data[~data['likely_vascular'].astype(bool)]) > 0:
        sns.lineplot(
            data=data[~data['likely_vascular'].astype(bool)], x='t', y='activity',
            color='gray', hue='name', alpha=0.5, linewidth=1,
            linestyle=":", legend=False, ax=axes
        )

    # Plot every vascular centroid as light gray to provide context.
    # These are plotted first to set them as background.
    if len(data[data['likely_vascular'].astype(bool)]) > 0:
        sns.lineplot(
            data=data[data['likely_vascular'].astype(bool)], x='t', y='activity',
            hue='name', palette=grays, alpha=0.5, linewidth=1,
            legend=False, ax=axes
        )

    # Next, plot the centroids that are the best for their k-means group
    # print(f"DEBUG: plotting {len(data[data['best_in_k']])} TACs")
    if len(data[data['best_in_k'].astype(bool)]) > 1:
        # Each line will be labelled automatically by its hue/palette
        sns.lineplot(
            data=data[data['best_in_k'].astype(bool)], x="t", y="activity",
            hue='name', palette=vascs, alpha=0.5, linewidth=1, ax=axes
        )
    elif len(data[data['best_in_k'].astype(bool)]) > 0:
        # This single line will not be labelled and needs help
        sns.lineplot(
            data=data[data['best_in_k'].astype(bool)], x="t", y="activity",
            hue='name', palette=vascs, alpha=0.5, linewidth=1,
            label=f"Best of k", ax=axes
        )
    if tall:
        # clear all legend labels, allowing the next lineplot to make a new one.
        for line in axes.lines:
            line.set_label(s='')

    # Finally, plot the very best centroid of the whole batch.
    if len(data[data['best_overall'].astype(bool)]) > 0:
        best_k = data[data['best_overall'].astype(bool)]['k'].unique()[0]
        best_label = data[data['best_overall'].astype(bool)]['label'].unique()[0]
        sns.lineplot(
            data=data[data['best_overall'].astype(bool)], x="t", y="activity",
            color=highlight_color, linestyle=":", linewidth=6, alpha=0.5,
            label=f"Best overall (k-{best_k:02d}-{best_label:02d})", ax=axes
        )
    else:
        # If no centroids are labeled best, the legend will be empty; fill it.
        sns.lineplot(x=(0.0, 0.0, ), y=(0.0, 0.0, ), alpha=0.0,
                     label="No best centroid", ax=axes)

    # Finish off the details so the plot is readable.
    axes.set_xlabel("Minutes")  # ranges 0 to 60
    axes.set_ylabel("Activity/cc")  # ranges -0.05 to +0.30
    if tall:
        axes.legend(
            bbox_to_anchor=(0.5, -0.15), loc="upper center", borderaxespad=0
        )
    else:
        axes.legend(
            bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0
        )
    if data['k'].min() == data['k'].max():
        fig.suptitle(f"Optimal vascular TACs: "
                     f"k={data['k'].min()}"
                     " k-means clusters")
    else:
        fig.suptitle(f"Optimal vascular TACs: "
                     f"k from {data['k'].min()}-{data['k'].max()}"
                     " k-means clusters")

    return fig


def prep_data(data):
    """ Attempt to turn any data into a plottable dataframe.

        :param Any data: The data to identify and manipulate
        :returns: A plottable dataframe
    """

    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, list) and (len(data) > 0):
        if isinstance(data[0], TimeActivityCurve):
            return tacs_to_plottable_dataframe(data)
    else:
        raise TypeError("prep_data can handle DataFrame objects or lists "
                        f"of TimeActivityCurve objects, but not {type(data)}.")


def plot_top_centroids_atlas(
        step_1_mask_img, step_2_mask_img, pet4d_img, color_map = None,
        title="", figsize=(8, 4)
):
    """ Plot the PET average background and masks over the top.

    """

    if (
            (step_1_mask_img is None and step_2_mask_img is None) or
            pet4d_img is None
    ):
        print(f"ERROR: Trying to 'plot_top_centroids_atlas' without images!")
        print(f"     : step_1_mask_img is '{str(step_1_mask_img)}")
        print(f"     : step_2_mask_img is '{str(step_2_mask_img)}")
        print(f"     : pet4d_img is '{str(pet4d_img)}")
        return plt.figure()

    # Extract image data, or create empty data to fill-in
    mean_pet_img = image.mean_img(pet4d_img, copy_header=True)
    if step_1_mask_img is None:
        step_1_data = np.zeros(mean_pet_img.shape)
    elif len(step_1_mask_img.shape) > 3:
        step_1_data = image.mean_img(step_1_mask_img, copy_header=True).get_fdata()
    else:
        step_1_data = step_1_mask_img.get_fdata()
    if step_2_mask_img is None:
        step_2_data = np.zeros(mean_pet_img.shape)
    elif len(step_2_mask_img.shape) > 3:
        step_2_data = image.mean_img(step_2_mask_img, copy_header=True).get_fdata()
    else:
        step_2_data = step_2_mask_img.get_fdata()

    # Build the plottable image with both masks
    atlas_combo_img = nib.Nifti1Image(
        step_1_data + step_2_data,
        affine=mean_pet_img.affine,
        dtype=np.uint8,
    )
    if color_map is None:
        color_map = ListedColormap(['orange', 'red', ])

    fig = plt.figure(figsize=figsize)
    axes = fig.add_axes((0, 0, 1, 1, ))
    display = plot_roi(
        roi_img=atlas_combo_img, bg_img=mean_pet_img,
        cmap=color_map, black_bg=False, axes=axes,
    )
    display.title(title, color='black', bgcolor='white')

    return fig


def plot_detailed_tacs(data, title=None, palette=None, dashes=None,
                       color_filter=None, figsize=(11, 8)):
    """ Plot a time activity curve (TAC), in three panels

        Given a long-format dataframe with TACs and their metadata,
        return a figure with their line plots.

        :param DataFrame data: The TACs to plot
        :param str title: The title of the figure
        :param palette: The colors of lines representing the TACs
        :param dashes: The style of lines representing the TACs
        :param str color_filter: if set, TACs must have that property
                                  set to True for color lines, otherwise
                                  they'll be plotted gray
        :param figsize: Override the size of the figure with a (w, h) two-tuple

        :returns Figure:
    """

    # Create the figure and lay out axes for three panels
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=4, ncols=4)

    ax_full = fig.add_subplot(gs[0:2, :])
    ax_early = fig.add_subplot(gs[2:, :2])
    ax_late = fig.add_subplot(gs[2:, 2:])
    axes = [ax_full, ax_early, ax_late, ]

    # Handle different types of data we may receive
    data = prep_data(data)
    if data['t'].max() < 250.0:
        early_late_threshold = 5.0
        x_label = "Time (minutes)"
    else:
        early_late_threshold = 300.0
        x_label = "Time (seconds)"

    # Force seaborn to treat K as categorical rather than continuous
    data.loc[:, 'K'] = data['k'].apply(
        lambda k: k if isinstance(k, str) else f"{k:02d}"
    )

    # Create color palettes that make all hues identical
    if palette is None or len(palette) == 0:
        palette = None
        # num_vasc_lines = len(data[data['best_in_k']]['name'].unique())
        # palette = ['black', ] * num_vasc_lines
    if dashes is None or len(dashes) == 0:
        dashes = None

    for i, ax in enumerate(axes):
        # Determine which time ranges are included in each axes
        if i == 2:
            t_filter = data['t'] >= early_late_threshold
            do_legend = False
        elif i == 1:
            t_filter = data['t'] <= early_late_threshold
            do_legend = False
        else:
            t_filter = [True, ] * len(data)
            do_legend = True
        # Determine which TACs get plotted in color
        if color_filter is None or str(color_filter) not in data.columns:
            c_filter = [True, ] * len(data)
        else:
            c_filter = data[color_filter]

        # Plot every single centroid as light gray to provide context.
        # These are plotted first to set them as background.
        grays = ['gray', ] * len(data[t_filter]['name'].unique())
        sns.lineplot(data=data[t_filter],
                     x="t", y="activity", hue='name',
                     style='name' if dashes is not None else None,
                     dashes=dashes, palette=grays, alpha=0.5, linewidth=1,
                     legend=False, estimator=None, ax=ax)

        # Plot circles on the next layer to demonstrate centroid data
        # underlying the model fits
        combined_filter = [
            t and pvc for t, pvc in zip(t_filter, data['name'] == "pvc")
        ]
        if np.sum(combined_filter) > 0:
            sns.scatterplot(
                data=data[combined_filter], x='t', y='activity',
                hue='name', palette=palette, alpha=0.5, s=25,
                legend=False, ax=ax
            )

        # Next, plot the centroids that are the best for their k-means group
        combined_filter = [t and c for t, c in zip(t_filter, c_filter)]
        if np.sum(combined_filter) > 0:
            sns.lineplot(
                data=data[combined_filter], x="t", y="activity",
                hue='name', palette=palette,
                style='name' if dashes is not None else None,
                dashes=dashes, alpha=0.5, linewidth=3,
                legend=do_legend, estimator=None, ax=ax
            )

        # Finally, plot the very best centroid of the whole batch.
        # best_k = data[data['best_overall'] & t_filter]['k'].unique()[0]
        # best_label = (data[data['best_overall'] &
        #               t_filter]['label'].unique()[0])
        # sns.lineplot(data=data[data['best_overall'] & t_filter],
        #              x="t", y="activity",
        #              color=highlight_color, linestyle=":",
        #              linewidth=6, alpha=0.5,
        #              label=f"Best (k-{best_k:02d}-{best_label:02d})", ax=ax)

        # Finish off the details so the plot is readable.
        ax.set_xlabel(x_label)  # ranges 0 to 60

    ax_early.set_ylabel("Activity in mCi/cc")  # ranges -0.05 to +2.00
    ax_late.set_ylabel("Activity in mCi/cc")  # ranges typically -0.05 to +0.30
    ax_late.yaxis.set_label_position("right")
    ax_late.get_yaxis().tick_right()

    # Ensure the TACs are labeled in a reasonable order.
    def label_seq_value(label):
        value = 0
        if "step 1" in label:
            value = 20
        elif "step 2" in label:
            value = 10
        if "pvc" in label:
            value = 0
        if "reduced" in label:
            value = value - 1
        return value

    handles, labels = ax_full.get_legend_handles_labels()
    re_mapper = list()
    for i, lbl in enumerate(labels):
        re_mapper.append((i, label_seq_value(lbl), lbl, handles[i]))
    new_handles, new_labels = list(), list()
    for new_idx, (old_idx, rank, label, handle) in enumerate(
            sorted(re_mapper, key=lambda x: x[1])
    ):
        new_handles.append(handle)
        new_labels.append(label)
    # ax_full.legend(new_handles, new_labels, bbox_to_anchor=(0.50, -0.25),
    #                loc="upper center", borderaxespad=0)
    ax_full.legend(new_handles, new_labels, bbox_to_anchor=(1.0, 1.0),
                   loc="upper right", borderaxespad=1)

    ax_full.set_title("Full scan")
    ax_early.set_title("Early")
    ax_late.set_title("Late")

    fig.suptitle(f"Optimal vascular TACs" if title is None else title)
    # Do NOT use tight_layout; it carves out extra space for the legend
    fig.tight_layout()

    return fig


def plot_tac_fits(
        fit_data, param_data, title="", figsize=(8, 5), save_to=None
):
    """ Plot the fits contained in the fit_data dict.

        This function is only used in a debug script, not STARE.
    """

    # Plot results for analysis
    fig, axes = plt.subplots(figsize=figsize)
    if "fit_1" in fit_data and "params_1" in param_data:
        sse_1 = np.sum(np.square(fit_data['y'] - fit_data['fit_1']))
        eq_1 = (r"$" + f"{param_data['params_1'][0]:0.3f}"
                r"e^{-" + f"{param_data['params_1'][1]:0.3f}"
                r"}$")
        label_1 = f"P1. {eq_1}  (sse = {sse_1:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_1", label=label_1, ax=axes)
    if "fit_2" in fit_data and "params_2" in param_data:
        sse_2 = np.sum(np.square(fit_data['y'] - fit_data['fit_2']))
        eq_2 = (r"$" + f"{param_data['params_2'][0]:0.3f}" + r"e^{-"
                f"{param_data['params_2'][1]:0.3f}"
                r"} + " + f"{param_data['params_2'][2]:0.3f}" + r"e^{-"
                f"{param_data['params_2'][3]:0.3f}"
                r"}$")
        label_2 = f"P2. {eq_2}  (sse = {sse_2:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_2", label=label_2, ax=axes)
    if "fit_3" in fit_data and "params_3" in param_data:
        sse_3 = np.sum(np.square(fit_data['y'] - fit_data['fit_3']))
        eq_3 = (r"$" + f"{param_data['params_3'][0]:0.3f}" + r"e^{-"
                f"{param_data['params_3'][1]:0.3f}"
                r"} + " + f"{param_data['params_3'][2]:0.3f}" + r"e^{-"
                f"{param_data['params_3'][3]:0.3f}"
                r"} + " + f"{param_data['params_3'][4]:0.3f}" + r"e^{-"
                f"{param_data['params_3'][5]:0.3f}"
                r"}$")
        label_3 = f"P3. {eq_3}  (sse = {sse_3:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_3", label=label_3, ax=axes)
    if "fit_original" in fit_data and "params_original" in param_data:
        sse_o = np.sum(np.square(fit_data['y'] - fit_data['fit_original']))
        eq_o = (r"$" + f"{param_data['params_original'][0]:0.3f}" + r"e^{-"
                f"{param_data['params_original'][1]:0.3f}"
                r"} + " + f"{param_data['params_original'][2]:0.3f}" + r"e^{-"
                f"{param_data['params_original'][3]:0.3f}"
                r"} + " + f"{param_data['params_original'][4]:0.3f}" + r"e^{-"
                f"{param_data['params_original'][5]:0.3f}"
                r"}$")
        label_o = f"ML. {eq_o}  (sse = {sse_o:0.3f})"
        sns.lineplot(data=fit_data, x="t", y="fit_original",
                     label=label_o, linestyle=":", ax=axes)
    sns.scatterplot(data=fit_data, x="t", y="y",
                    label="data", color='black', ax=axes)
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

    sns.lineplot(data=df_before, x="t", y="activity",
                 hue="region", legend=None, linestyle=":")
    sns.lineplot(data=df_after, x="t", y="activity",
                 hue="region")

    axes.set_title(title)
    fig.tight_layout()

    if save_to is not None and Path(save_to).parent.exists():
        fig.savefig(Path(save_to))

    return fig


def winsorize_curves(curves, sds=2):
    """ Remove outliers and return subset of curves. """

    # curves is a 'length of a curve (551)' x 'number of curves (500)' dataframe
    curve_means = np.mean(np.array(curves), axis=1)
    curve_stds = np.std(np.array(curves), axis=1)
    curve_mins = curve_means - (sds * curve_stds)
    curve_maxs = curve_means + (sds * curve_stds)
    pruned_curves = [
        curves[i] for i in range(curves.shape[1])
        if np.sum(
            [((_ < curve_mins[j]) or (_ > curve_maxs[j]))
             for j, _ in enumerate(curves[i])]
        ) == 0
    ]
    return pd.DataFrame(pruned_curves).T


def plot_bootstrap_curves(
        curves, time_tac, vasc_tac, subject, skip_outliers=False
):
    """ """

    # Assign the mean activity to the TAC containing the common timepoints
    mean_tac = TimeActivityCurve(
        activity=np.mean(np.array(curves), axis=0),
        timepoints=time_tac.timepoints,
        source="bootstrapping",
        name=f"mean of {len(curves)} curves",
    )

    curve_means = np.mean(np.array(curves), axis=0)
    curve_stds = np.std(np.array(curves), axis=0)
    if skip_outliers:
        curve_mins = curve_means - 2 * curve_stds
        curve_maxs = curve_means + 2 * curve_stds
    else:
        curve_mins = np.min(np.array(curves), axis=0)
        curve_maxs = np.max(np.array(curves), axis=0)

    # Add curves to a dataframe of TACs, but only if they lie within thresholds
    df = tacs_to_plottable_dataframe(
        [
            TimeActivityCurve(
                activity=curves[i],
                timepoints=time_tac.timepoints,
                source=f"bootstrap {i}",
                name=f"bootstrap curve {i}",
            ) for i in range(len(curves))
            if np.sum([
                ((_ < curve_mins[j]) or (_ > curve_maxs[j]))
                for j, _ in enumerate(curves[i])
            ]) == 0
        ] + [
            mean_tac, vasc_tac
        ])
    # Allow all curves to be bland gray, then set off the mean curve as special
    df['color_filter'] = False
    df.loc[df['name'] == mean_tac.name, 'color_filter'] = True
    df.loc[df['name'] == vasc_tac.name, 'color_filter'] = True

    fig = plot_detailed_tacs(df, color_filter='color_filter', )
    fig.suptitle(f"{len(curves)} bootstrapped curves for {subject}")

    return fig


def draw_bounds_and_peak(
        values, color, ax, label_prefix="", num_bootstraps=1000,
        verbose=False
):
    fwhm, kde_x, kde_y = get_kde_fwhm_points(
        values, num_bootstraps=num_bootstraps, stat='probability'
    )
    xl, xc, xr = fwhm[:, 0]  # along the x-axis: left, center, right
    yb = 0.0
    yc, yt, _ = fwhm[:, 1]  # along the y-axis: bottom, center, top

    if verbose:
        print(f"<in draw_bounds_and_peak> {label_prefix}: "
              f"({xl:0.3f}, {yc:0.1%}), "
              f"({xc:0.3f}, {yt:0.1%}), "
              f"({xr:0.3f}, {yc:0.1%})")

    ax.plot(kde_x, kde_y, ls=":", c=color)

    ax.add_artist(lines.Line2D(
        xdata=[xl, xl, xr, xr, ],
        ydata=[yb, yc, yc, yb, ],
        color=color,
        label=" ".join(
            [label_prefix, f"HM {yc:0.1%} @ ({xl:0.3f} to {xr:0.3f})"]
        ),
    ))
    ax.add_artist(lines.Line2D(
        xdata=[xc, xc, ],
        ydata=[yc, yt, ],
        color=color,
        label=" ".join(
            [label_prefix, f"Peak {yt:0.1%} @ {xc:0.3f}"]
        ),
    ))
    # This mean value is pretty meaningless.
    # TODO: It would be better to plot the matching parameter from fitting the PVC-corrected centroid.
    ax.text(np.mean(values), 0.0, "*", ha="center", va="bottom", color=color)


def plot_density(
        rate_constants, ax, num_bootstraps=1000,
        hist_color='gray', fwhm_color='black',
        label_prefix='', n_text_y=0.10, verbose=False
):
    """ Plot densities """

    indices = rng.choice(len(rate_constants), num_bootstraps)
    plottable_rcs = np.take(rate_constants, indices, axis=0)
    sns.histplot(
        plottable_rcs, bins=100, stat='probability', kde=False,
        color=hist_color, alpha=0.50,
        ax=ax
    )
    n_string = f"{label_prefix} n = {len(rate_constants)}"
    ax.text(0.98, n_text_y, n_string,
            ha='right', va='bottom', transform=ax.transAxes, )

    draw_bounds_and_peak(
        rate_constants, fwhm_color, ax, label_prefix=label_prefix,
        num_bootstraps=num_bootstraps, verbose=verbose,
    )

    ax.legend()


def plot_regional_densities(
        region_names, rate_constants, comp_rate_constants=None,
        hist_color='gray', fwhm_color='black',
        comp_hist_color='gray', comp_fwhm_color='black',
        num_bootstraps=1000, coefficient="K", subject="Unknown", tracer='FDG',
        verbose=False
):
    """ Plot all regions in a figure.

        Plot rate_constants with specified colors. Optionally, also plot
        comp_rate_constants if provided.
    """

    if comp_rate_constants is None:
        # There are no comparisons, so just plot our one set of constants.
        main_prefix = "stare"
        comp_prefix = "na"  # doesn't matter, won't get plotted
    else:
        # A comparison is taking place, so label appropriately
        # at least the only appropriate way this will be used in the near future
        main_prefix = "python"
        comp_prefix = "matlab"

    # Figure out layout and figure size
    n_cols = int(np.ceil(np.sqrt(len(region_names))))
    n_rows = int(np.ceil(len(region_names) / n_cols))
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, sharex='all', sharey='all',
        figsize=(n_cols * 5, n_rows * 5 + 1)
    )

    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            if len(region_names) > i:
                if comp_rate_constants is not None:
                    plot_density(
                        comp_rate_constants[:, i], ax=ax,
                        num_bootstraps=num_bootstraps,
                        hist_color=comp_hist_color, fwhm_color=comp_fwhm_color,
                        n_text_y=0.15, label_prefix=comp_prefix,
                        verbose=verbose
                    )
                plot_density(
                    rate_constants[:, i], ax=ax,
                    num_bootstraps=num_bootstraps,
                    hist_color=hist_color, fwhm_color=fwhm_color,
                    n_text_y=0.10, label_prefix=main_prefix,
                    verbose=verbose
                )
                ax.set_title(region_names[i])
                ax.set_xlabel("micro-parameter estimate")
                ax.set_ylabel("bootstrap iterations")
            else:
                ax.remove()
            i += 1

    fig.suptitle(f"Regional {tracer} {coefficient} estimates for {subject}")
    return fig


def gen_biased_color(col_name):
    if "py" in str(col_name):
        return (
            rng.random(),
            (rng.random() + 1.0) / 2.0,  # heavy on green for python
            rng.random() * 0.75,  # light on blue for python
            0.50
        )
    elif "ml" in str(col_name):
        return (
            rng.random(),
            rng.random() * 0.75,  # light on green for matlab
            (rng.random() + 1.0) / 2.0,  # heavy on blue for matlab
            0.50
        )
    else:
        return 0.0, 0.0, 0.0, 1.0


def prep_curve_df(df, base_tac, src_abbr, src_name):
    """ """

    local_df = df.copy()
    new_underscored_columns = [f"_{_:04d}_{src_abbr}" for _ in df.columns]
    local_df.columns = new_underscored_columns
    if len(local_df) == len(base_tac.timepoints):
        local_df['t'] = base_tac.timepoints
    elif len(local_df) == len(base_tac.post_peak_timepoints()):
        local_df['t'] = base_tac.post_peak_timepoints()
    else:
        raise ValueError(
            f"Length of curves ({len(local_df)}) "
            f"does not match full tac ({len(base_tac.timepoints)}) "
            f"or post-peak tac ({len(base_tac.post_peak_timepoints())}) "
        )
    local_df['src'] = src_name

    local_df_long = local_df.melt(id_vars=['src', 't'])
    local_df_long = local_df_long.rename(
        columns={"variable": "curve_id", "value": "activity"}
    )

    return local_df_long


def plot_tac_with_sd_lines(
        tac, ax, color='black', sd_color='gray', label='tac',
        points=None, linestyle='dashed', scatter=False,
):
    """ Plot one TAC, with its SD boundaries. """

    _n = len(tac.timepoints)
    if points is not None:
        _n = np.min([len(tac.timepoints), int(points), ])

    if tac.sd is not None:
        sns.lineplot(
            x=tac.timepoints[:_n], y=(tac.activity - tac.sd)[:_n],
            color=sd_color, linewidth="3", linestyle="dotted", label="_sd",
            alpha=0.50, ax=ax,
        )
        sns.lineplot(
            x=tac.timepoints[:_n], y=(tac.activity + tac.sd)[:_n],
            color=sd_color, linewidth="3", linestyle="dotted", label="_sd",
            alpha=0.50, ax=ax,
        )
    sns.lineplot(
        x=tac.timepoints[:_n], y=tac.activity[:_n],
        color=color, linewidth="5", linestyle=linestyle, label=label,
        alpha=0.50, ax=ax,
    )
    if scatter:
        sns.scatterplot(
            x=tac.timepoints[:_n], y=tac.activity[:_n], label="_",
            color=color, s=100, ax=ax,
        )


def plot_many_curves(primary_curves, primary_tac,
                     secondary_curves, secondary_tac,
                     subject_name, skip_outliers=False):
    """ """

    if skip_outliers:
        _primary_curves = winsorize_curves(primary_curves)
        _secondary_curves = winsorize_curves(secondary_curves)
    else:
        _primary_curves = primary_curves
        _secondary_curves = secondary_curves

    # Label primary and secondary dataframes, then concatenate them into one.
    ml_boot_curve_df_long = prep_curve_df(
        _secondary_curves, secondary_tac, 'ml', 'matlab',
    )
    py_boot_curve_df_long = prep_curve_df(
        _primary_curves, primary_tac, 'py', 'python',
    )
    all_boot_curves = pd.concat(
        [ml_boot_curve_df_long, py_boot_curve_df_long, ]
    ).sort_values(['curve_id', 't'])

    biased_palette = {}
    for curve_id in all_boot_curves['curve_id'].unique():
        biased_palette[curve_id] = gen_biased_color(curve_id)

    fig_bootstraps, axes = plt.subplots(nrows=2, figsize=(15, 20))

    # Plot the full-length TACs
    sns.lineplot(
        data=all_boot_curves,
        x='t', y='activity', hue='curve_id',
        palette=biased_palette, linewidth=1, ax=axes[0]
    )

    plot_tac_with_sd_lines(
        secondary_tac, axes[0],
        color='blue', sd_color='cornflowerblue', label='matlab'
    )
    plot_tac_with_sd_lines(
        primary_tac, axes[0],
        color='green', sd_color='palegreen', label='python'
    )

    axes[0].set_title("Full TAC")

    # Stretch out the early TACs for better visibility
    short_timepoints = np.array([_ for _ in primary_tac.timepoints if _ < 5.0])
    past_plotting_index = len(short_timepoints)

    sns.lineplot(
        data=all_boot_curves[all_boot_curves['t'] < 5.0],
        x='t', y='activity', hue='curve_id',  # label=None,
        palette=biased_palette, linewidth=1, ax=axes[1]
    )

    short_secondary_tac = TimeActivityCurve(
        activity=secondary_tac.activity[:past_plotting_index],
        timepoints=secondary_tac.timepoints[:past_plotting_index],
        source=secondary_tac.source,
        sd=(None if secondary_tac.sd is None
            else secondary_tac.sd[:past_plotting_index]),
    )
    plot_tac_with_sd_lines(
        short_secondary_tac, axes[1],
        color='blue', sd_color='cornflowerblue', label='matlab'
    )

    short_primary_tac = TimeActivityCurve(
        activity=primary_tac.activity[:past_plotting_index],
        timepoints=primary_tac.timepoints[:past_plotting_index],
        source=primary_tac.source,
        sd=(None if primary_tac.sd is None
            else primary_tac.sd[:past_plotting_index]),
    )
    plot_tac_with_sd_lines(
        short_primary_tac, axes[1],
        color='green', sd_color='palegreen', label='python'
    )

    axes[1].set_title("First five seconds")

    fig_bootstraps.suptitle(
        f"{subject_name}'s "
        f"{_primary_curves.shape[1]}/{primary_curves.shape[1]} "
        f"{primary_tac.source} and "
        f"{_secondary_curves.shape[1]}/{secondary_curves.shape[1]} "
        f"{secondary_tac.source} "
        f"curves"
    )

    return fig_bootstraps


def melt_tac_dataframe(df, t):
    """ Format a wide regional TAC dataframe into a long plottable dataframe.
    """

    if "t" in df.columns:
        plottable_data = df.drop("t", axis='columns')
    else:
        plottable_data = df.copy()
    plottable_data['minutes'] = t
    return pd.melt(
        plottable_data,
        id_vars=['minutes'], var_name="region", value_name="microCi"
    )


def plot_stare_tac_fits(
        tac_data, mid_times, source_region, target_fits, cost,
        figsize=(7.5, 7.5),
):
    """ Plot TACs from simulated annealing fitting,

        This function is not used anywhere.
    """

    fig, axes = plt.subplots(nrows=2, figsize=figsize)

    plottable_data = melt_tac_dataframe(tac_data, mid_times)
    source_data = plottable_data[plottable_data["region"] == source_region]
    target_data = plottable_data[plottable_data["region"] != source_region]

    tac_palette = palette_from_tac_regions(tac_data)

    plottable_fits = melt_tac_dataframe(pd.DataFrame(
        data=target_fits,
        columns=[f"fit_{col}" for col in tac_data.columns
                 if col not in [source_region, 't', ]],
    ), mid_times)

    sns.scatterplot(
        x="minutes", y="microCi", data=source_data,
        color='black', marker="*", ec='face', s=60,
        label=f"Source {source_region}",
        ax=axes[0]
    )
    sns.scatterplot(
        x="minutes", y="microCi", hue="region", data=target_data,
        palette=tac_palette,  # sns.color_palette("muted"),
        marker=r"$\circ$", ec='face', s=50,
        ax=axes[0]
    )
    axes[0].legend(bbox_to_anchor=(1.04, 0.5),
                   loc="center left", borderaxespad=0)
    axes[0].set_title(f"Source ({source_region}) and target region TACs")

    # Bottom panel
    sns.scatterplot(
        x="minutes", y="microCi", hue="region", data=target_data,
        palette=tac_palette,  # sns.color_palette("muted"),
        marker=r"$\circ$", ec='face', s=50,
        ax=axes[1]
    )
    sns.lineplot(
        x="minutes", y="microCi", hue="region", data=plottable_fits,
        palette=tac_palette, marker=None,
        ax=axes[1]
    )
    axes[1].legend(bbox_to_anchor=(1.04, 0.5),
                   loc="center left", borderaxespad=0)
    axes[1].set_title(f"Target region TACs and fits (cost {cost:0.4f})")

    fig.tight_layout()

    return fig


def plot_all_stare_tac_fits(
        tac_data, mid_times, optimizations, comparisons=None,
        figsize=(10.0, 7.5), title="TAC Fits",
):
    """ Plot TACs from simulated annealing fitting,

        This plotter is called after simulated_annealing,
        and in a simulated annealing debugger script.
    """

    # Original FDG testing was all done with 6 regions, so a 2-row x 3-col
    # grid was great. But with variable regions, this needs to be more
    # versatile. How should we lay out the grid?
    n_cols = int(np.ceil(np.sqrt(tac_data.shape[1])))  # for 6, round 2.4 to 3
    n_rows = int(np.ceil(tac_data.shape[1] / n_cols))
    if figsize is None:
        figsize = ((n_cols * 3) + 2, n_rows * 3)
    fig = plt.figure(figsize=figsize, layout="tight")
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)
    ax_handles = []
    ax_labels = []

    plottable_data = melt_tac_dataframe(tac_data, mid_times)

    tac_palette = palette_from_tac_regions(tac_data)

    row, col = 0, -1
    for i, source_region in enumerate(tac_data.columns):
        # Figure out what axes we're dealing with
        col = col + 1
        if col >= n_cols:
            col = 0
            row = row + 1
        panel = fig.add_subplot(gs[row, col])

        # Divide and format the data for plotting
        src_optimization = None
        for optimization in optimizations:
            if optimization['source_tac'].name == source_region:
                src_optimization = optimization
        source_data = plottable_data[plottable_data["region"] == source_region]
        target_data = plottable_data[plottable_data["region"] != source_region]
        plottable_fits = melt_tac_dataframe(pd.DataFrame(
            data=src_optimization['tgt_tac_fits'],
            columns=[f"fit_{col}" for col in tac_data.columns
                     if col not in [source_region, 't', ]],
        ), mid_times)

        # Draw each panel
        panel.set_title(
            "Target TACs (fit cost {:0.5f})\nsource {}".format(
                src_optimization["cost"], source_region,
            )
        )
        sns.lineplot(
            x="minutes", y="microCi", hue="region", data=plottable_fits,
            palette=tac_palette, marker=None,
            ax=panel
        )
        sns.scatterplot(
            x="minutes", y="microCi", hue="region", data=target_data,
            palette=tac_palette,  # sns.color_palette("muted"),
            marker=r"$\circ$", ec='face', s=60,
            ax=panel
        )
        sns.scatterplot(
            x="minutes", y="microCi", data=source_data,
            color=tac_palette[source_region], marker="*", s=120,
            label=f"{source_region} (source)",
            ax=panel
        )

        # Optionally, also plot matlab results for comparison
        if comparisons is not None and i in comparisons:
            plottable_comp = melt_tac_dataframe(pd.DataFrame(
                data=comparisons[i]["tgt_tac_fits"],
                columns=[f"fit_{col}"
                         for col in tac_data.columns
                         if col not in [source_region, 't', ]],
            ), mid_times, )
            sns.lineplot(
                x="minutes", y="microCi", hue="region", data=plottable_comp,
                palette=tac_palette, linestyle=":", linewidth=2,
                label=None, legend=False, marker=None,
                ax=panel
            )
            panel.text(
                0.98, 0.02,
                f"matlab cost {comparisons[i]['cost']:0.4f}",
                ha='right', va='bottom', transform=panel.transAxes,
            )

        # Legends
        # Do not plot legends on axes. Create a new legend in the spare column
        handles, labels = panel.get_legend_handles_labels()
        ax_handles += handles
        ax_labels += labels
        panel.get_legend().remove()

    # Prune the long list of multiple legend items
    # and create a new legend, in order of regions
    final_legend_handles, final_legend_labels = [], []
    for label_template in ["{}", "fit_{}", ]:
        for region in tac_data.columns:
            data_source = label_template.format(region)
            for i, label in enumerate(ax_labels):
                if (
                        label == data_source
                        and label not in final_legend_labels
                        and "source" not in ax_labels[i]
                ):
                    final_legend_handles.append(ax_handles[i])
                    final_legend_labels.append(ax_labels[i])
    fig.legend(
        final_legend_handles, final_legend_labels,
        bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0
    )
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_ks(py_data, ml_data, title="Parameter distributions", figsize=(16, 8)):
    """ Plot k values """

    params = sorted(set(py_data['k']).union(ml_data['k']))
    fig, axes = plt.subplots(
        ncols=len(params), figsize=figsize, layout="tight"
    )
    fig.suptitle(title)

    # Save and re-use axes legend components
    ax_handles, ax_labels = [], []

    # For summarizing python data,
    py_sum_data = py_data.groupby(["tgt", "k"])['value'].mean().reset_index()
    palette = palette_from_tac_regions(py_sum_data)

    for i, k in enumerate(params):
        panel = axes[i]
        panel.set_title(k)
        sns.stripplot(
            data=ml_data[ml_data['k'] == k], x='tgt', y='value',
            color='black', marker="D", s=10, alpha=0.5, label="matlab",
            ax=panel
        )
        sns.stripplot(
            data=py_sum_data[py_sum_data['k'] == k], x='tgt', y='value',
            c='gray', marker="o", s=10, alpha=0.5, label="python",
            ax=panel
        )
        sns.stripplot(
            data=py_data[py_data['k'] == k], x='tgt', y='value',
            hue='src', palette=palette, s=5, ax=panel
        )
        # Do not plot legends on axes. Create a new legend in the spare column
        handles, labels = panel.get_legend_handles_labels()
        ax_handles += handles
        ax_labels += labels
        panel.get_legend().remove()

    # Prune the long list of multiple legend items
    # and create a new legend, in order of regions
    final_legend_handles, final_legend_labels = [], []
    for i, label in enumerate(ax_labels):
        if label not in final_legend_labels:
            final_legend_handles.append(ax_handles[i])
            final_legend_labels.append(ax_labels[i])
    fig.legend(
        final_legend_handles, final_legend_labels,
        ncol=len(final_legend_labels),
        bbox_to_anchor=(0.5, 0.0), loc="upper center", borderaxespad=0
    )

    return fig


def plot_cluster_comparisons(
        probability_mask, agreement_data, ax_hist, ax_strip
):
    """ Plot a voxel-wise histogram and a cluster-wise strip plot. """

    vals, cnts = np.unique(probability_mask, return_counts=True)
    num_selected_voxels = np.sum(cnts[1:])
    annot_text = ""
    for i in range(1, len(vals)):
        new_line = "{:,} voxels ({:0.0%}) with {:0.0%} agreement\n".format(
            cnts[i], cnts[i] / num_selected_voxels, vals[i]
        )
        annot_text = new_line + annot_text
    annot_text = (f"{num_selected_voxels:,} voxels selected by any iteration"
                  "\n----\n" + annot_text)

    sns.histplot(
        probability_mask.ravel()[np.nonzero(probability_mask.ravel())],
        ax=ax_hist
    )
    ax_hist.annotate(annot_text, (0.5, np.max(cnts[1:])), ha='center', va='top')
    ax_hist.set_title(f"Overlapping Voxels across runs")
    ax_hist.set_xlim(-0.05, 1.05)
    ax_hist.set_xticks([0.0, 0.5, 1.0, ])
    ax_hist.set_xticklabels(['0', '0.5', '1', ])
    ax_hist.set_ylabel("How many voxels in pct agreement bin")

    # Plot the strip plot, taking the bottom 20%
    sns.stripplot(
        data=agreement_data, x='dice', y='subject',
        jitter=True, ax=ax_strip
    )
    ax_strip.set_title(f"Overlap between whole masks")
    ax_strip.set_xlim(-0.05, 1.05)
    ax_strip.set_xticks([0.0, 0.5, 1.0, ])
    ax_strip.set_xticklabels(['0', '0.5', '1', ])
    ax_strip.set_yticks([])
    ax_strip.set_yticklabels([])
    ax_strip.set_ylabel("")


def line_props_from_tac(tac):
    """ Provide line styles for different classes of TAC. """

    plot_color = 'black'
    line_style = "."
    if likely_vascular(tac):
        plot_color = 'green'
        line_style = "-"
    elif likely_noise(tac):
        plot_color = 'gray'
        line_style = ":"
    elif likely_irreversible(tac):
        plot_color = 'red'
        line_style = ":"
    elif likely_peripheral(tac):
        plot_color = 'orange'
        line_style = "--"

    return dict(color=plot_color, linestyle=line_style)


def tacs_from_ics(mixing_matrix, original_data, mid_times=None):
    """ Return positive and negative components of each col of a mixing matrix.

        Each component may correlate negatively or positively with the
        original image, so we need to identify which one may be of use to
        us. This function doesn't diagnose anything, but extracts the
        positive and negative thresholds separately and returns them both.
    """

    if mid_times is None:
        # Just fill in a vanilla sequence if we don't know actual timing
        mid_times = list(range(original_data.shape[1]))

    tacs = {"hi": dict(), "lo": dict(), "all": dict(), }
    for i in range(mixing_matrix.shape[1]):
        _mat = mixing_matrix[:, i]
        _mu, _sd = np.mean(_mat), np.std(_mat)
        for direction in ("hi", "lo", "all",):
            # Create a 2-SD threshold mask
            if direction == "hi":
                _mask = _mat > _mu + _sd + _sd
            elif direction == "lo":
                _mask = _mat < _mu - _sd - _sd
            else:
                _mask = ((_mat > _mu + _sd + _sd) | (_mat < _mu - _sd - _sd))
            # Format it as a TimeActivityCurve
            _tac = TimeActivityCurve(
                np.mean(original_data[_mask], axis=0),
                mid_times,
                f"ica {direction}"
            )
            # Store it in the dict
            tacs[direction][i] = _tac

    return tacs


def plot_components(mixing_matrix, original_data,
                    mid_times=None, title="", save_as=None):
    """ Plot each component in the mixing matrix.

        :param mixing_matrix: The mixing matrix from fitting components
        :param original_data: The 2D data used to fit components
        :param mid_times: TAC mid-times for proper TAC spacing
        :param str title: The title of the plot
        :param save_as: If plot should be saved, the name of the file
    """

    if mixing_matrix.shape[1] < 4:
        n_rows, n_cols = 1, mixing_matrix.shape[1]
    elif mixing_matrix.shape[1] < 9:
        n_rows, n_cols = 2, ceil(mixing_matrix.shape[1] / 2)
    else:
        n_rows, n_cols = 3, ceil(mixing_matrix.shape[1] / 3)
    fig_tacs, axes_tacs = plt.subplots(
        nrows=n_rows, ncols=n_cols, sharex='all', sharey='all', figsize=(11, 5)
    )
    tacs = tacs_from_ics(mixing_matrix, original_data, mid_times)
    i = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if n_rows == 1:
                _ax = axes_tacs[i]
            else:
                _ax = axes_tacs[row, col]
            if i < mixing_matrix.shape[1]:
                tac_hi = tacs["hi"][i]
                tac_lo = tacs["lo"][i]
                sns.lineplot(x=tac_hi.timepoints, y=tac_hi.activity, ax=_ax,
                             **line_props_from_tac(tac_hi))
                sns.lineplot(x=tac_hi.timepoints, y=tac_lo.activity, ax=_ax,
                             **line_props_from_tac(tac_lo))
                _ax.set_title(f"{title[0:2]} {i}")
                if likely_vascular(tac_hi):
                    tacs[(i, "pos")] = tac_hi
                if likely_vascular(tac_lo):
                    tacs[(i, "neg")] = tac_lo
            i += 1

    fig_tacs.suptitle(title)

    if save_as is not None:
        fig_tacs.savefig(save_as)

    return fig_tacs


def plot_mixing_matrix(mixing_matrix, title="A", save_as=None):
    """ Plot the mixing matrix itself.

        :param mixing_matrix: The mixing matrix from fitting components
        :param str title: The title of the plot
        :param save_as: If plot should be saved, the name of the file
    """

    fig_mixing, axes_mixing = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(mixing_matrix, ax=axes_mixing)
    axes_mixing.set_title(title)

    if save_as is not None:
        fig_mixing.savefig(save_as)
    return fig_mixing


def plot_pca_variance(pca_transformer, title="", save_as=None):
    """ Plot the mixing matrix itself.

        :param pca_transformer: The fit PCA model from scikit-learn
        :param str title: The title of the plot
        :param save_as: If plot should be saved, the name of the file
    """

    plot_data = pd.DataFrame({
        "iter": [i for i in range(pca_transformer.n_components_)],
        "var_exp": pca_transformer.explained_variance_ratio_,
    })

    fig_var_exp, axes_var_exp = plt.subplots(figsize=(3, 2.5))
    sns.barplot(
        data=plot_data, x="iter", y="var_exp", ax=axes_var_exp
    )
    axes_var_exp.set_title(title)

    if save_as is not None:
        fig_var_exp.savefig(save_as)
    return fig_var_exp


def plot_confetti_score_on_mask_z(data, name=""):
    """ Plot the positive and negative evidence a given mask is "just confetti"

        :param data: A dataframe, generated as a centroid feature
        :returns figure, axes: The plot figure and axes
    """

    if data is None:
        return None, None

    fig_weights, axes_weights = plt.subplots(
        ncols=3, figsize=(4, 7), layout='tight'
    )

    wt_ax = axes_weights[0]
    sns.lineplot(data=data[data['var'] == 'weight'], x='val', y='z', orient="y",
                 ax=wt_ax)
    wt_ax.axvline(x=0, color='gray', linestyle='--')
    wt_ax.set_xlabel('weight')
    wt_ax.set_xticks([0.0, ])
    wt_ax.set_xticklabels(['0', ])

    score_ax = axes_weights[1]
    scatter_data = data[(data['var'] == 'score') & (data['valence'] != 'neutral')]
    sns.scatterplot(data=scatter_data, x='val', y='z',
                    hue='valence', palette={'positive': 'green', 'negative': 'red', 'neutral': 'gray', },
                    legend=False,
                    ax=score_ax)
    neg_sum = scatter_data[scatter_data['val'] < 0]['val'].sum()
    pos_sum = scatter_data[scatter_data['val'] > 0]['val'].sum()
    equation = f"{neg_sum:0.2f}\n+\n{pos_sum:0.2f}\n=\n{neg_sum + pos_sum:0.2f}"
    score_ax.text(0.50, 0.99, equation, ha='center', va='top',
                  transform=score_ax.transAxes)
    score_ax.axvline(x=0, ymin=0, ymax=0.75, color='gray', linestyle='--')
    score_ax.set_xticks([0.0, ])
    score_ax.set_xticklabels(['0', ])
    score_ax.set_yticklabels([])
    score_ax.set_ylabel('')
    score_ax.set_ylim(wt_ax.get_ylim())
    score_ax.set_xlabel('score')

    bar_ax = axes_weights[2]
    sns.barplot(data=data[data['var'] == 'ratio'], x='val', y='z', orient="y",
                order=data[data['var'] == 'ratio']['z'][::-1],
                ax=bar_ax)
    bar_ax.set_xticks([])
    bar_ax.set_xticklabels([])
    bar_ax.set_yticklabels([])
    bar_ax.set_ylabel('')
    bar_ax.set_xlabel('mask')

    fig_weights.suptitle(name)

    return fig_weights, axes_weights
