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

from .timeactivitycurve import TimeActivityCurve
from .util import get_kde_fwhm_points


# One global random number generator should be sufficient
rng = np.random.default_rng()

# Make a color palette matching Betsy's in matplotlib
betsy_palette = {
    "cerfullcs_c": 'blue', "fit_cerfullcs_c": 'blue',
    "cerfullc_c": 'blue', "fit_cerfullc_c": 'blue',
    "cerebellum": 'blue', "fit_cerebellum": 'blue',
    "cin": 'red', "fit_cin": 'red',
    "cingulate": 'red', "fit_cingulate": 'red',
    "hip": 'orange', "fit_hip": 'orange',
    "hippocampus": 'orange', "fit_hippocampus": 'orange',
    "par": 'purple', "fit_par": 'purple',
    "parietal": 'purple', "fit_parietal": 'purple',
    "pph": 'green', "fit_pph": 'green',
    "med": 'green', "fit_med": 'green',
    "prefrontal": 'green', "fit_prefrontal": 'green',
    "pip": 'cyan', "fit_pip": 'cyan',
    "parahippocampal": 'cyan', "fit_parahippocampal": 'cyan',
}


def palette_from_tac_regions(data):
    """ Generate a palette to cover all regions in data.
    """

    palette = dict()
    for region in data.columns:
        if region not in betsy_palette.keys():
            palette[region] = 'gray'
    palette.update(betsy_palette)

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
            for i, activity in enumerate(tac.activity):
                row = {
                    "t": tac.timepoints[i],  # Ignore index values, take i'th t
                    "activity": activity,  # The y-axis plotted value, in mCis
                    "k": tac.k if hasattr(tac, 'k') else 0,
                    "label": tac.label if hasattr(tac, 'label') else 0,
                    "source": tac.source,
                    "best_overall": (tac.best_overall
                                     if hasattr(tac, 'best_overall')
                                     else False),
                    "best_in_k": (tac.best_in_k
                                  if hasattr(tac, 'best_in_k')
                                  else False),
                    "name": "n/a" if tac.name is None else tac.name,
                }
                for feature, label in tac.features.items():
                    row[feature] = label
                rows.append(row)

    return pd.DataFrame(rows)


def plot_vascular_tacs(
        data, vascular_color='blue', highlight_color='red',
        tall=False, large=False, ax=None
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
        :param ax: Optionally draw on your own axes
        :returns Figure:
    """

    if tall:
        figsize=(6, 6)
    elif large:
        figsize=(16, 9)
    else:
        figsize=(10, 6)
    if ax is None:
        fig, axes = plt.subplots(figsize=figsize, layout='tight')
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
    sns.lineplot(data=data[data['likely_vascular']], x='t', y='activity',
                 hue='name', palette=grays, alpha=0.5, linewidth=1,
                 legend=False, ax=axes)

    # Next, plot the centroids that are the best for their k-means group
    sns.lineplot(
        data=data[data['best_in_k']], x="t", y="activity",
        hue='name', palette=vascs, alpha=0.5, linewidth=1,
        ax=axes
    )
    if tall:
        # clear all legend labels, allowing the next lineplot to make a new one.
        for line in axes.lines:
            line.set_label(s='')

    # Finally, plot the very best centroid of the whole batch.
    best_k = data[data['best_overall']]['k'].unique()[0]
    best_label = data[data['best_overall']]['label'].unique()[0]
    sns.lineplot(data=data[data['best_overall']], x="t", y="activity",
                 color=highlight_color, linestyle=":", linewidth=6, alpha=0.5,
                 label=f"Best (k-{best_k:02d}-{best_label:02d})", ax=axes)

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
    fig.suptitle(f"Optimal vascular TACs: {data['k'].min()}-{data['k'].max()}"
                 " k-means clusters")

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


def plot_top_centroids_atlas(
        step_1_mask_img, step_2_mask_img, pet4d_img, figsize=(8, 4)
):
    """ Plot the PET average background and masks over the top.

    """

    mean_pet_img = image.mean_img(pet4d_img)
    atlas_combo_img = nib.Nifti1Image(
        step_1_mask_img.get_fdata() + step_2_mask_img.get_fdata(),
        affine=mean_pet_img.affine, dtype=np.uint8,
    )
    two_grade_cmap = ListedColormap(['orange', 'red', ])
    fig, axes = plt.subplots(figsize=figsize)
    plot_roi(roi_img=atlas_combo_img, bg_img=mean_pet_img,
             cmap=two_grade_cmap, black_bg=False, axes=axes,)

    return fig


def plot_detailed_tacs(data, title=None, palette=None, dashes=None,
                       color_filter=None, figsize=(11, 11)):
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
    gs = gridspec.GridSpec(nrows=7, ncols=4)

    ax_full = fig.add_subplot(gs[0:3, :])
    ax_early = fig.add_subplot(gs[5:, :2])
    ax_late = fig.add_subplot(gs[5:, 2:])
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
    if dashes is None or len(dashes) == 0:
        dashes = None

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
        if color_filter is None or str(color_filter) not in data.columns:
            c_filter = [True, ] * len(data)
        else:
            c_filter = data[color_filter]

        # Plot every single centroid as light gray to provide context.
        # These are plotted first to set them as background.
        grays = ['gray', ] * len(data[t_filter]['name'].unique())
        sns.lineplot(data=data[t_filter],
                     x="t", y="activity", hue='name',
                     palette=grays, alpha=0.5, linewidth=1, legend=False,
                     estimator=None, ax=ax)

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
        ax.set_xlabel("Minutes")  # ranges 0 to 60

    ax_early.set_ylabel("Activity in mCi/cc")  # ranges -0.05 to +2.00
    ax_late.set_ylabel("Activity in mCi/cc")  # ranges typically -0.05 to +0.30
    ax_late.yaxis.set_label_position("right")
    ax_late.get_yaxis().tick_right()

    ax_full.legend(bbox_to_anchor=(0.50, -0.25),
                   loc="upper center", borderaxespad=0)

    ax_full.set_title("Full scan")
    ax_early.set_title("Early")
    ax_late.set_title("Late")

    fig.suptitle(f"Optimal vascular TACs" if title is None else title)
    # Do NOT use tight_layout; it carves out extra space for the legend
    # fig.tight_layout()

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


def plot_bootstrap_constant(
        region_names,
        rate_constants,
        k, subject, tracer
):
    """ Plot density of constants for each of six regions """

    fig = plot_regional_densities(
        region_names=region_names,
        rate_constants=rate_constants,
        num_bootstraps=1000,
        coefficient=k,
        subject=subject,
        tracer=tracer,
        verbose=False
    )

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
    xl, xc, xr = fwhm[:, 0]  # along the x axis: left, center, right
    yb = 0.0
    yc, yt, _ = fwhm[:, 1]  # along the y axis: bottom, center, top

    if verbose:
        print(f"{label_prefix}: "
              f"({xl:0.3f}, {yc:0.1%}), "
              f"({xc:0.3f}, {yt:0.1%}), "
              f"({xr:0.3f}, {yc:0.1%})")

    ax.plot(kde_x, kde_y, ls=":", c=color)

    ax.add_artist(lines.Line2D(
        xdata=[xl, xl, xr, xr, ],
        ydata=[yb, yc, yc, yb, ],
        color=color,
        label=" ".join([label_prefix,
                        f"HM {yc:0.1%} @ ({xl:0.3f} to {xr:0.3f})"]),
    ))
    ax.add_artist(lines.Line2D(
        xdata=[xc, xc, ],
        ydata=[yc, yt, ],
        color=color,
        label=" ".join([label_prefix,
                        f"Peak {yt:0.1%} @ {xc:0.3f}"]),
    ))
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
    """ Plot 6 regions in a figure.

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

    fig, axes = plt.subplots(
        nrows=2, ncols=3, sharex='all', sharey='all', figsize=(15, 11)
    )

    i = 0
    for row in range(2):
        for col in range(3):
            if len(region_names) > i:
                ax = axes[row, col]
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
        tac, ax, color='black', sd_color='gray', label='tac'
):
    """ Plot one TAC, with its SD boundaries. """

    if tac.sd is not None:
        sns.lineplot(
            x=tac.timepoints, y=tac.activity - tac.sd,
            color=sd_color, linewidth="3", linestyle="dotted", label="_sd",
            alpha=0.50, ax=ax,
        )
        sns.lineplot(
            x=tac.timepoints, y=tac.activity + tac.sd,
            color=sd_color, linewidth="3", linestyle="dotted", label="_sd",
            alpha=0.50, ax=ax,
        )
    sns.lineplot(
        x=tac.timepoints, y=tac.activity,
        color=color, linewidth="5", linestyle="dashed", label=label,
        alpha=0.50, ax=ax,
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
        x='t', y='activity', hue='curve_id', label=None,
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
        marker="$\circ$", ec='face', s=50,
        ax=axes[0]
    )
    axes[0].legend(bbox_to_anchor=(1.04, 0.5),
                   loc="center left", borderaxespad=0)
    axes[0].set_title(f"Source ({source_region}) and target region TACs")

    # Bottom panel
    sns.scatterplot(
        x="minutes", y="microCi", hue="region", data=target_data,
        palette=tac_palette,  # sns.color_palette("muted"),
        marker="$\circ$", ec='face', s=50,
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

    fig = plt.figure(figsize=figsize, layout="tight")
    gs = gridspec.GridSpec(2, 3, figure=fig)
    ax_handles = []
    ax_labels = []

    plottable_data = melt_tac_dataframe(tac_data, mid_times)

    tac_palette = palette_from_tac_regions(tac_data)

    row, col = 0, -1
    for i, source_region in enumerate(tac_data.columns):
        # Figure out what axes we're dealing with
        col = col + 1
        if col > 2:
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
            "Target TACs (source {})\n(fit cost {:0.5f})".format(
                source_region, src_optimization["cost"]
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
            marker="$\circ$", ec='face', s=60,
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
            hue='src', palette=betsy_palette, s=5, ax=panel
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
