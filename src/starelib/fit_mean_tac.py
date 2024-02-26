import sys
import numpy as np
import pickle
from scipy.interpolate import interp1d, PchipInterpolator

from .timeactivitycurve import TimeActivityCurve
from .plotting import plot_detailed_tacs
from .fitting_models import decay_model, find_curve_fits
from .util import from_cache, to_cache


def select_best_fit(fits, weighted=True):
    """ From any number of curve fits, return the one with the lowest MSE

        :param Iterable fits: An iterable of (parameters, covariance) tuples
        :param bool weighted: To use raw sum of squares, set to False
        :returns: the best tuple from within fits
    """

    cost_term = 'wrms' if weighted else 'rms'
    best_fit = None
    for fit in fits:
        if best_fit is None:
            best_fit = fit
        elif fit.get(cost_term, np.Inf) < best_fit.get(cost_term, 0.0):
            best_fit = fit

    return best_fit


def interpolate_full_tac(
        actual_tac, best_fit, model, tac_name="high res decay model"
):
    """ Interpolate pre- and post-peak separately and return combined TAC.

        Actual time points are sparse and spread out non-linearly, but it
        is nice to have smooth curves with evenly spaced points. This function
        takes a sparse 'actual_tac' and uses interpolation for data points
        before the peak, combined with curve fitting from 'model' to calculate
        points beyond the peak, and puts them all together to create a smooth,
        evenly spaced TAC representation of 'actual_tac'.

        :param TimeActivityCurve actual_tac: TAC containing measured data
        :param dict best_fit: dict with parameters from fitting decay_model
        :param function model: Function used to attain best_fit
        :param str tac_name: A name for the returned TAC
    """

    if best_fit is None:
        return None

    # For debugging comparison:
    hi_res_tac = actual_tac.get_uniform_time_curve()

    # Calculate uniform post-peak activity values via prior fit.
    post_peak_hi_res_fit = model(
        hi_res_tac.post_peak_timepoints(), *best_fit['parameters']
    )

    # Interpolate uniform pre-peak activity values via prior fit.
    # pre-peak activity must include one more time point to include the peak.
    linear_interpolator = interp1d(
        actual_tac.timepoints[:actual_tac.peak_index + 1],
        actual_tac.activity[:actual_tac.peak_index + 1],
        fill_value='extrapolate',
    )
    pre_peak_hi_res_fit = linear_interpolator(
        hi_res_tac.timepoints[:hi_res_tac.peak_index + 1]
    )

    # Concatenate pre- and post- peak into one full-length TAC
    hi_res_tac.activity = np.concatenate([
        pre_peak_hi_res_fit[:-1], post_peak_hi_res_fit,
    ])
    hi_res_tac.source = "interpolation"
    hi_res_tac.name = tac_name

    # Resample full high-res fit back down to original mid_times.
    pchip_interpolator = PchipInterpolator(
        hi_res_tac.timepoints, hi_res_tac.activity,
    )

    orig_res_tac = TimeActivityCurve(
        timepoints=actual_tac.timepoints,
        activity=pchip_interpolator(actual_tac.timepoints),
        missing_timepoints=actual_tac.missing_timepoints,
        source="decay_model",
        name="original res decay model",
    )

    return orig_res_tac, hi_res_tac


def html_equation_from_fit(fit):
    """ Return html for an equation representing the best fit parameters.
    """

    best_params = fit.get("parameters")
    pairs = sorted(
        [
            (best_params[0], best_params[1]),
            (best_params[2], best_params[3]),
            (best_params[4], best_params[5]),
        ],
        key=lambda pair: pair[0],
        reverse=True
    )
    equation = " + ".join([
        "{:0.2f}e^{{{:0.2f}t}}".format(pair[0], pair[1]) for pair in pairs
    ])
    return f"$${equation}$$ with final wRMSE {fit.get('wrms'):0.3f}"


def fit_vascular_mean_tac(results):
    """ Fit vascular mean TAC

        :param StareResults results: An object containing pipeline data
        :returns: TAC from weighted model fit
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Fit exponential decay model")

    # -------------------------------------------------------------------------
    # Step 3. Correct TACs by extracting the mean signal from each cluster.
    # Needs to know about ignored mid-times to weight durations appropriately

    cache_file = f"sub-{results.args.subject}_step-3_decay_model_fits.pkl"
    successes, failures = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if successes is None:
        # Fit repeatedly until we have ten successes or complete failure.
        # Only fitting decay of activity past the peak - not pre-peak rise
        successes, failures = find_curve_fits(
            decay_model,
            results.pvc_mean_vascular_tac.post_peak_timepoints(),
            results.pvc_mean_vascular_tac.post_peak_activity(),
            sigmas=results.pvc_mean_vascular_tac.post_peak_sigmas(
                method='sqrt'
            ),
            success_limit=10, failure_limit=16384
        )
        to_cache((successes, failures), results.args.cache_path, cache_file)
    else:
        logger.info("  loaded cached step 3 decay model fits to save time")

    if results.args.debug and results.args.debug_path.exists():
        with open(
                results.args.debug_path /
                f"sub-{results.args.subject}_fits.pkl",
                "wb"
        ) as f:
            pickle.dump(successes, f)

    if successes is None:
        logger.error("No fits were found for PVC data. "
                     "STARE cannot continue. "
                     "The mostly likely problem is a bad cluster selection?")
        sys.exit(1)

    best_fit = select_best_fit(successes)
    lores_tac, hires_tac = interpolate_full_tac(
        results.pvc_mean_vascular_tac, best_fit, decay_model
    )
    lores_tac.name = "decay model fit"
    lores_tac.sd = results.pvc_mean_vascular_tac.sd

    fig = plot_detailed_tacs([lores_tac, hires_tac])
    fig_name = f"sub-{results.args.subject}_step-3_fits_hi_v_lo.png"
    fig.savefig(results.args.fig_path / fig_name)
    caption = f"Decay model fit in high and low resolution"
    rpt_sect.add_figure(results.args.fig_path / fig_name, caption,
                        css_class='right_fig')

    rpt_sect.add_line(f"{len(successes)} curves were successfully fit, "
                      f"amid {len(failures)} failures, "
                      f"to the three-level exponential decay model. "
                      "The best fit (past the peak) is shown below in "
                      "original and high resolution.")
    rpt_sect.add_line(html_equation_from_fit(best_fit), css_class='equation')

    if results.args.debug and results.args.debug_path.exists():
        with open(
                results.args.debug_path /
                f"sub-{results.args.subject}_tac_from_fitting.pkl",
                "wb"
        ) as f:
            pickle.dump(lores_tac, f)
        with open(
                results.args.debug_path /
                f"sub-{results.args.subject}_hires_tac_from_fitting.pkl",
                "wb"
        ) as f:
            pickle.dump(hires_tac, f)

    # Return the one best, properly weighted, interpolated TAC in original res.
    # This can be used to reduce the vascular influence on measured TACs later.
    results.fitting_successes = successes
    results.fitted_tac = lores_tac
    results.fitted_hires_tac = hires_tac

    rpt_sect.end()
    results.write_report()
    return results
