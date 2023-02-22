import numpy as np
import pickle
import logging
from scipy.interpolate import interp1d, PchipInterpolator

from .timeactivitycurve import TimeActivityCurve
from .plotting import plot_detailed_tacs
from .fitting_models import decay_model, find_curve_fits


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


def interpolate_full_tac(actual_tac, best_fit, model,
                         tac_name="high res decay model"):
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

    peak_activity_index = np.argmax(actual_tac.activity)
    post_peak_mid_times = actual_tac.timepoints[peak_activity_index:]

    # Set up uniform sampling intervals, with 0.1min between samples.
    post_peak_uniform_t = np.asarray([_ / 10.0 for _ in range(
        int(10.0 * round(post_peak_mid_times[0], 1)),
        int(10.0 * post_peak_mid_times[-1] + 1.0),
    )])
    pre_peak_uniform_t = np.asarray([_ / 10.0 for _ in range(
        0,
        int(10.0 * round(post_peak_mid_times[0], 1)),
    )])
    all_uniform_t = np.concatenate([pre_peak_uniform_t, post_peak_uniform_t, ])

    # Calculate uniform post-peak activity values via prior fit.
    post_peak_uniform_fit = model(
        post_peak_uniform_t, *best_fit['parameters']
    )

    # Interpolate uniform pre-peak activity values via prior fit.
    linear_interpolator = interp1d(
        actual_tac.timepoints[:peak_activity_index + 1],
        actual_tac.activity[:peak_activity_index + 1],
        fill_value='extrapolate',
    )
    pre_peak_uniform_fit = linear_interpolator(pre_peak_uniform_t)

    # Concatenate pre- and post- peak into one full-length TAC
    high_res_tac = TimeActivityCurve(
        activity=np.concatenate([
            pre_peak_uniform_fit, post_peak_uniform_fit,
        ]),
        timepoints=all_uniform_t,
        source="interpolation",
        name=tac_name,
    )

    # Resample full high-res fit back down to original mid_times.
    pchip_interpolator = PchipInterpolator(
        high_res_tac.timepoints, high_res_tac.activity,
    )

    orig_res_tac = TimeActivityCurve(
        timepoints=actual_tac.timepoints,
        activity=pchip_interpolator(actual_tac.timepoints),
        missing_timepoints=actual_tac.missing_timepoints,
        source="decay_model",
        name="original res decay model",
    )

    return orig_res_tac, high_res_tac


def fit_vascular_mean_tac(
        pvc_tac, figure_path,
        debug_path=None, cache_path=None, force=False,
        verbose=1
):
    """ Fit vascular mean TAC

        :param TimeActivityCurve pvc_tac: The best TAC thus far (from PVC)
        :param Path figure_path: The path for saving out figures
        :param Path debug_path: The path for saving out debug information
        :param Path cache_path: The path for saving out cached data
        :param bool force: Set to true to override caches and recalculate
        :param int verbose: Set to non-zero to trigger logging, higher is more

        :returns: TAC from weighted model fit
    """

    """ If necessary for testing, use this TAC
    # Test TAC
    from .timeactivitycurve import TimeActivityCurve
    vascular_tac = TimeActivityCurve(
        activity=np.array([
            -0.0168, -0.0202, 0.1915, 1.2577, 2.0860, 1.8862, 0.5807, 0.4264,
            0.3567, 0.3860, 0.3703, 0.3676, 0.3018, 0.3472, 0.3442, 0.2877,
            0.2800, 0.2177, 0.2185, 0.1985, 0.1899, 0.1425, 0.1483, 0.1196,
            0.1191,
        ]),
        timepoints=np.array([
            0.1250, 0.3750, 0.6250, 0.8750, 1.1250, 1.3750, 1.6250, 2.2500,
            2.7500, 3.2500, 3.7500, 4.2500, 4.7500, 5.5000, 6.5000, 7.5000,
            8.5000, 9.5000, 12.500, 17.500, 22.500, 27.500, 35.000, 45.000,
            55.0000,
        ]),
        source="made_up",
    )
    """

    """
    # Calculate timing blocks
    # Weights are calculated on the duration of each block of time in
    # the TAC, but the duration must be calculated on ALL blocks, even
    # if ignored, because the duration is unknown if the block before
    # or after was removed.
    time_blocks = characterize_mid_times(
        vascular_tac.timepoints,
        missing_mid_times=vascular_tac.missing_timepoints
    )

    # Get weights (square root of frame duration) for fitting.
    # Original matlab code used 'weights', but python curve fitting
    # uses 'sigmas' instead, so we'll calculate both here, but only
    # use 'sigmas', which approximate underlying standard deviation.
    # If a data point represents 5x the timespan, it should be weighted
    # 5 times more heavily (but isn't - it's actually sqrt(5) in matlab)
    # which is the same as sigma = 1/sqrt(5) or sqrt(5)/5.
    time_blocks['full_weight'] = np.real(time_blocks['duration'])
    time_blocks['weight'] = np.real(np.sqrt(time_blocks['duration']))
    time_blocks['full_sigma'] = np.real(1 / np.sqrt(time_blocks['full_weight']))
    time_blocks['sigma'] = np.real(1 / np.sqrt(time_blocks['weight']))
    usable_sigmas = time_blocks[time_blocks['used']]['sigma'].values

    # Get midtime and vascular TAC data post-peak
    peak_activity_index = np.argmax(vascular_tac.activity)
    post_peak_mid_times = vascular_tac.timepoints[peak_activity_index:]
    post_peak_activity = vascular_tac.activity[peak_activity_index:]
    post_peak_sigmas = usable_sigmas[peak_activity_index:]

    # Just to compare prior buggy results with corrected, do this wrong.
    # no missing mid_times are included, so characterization goes bad.
    bad_blocks = characterize_mid_times(vascular_tac.timepoints)  
    bad_blocks['sigma'] = np.real(1 / np.sqrt(np.sqrt(bad_blocks['duration'])))
    duration_sigmas = time_blocks[
        time_blocks['used']
    ]['full_sigma'].values[peak_activity_index:]

    tacs = {"vascular": vascular_tac, }
    hires_tacs = {}
    # Fit the data to the decay_model (here we fit three variants of the data)
    for attempt in [
        {"name": "raw", "sigmas": None, },
        {"name": "bad_weights",
         "sigmas": bad_blocks['sigma'].values[peak_activity_index:], },
        {"name": "sqrt_weights", "sigmas": post_peak_sigmas, },
        {"name": "duration_weights", "sigmas": duration_sigmas, },
    ]:
        pass
    """

    logger = logging.getLogger("STARE")

    if cache_path is not None and cache_path.exists():
        cache_file = cache_path / "step_3_decay_model_fits.pkl"
    else:
        cache_file = None
    if cache_file is not None and cache_file.exists() and not force:
        logger.info("  loading cached step 3 decay model fits to save time")
        successes = pickle.load(cache_file.open("rb"))
    else:
        # Fit repeatedly until we have ten successes or complete failure.
        # Only fitting decay of activity past the peak - not pre-peak rise
        successes = find_curve_fits(
            decay_model,
            pvc_tac.post_peak_timepoints(),
            pvc_tac.post_peak_activity(),
            sigmas=pvc_tac.post_peak_sigmas(method='sqrt'),
            success_limit=10, failure_limit=8192
        )
        if cache_file is not None:
            pickle.dump(successes, cache_file.open("wb"))
            logger.debug(f"WROTE {cache_file.name} (pickled rate_constants) "
                         f"to {str(cache_path)}")
    if debug_path is not None:
        pickle.dump(
            successes,
            open(debug_path / f"fits.pkl", "wb")
        )
    best_fit = select_best_fit(successes)
    lores_tac, hires_tac = interpolate_full_tac(
        pvc_tac, best_fit, decay_model
    )
    lores_tac.name = "decay model fit"
    lores_tac.sd = pvc_tac.sd

    fig = plot_detailed_tacs([lores_tac, hires_tac])
    fig.savefig(figure_path / "fits_hi_vs_lo_res.png")
    if verbose > 0 and debug_path is not None:
        pickle.dump(
            lores_tac,
            open(debug_path / f"tac_from_fitting.pkl", "wb")
        )
        pickle.dump(
            hires_tac,
            open(debug_path / f"hires_tac_from_fitting.pkl", "wb")
        )

    # Return the one best, properly weighted, interpolated TAC in original res.
    # This can be used to reduce the vascular influence on measured TACs later.
    return lores_tac, hires_tac
