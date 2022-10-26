import numpy as np
import logging
import random
import pickle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, PchipInterpolator

from .util import characterize_mid_times
from .timeactivitycurve import TimeActivityCurve
from .plotting import plot_detailed_tacs


def root_mean_square(actual, predicted):
    """ Return the RMS error between actual and predicted values.

        :param ndarray actual: Measured values
        :param ndarray predicted: Predicted or modeled values
        :returns: Root sum of squared error
    """

    return np.sqrt(np.sum((actual - predicted)**2))


def find_curve_fits(f, x, y,
                    sigmas=None, success_limit=10, failure_limit=8192):
    """ Find several options for fitting data to our model.

        :param function f: the curve to fit, returns a y for any x
        :param ndarray x: actual x values to fit
        :param ndarray y: actual y values to fit
        :param ndarray sigmas: Optionally, provide uncertainty as SD
        :param success_limit: How many fits should be found and returned
        :param failure_limit: How many failures before we give up
        :return: a list of dicts, each representing one fit curve
    """

    logger = logging.getLogger("STARE")

    # Fit repeatedly until we have ten successes or complete failure.
    successes = []
    num_successes, num_failures, num_attempts = 0, 0, 0
    while len(successes) < success_limit and num_failures < failure_limit:
        np.random.seed = 42 * (num_failures + 7)
        p0 = get_initial_parameters(6)
        num_attempts += 1
        try:
            # Fit the data to the model, returning parameters and covariance.
            retval = curve_fit(
                f, x, y, p0=p0, method='lm', maxfev=4096,
                sigma=sigmas, absolute_sigma=(sigmas is not None),
                full_output=True
            )
            fit_parameters = retval[0]
            fit = f(x, *fit_parameters)
            residuals = (y - fit)
            rms = root_mean_square(y, fit)
            if sigmas is None:
                weighted_error = np.sum(residuals**2)
            else:
                weighted_error = np.sum((1 / np.sqrt(sigmas)) * residuals**2)

            # There are many ways for these fits to be shit.
            # They can throw an exception, usually due to overflows, which
            # indicate pretty far-out values we don't want. Or they can have
            # infinite variance or NaN values. Those actually converge,
            # but are still worthless. Count all of them as failures.
            # So far, normal sums of squared errors would be greater
            # than zero, and less than 1. Anything over 10 is truly missing
            # the curve and can be dismissed as failure.
            if np.isnan(retval[0]).any() or np.isnan(retval[1]).any():
                num_failures += 1
                logger.info("a curve fit converged, but converged to NaN, "
                            f"failure {num_failures} for this model, "
                            f"{len(successes)} successes.")
            elif np.isinf(retval[0]).any() or np.isinf(retval[1]).any():
                num_failures += 1
                logger.info("a curve fit converged, but converged to infinity, "
                            f"failure {num_failures} for this model, "
                            f"{len(successes)} successes.")
            elif weighted_error > 10.0:
                num_failures += 1
                logger.info("a curve fit converged, but weighted error of "
                            f"{weighted_error:0.2f} (rms {rms:0.2f}) is high, "
                            f"failure {num_failures} for this model, "
                            f"{len(successes)} successes.")
            else:
                num_successes += 1
                successes.append({
                    "parameters": fit_parameters,
                    "covariance": retval[1],
                    "residuals": residuals,
                    "fit": fit,
                    "rms": rms,
                    "wrms": weighted_error,
                })
        except RuntimeError:
            num_failures += 1
            logger.info("a curve fit failed to converge, "
                        f"failure {num_failures} for this model, "
                        f"{len(successes)} successes.")
            # logger.debug("x: [" + ",".join([f"{_}:0.1f" for _ in x]) + "]")
            # logger.debug("y: [" + ",".join([f"{_}:0.1f" for _ in y]) + "]")

    logger.info(f"Of {num_attempts} attempts, "
                f"{num_successes} converged and "
                f"{num_failures} failed to converge.")

    return successes


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


def get_initial_parameters(n=6, init_params=None):
    """ Use init_params, if provided, and randomize other parameters.

        :param int n: how many parameters to return
        :param list init_params: Specified parameters to use
        :return: list of parameters
    """

    # What range should coefficients be randomized within?
    coef_range = (0.0, 128.0)
    exp_range = (-2.0, 10.0)

    # Coefficients should be positive, and can be large.
    random_coefficients = [random.uniform(*coef_range) for _ in range(n)]
    # Exponents may be negative, but should start small.
    random_exponents = [random.uniform(*exp_range) for _ in range(n)]

    parameters = []
    for i in range(n):
        if init_params is not None and i < len(init_params):
            parameters.append(init_params[i])
        else:
            if i % 2 == 0:
                parameters.append(random_coefficients[i])
            else:
                parameters.append(random_exponents[i])

    return parameters


def interpolate_full_tac(actual_tac, best_fit, model, tac_name="high res decay model"):
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
        timepoints=all_uniform_t,
        activity=np.concatenate([
            pre_peak_uniform_fit, post_peak_uniform_fit,
        ]),
        source="interpolation",
        name=tac_name,
    )

    # Resample full high-res fit back down to original mid_times.
    pchip_interpolator = PchipInterpolator(
        high_res_tac.timepoints, high_res_tac.activity,
    )
    original_mid_times = np.asarray(actual_tac.timepoints)

    orig_res_tac = TimeActivityCurve(
        timepoints=original_mid_times,
        activity=pchip_interpolator(original_mid_times),
        source="decay_model",
        name="original res decay model",
    )

    return orig_res_tac, high_res_tac


# Define a model to use for fitting and interpolation of decay.
# We need to map from x (timepoints, stored in post_peak_mid_times)
# to y (activity, stored in post_peak_activity),
# where x and y are each a vector (numpy.ndarray) of about 21 floats.
# From the TAC's peak activity onward, we anticipate exponential decay,
# a steep then gradual decline asymptoting to near zero.
# This curve can be generally described by y=e**(-x)
def decay_model(x, c1, lambda1, c2, lambda2, c3, lambda3):
    """ Use e**(-x) as an exponential decay motif, but stack three of them,
        each with a different coefficient (magnitude shift)
        and lambda (steepener)
    """
    y = (c1 * np.exp(-1.0 * lambda1 * x)) + \
        (c2 * np.exp(-1.0 * lambda2 * x)) + \
        (c3 * np.exp(-1.0 * lambda3 * x))
    return y


def decay_model_1(xs, c1, lambda1):
    return c1 * np.exp(-1.0 * lambda1 * xs)


def decay_model_2(xs, c1, lambda1, c2, lambda2):
    return (c1 * np.exp(-1.0 * lambda1 * xs)) + \
           (c2 * np.exp(-1.0 * lambda2 * xs))


def fit_vascular_mean_tac(
        vascular_tac, missing_mid_times, figure_path,
        debug_path=None, verbose=1
):
    """ Fit vascular mean TAC

        :param TimeActivityCurve vascular_tac: The best TAC thus far (from PVC)
        :param np.ndarray missing_mid_times: Mid-times missing from vascular_tac
        :param Path figure_path: The path for saving out figures
        :param Path debug_path: The path for saving out debug information
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

    # Calculate timing blocks
    # Weights are calculated on the duration of each block of time in
    # the TAC, but the duration must be calculated on ALL blocks, even
    # if ignored, because the duration is unknown if the block before
    # or after was removed.
    time_blocks = characterize_mid_times(
        vascular_tac.timepoints, missing_mid_times=missing_mid_times
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
    bad_blocks = characterize_mid_times(vascular_tac.timepoints)  # no missing mid_times
    bad_blocks['sigma'] = np.real(1 / np.sqrt(np.sqrt(bad_blocks['duration'])))
    duration_sigmas = time_blocks[
        time_blocks['used']
    ]['full_sigma'].values[peak_activity_index:]

    tacs = {"vascular": vascular_tac, }
    hires_tacs = {}
    # Fit the data to the decay_model (here we fit three variants of the data)
    for attempt in [
        {"name": "raw", "sigmas": None, },
        {"name": "bad_weights", "sigmas": bad_blocks['sigma'].values[peak_activity_index:], },
        {"name": "sqrt_weights", "sigmas": post_peak_sigmas, },
        {"name": "duration_weights", "sigmas": duration_sigmas, },
    ]:
        # Fit repeatedly until we have ten successes or complete failure.
        successes = find_curve_fits(
            decay_model, post_peak_mid_times, post_peak_activity,
            sigmas=attempt['sigmas']
        )
        if debug_path is not None:
            pickle.dump(
                successes,
                open(debug_path / f"fits_from_{attempt['name']}.pkl", "wb")
            )
        best_fit = select_best_fit(successes)
        lores_tac, hires_tac = interpolate_full_tac(
            vascular_tac, best_fit, decay_model
        )
        lores_tac.name = f"{attempt['name']} model fit"
        tacs[attempt['name']] = lores_tac
        hires_tac.name = f"{attempt['name']} model fit"
        hires_tacs[attempt['name']] = hires_tac

    fig = plot_detailed_tacs([v for k, v in tacs.items()])
    fig.savefig(figure_path / "compare_model_fits.png")
    if verbose > 0 and debug_path is not None:
        pickle.dump(
            tacs,
            open(debug_path / f"tacs_dict_from_fitting.pkl", "wb")
        )
        pickle.dump(
            hires_tacs,
            open(debug_path / f"hires_tacs_dict_from_fitting.pkl", "wb")
        )

    # Return the one best, properly weighted, interpolated TAC in original res.
    # This can be used to reduce the vascular influence on measured TACs later.
    return tacs["duration_weights"]
