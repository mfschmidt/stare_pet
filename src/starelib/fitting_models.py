import warnings
import numpy as np
import logging
import random
from scipy.optimize import curve_fit
from scipy.interpolate import pchip_interpolate
from scipy.signal import lfilter
from scipy.optimize import OptimizeWarning


def root_mean_square(actual, predicted):
    """ Return the RMS error between actual and predicted values.

        :param ndarray actual: Measured values
        :param ndarray predicted: Predicted or modeled values
        :returns: Root sum of squared error
    """

    return np.sqrt(np.sum((actual - predicted)**2))


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


def func2tc_model(xs, uniform_mid_times, mid_times, full_boot_curve_fit_uniform,
                  tac, weights, tracer='FDG'):
    """ Runs two tissue compartment (2TC) models

        see [Turku](http://www.turkupetcentre.net/petanalysis/model_2tcm.html)
    """

    k1, k2, k3 = xs[0], xs[1], xs[2]
    k4 = 0 if tracer == 'FDG' else xs[3]

    # Equations from Bartlett's matlab implementation of STARE,
    # which used equations from Phelps, 1979, for 2TC model
    alpha_1 = (k2 + k3 + k4 - np.sqrt((k2 + k3 + k4) ** 2 - (4 * k2 * k4))) / 2
    alpha_2 = (k2 + k3 + k4 + np.sqrt((k2 + k3 + k4) ** 2 - (4 * k2 * k4))) / 2
    impulse_response_function = (
            k1 / (alpha_2 - alpha_1) *
            (
                ((k3 + k4 - alpha_1) *
                 np.exp(-1.0 * alpha_1 * uniform_mid_times))
                +
                ((alpha_2 - k3 - k4) *
                 np.exp(-1.0 * alpha_2 * uniform_mid_times))
            )
    )
    tac_fit_uniform = (
            (uniform_mid_times[1] - uniform_mid_times[0])
            *
            lfilter(full_boot_curve_fit_uniform, 1, impulse_response_function)
    )

    # Sample the resulting total concentration in the tissue back to the
    # original mid_times sampling for comparison with TAC data
    tac_fit = pchip_interpolate(uniform_mid_times, tac_fit_uniform, mid_times)

    weighted_residual = weights * (tac - tac_fit)

    return weighted_residual


def func2tc_model_opt(xs, uniform_mid_times, mid_times,
                      full_boot_curve_fit_uniform, tac, weights, tracer='FDG'):
    """ Runs two tissue compartment (2TC) models, optimized

        see [Turku](http://www.turkupetcentre.net/petanalysis/model_2tcm.html)
    """

    k1, k2, k3 = xs[0], xs[1], xs[2]
    k4 = 0 if tracer == 'FDG' else xs[3]

    # Equations from Bartlett's matlab implementation of STARE,
    # which used equations from Phelps, 1979, for 2TC model
    k_sums = k2 + k3 + k4
    root_of_squares = np.sqrt(k_sums**2 - (4 * k2 * k4))
    alpha_1 = (k_sums - root_of_squares) / 2
    exp_alpha_1 = np.exp(-1.0 * alpha_1 * uniform_mid_times)
    alpha_1_part = (k3 + k4 - alpha_1) * exp_alpha_1
    alpha_2 = (k_sums + root_of_squares) / 2
    exp_alpha_2 = np.exp(-1.0 * alpha_2 * uniform_mid_times)
    alpha_2_part = (alpha_2 - k3 - k4) * exp_alpha_2
    impulse_response_function = (
            k1 / (alpha_2 - alpha_1) * (alpha_1_part + alpha_2_part)
    )
    tac_fit_uniform = (
            (uniform_mid_times[1] - uniform_mid_times[0])
            *
            lfilter(full_boot_curve_fit_uniform, 1, impulse_response_function)
    )

    # Sample the resulting total concentration in the tissue back to the
    # original mid_times sampling for comparison with TAC data
    tac_fit = pchip_interpolate(uniform_mid_times, tac_fit_uniform, mid_times)

    weighted_residual = weights * (tac - tac_fit)

    return weighted_residual


def randomize_stacked_exponential_parameters(n=6, init_params=None):
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


def find_curve_fits(
        f, x, y, sigmas=None, success_limit=10, failure_limit=8192
):
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
    warnings.simplefilter("error", OptimizeWarning)
    warnings.simplefilter("error", RuntimeWarning)

    # Fit repeatedly until we have ten successes or complete failure.
    successes = []
    failures = []
    while len(successes) < success_limit and len(failures) < failure_limit:
        np.random.seed = 42 * (len(failures) + 7)
        p0 = randomize_stacked_exponential_parameters(6)
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
                failures.append({
                    "code": 2,
                    "fit": "exp",
                    "desc": "fit converged, but converged to NaN",
                    "p0": p0,
                })
                logger.debug("a curve fit converged, but converged to NaN, "
                             f"failure {len(failures)} for this model, "
                             f"{len(successes)} successes.")
            elif np.isinf(retval[0]).any() or np.isinf(retval[1]).any():
                failures.append({
                    "code": 3,
                    "fit": "exp",
                    "desc": "fit converged, but converged to Infinity",
                    "p0": p0,
                })
                logger.debug("a curve fit converged, but converged to infinity,"
                             f" failure {len(failures)} for this model, "
                             f"{len(successes)} successes.")
            elif weighted_error > 10.0:
                failures.append({
                    "code": 4,
                    "fit": "exp",
                    "desc": "fit converged, but weighted error of "
                            f"{weighted_error:0.2f} (rms {rms:0.2f}) is high.",
                    "p0": p0,
                })
                logger.debug("a curve fit converged, but weighted error of "
                             f"{weighted_error:0.2f} (rms {rms:0.2f}) is high, "
                             f"failure {len(failures)} for this model, "
                             f"{len(successes)} successes.")
            else:
                successes.append({
                    "parameters": fit_parameters,
                    "covariance": retval[1],
                    "residuals": residuals,
                    "fit": fit,
                    "rms": rms,
                    "wrms": weighted_error,
                    "p0": p0,
                })
        except RuntimeError as e:
            failures.append({
                "code": 11, "fit": "exp", "desc": e.args[0], "p0": p0,
            })
            logger.debug("a curve fit failed to converge, "
                         f"failure {len(failures)} for this model, "
                         f"{len(successes)} successes.")
            # logger.debug("x: [" + ",".join([f"{_}:0.1f" for _ in x]) + "]")
            # logger.debug("y: [" + ",".join([f"{_}:0.1f" for _ in y]) + "]")
        except OptimizeWarning as e:
            failures.append({
                "code": 12, "fit": "exp", "desc": e.args[0], "p0": p0,
            })
            logger.debug("a curve fit raised an optimize warning, "
                         f"failure {len(failures)} for this model, "
                         f"{len(successes)} successes.")
            # logger.debug("x: [" + ",".join([f"{_}:0.1f" for _ in x]) + "]")
            # logger.debug("y: [" + ",".join([f"{_}:0.1f" for _ in y]) + "]")
        except RuntimeWarning:
            failures.append({
                "code": 13, "fit": "exp",
                "desc": "overflow encountered, probably", "p0": p0,
            })
            logger.debug("a curve fit raised a runtime warning, "
                         f"failure {len(failures)} for this model, "
                         f"{len(successes)} successes.")
            # logger.debug("x: [" + ",".join([f"{_}:0.1f" for _ in x]) + "]")
            # logger.debug("y: [" + ",".join([f"{_}:0.1f" for _ in y]) + "]")
    warnings.resetwarnings()
    warnings.filterwarnings("ignore")
    logger.info(f"Curve fitting summary: Seeking {success_limit} fits, "
                f"and tolerating {failure_limit} failures before quitting, "
                f"{len(failures)} failed to converge and "
                f"{len(successes)} converged.")

    # In the case of success, this is a list of dicts.
    # In the case of failure, it is an empty list.
    return successes, failures


def source_to_target_tissue_model(
        src_ks, tgt_ks, source_tac, uniform_tac
):
    """ The primary model for rotating source-to-target parameter estimates

        :param np.array src_ks: parameter estimates, in a [ks] array
        :param np.array tgt_ks: parameter estimates, in a [regions x ks] array
        :param np.array source_tac: Corrected Time Activity Curve for source
        :param np.array uniform_tac: Time Activity Curve at uniform hi-res

        st2model needs k1, k2, k3 from each region (6 regions -> 18 ks)
        one of six is source, the other five are target
    """

    # Note that numpy matrix operators * / are element-wise,
    # which differs from the matlab operators, which require .* or ./

    # Extract parameter estimates for the source region, as scalars
    # src_ks is a 1D array, with 4 ks, including ki
    k_1_s, k_2_s, k_3_s = src_ks[0], src_ks[1], src_ks[2]

    # Target parameters, a 2D array with each k for each target
    k_1_t, k_2_t, k_3_t = tgt_ks[0, :], tgt_ks[1, :], tgt_ks[2, :]

    # Source p, q, r
    # These can be scalars because numpy broadcasts to fill arrays
    r_s = k_2_s + k_3_s
    p_s = k_2_s / r_s
    q_s = k_3_s / r_s

    # Target p, q, r
    r_t = k_2_t + k_3_t
    p_t = k_2_t / r_t
    q_t = k_3_t / r_t

    # Fill out a system of equations
    alpha = (q_t * r_t) / (p_t + q_t) + r_s - (q_s * r_s) / (p_s + q_s) - r_t
    beta = (q_t * r_t * r_s) / (p_t + q_t) - (q_s * r_t * r_s) / (p_s + q_s)
    gamma = (q_s * r_s) / (p_s + q_s) + r_t
    omega = (q_s * r_t * r_s) / (p_s + q_s)

    nu = (-gamma + np.sqrt(gamma ** 2 - 4 * omega)) / 2
    epsilon = (-gamma - np.sqrt(gamma ** 2 - 4 * omega)) / 2

    big_el = alpha - (beta + alpha * epsilon) / (epsilon - nu)
    big_em = (beta + alpha * epsilon) / (epsilon - nu)

    # Fit to uniformly sampled data
    source_tac_uniform = pchip_interpolate(
        source_tac.timepoints, source_tac.activity, uniform_tac.timepoints
    )

    num_targets = len(k_1_t)
    impulse_response_func = np.ones((len(source_tac.activity), num_targets))
    target_tac_fits = np.ones((len(source_tac.activity), num_targets))
    for i in range(num_targets):
        zeta = (
            big_el[i] * np.exp(nu[i] * uniform_tac.timepoints)
            +
            big_em[i] * np.exp(epsilon[i] * uniform_tac.timepoints)
        )
        target_tac_fit_uniform = (
            (
                (k_1_t[i] / k_1_s) *
                uniform_tac.uniform_time_delta *
                lfilter(source_tac_uniform, 1, zeta)
            )
            +
            (
                (k_1_t[i] / k_1_s) *
                source_tac_uniform
            )
        )
        # TODO: check for infinite values in zeta? There's a value error
        #       running NHFDG047, but only once. Maybe a
        #       try/catch ValueError block with a printout of all the vars?
        impulse_response_func[:, i] = pchip_interpolate(
            xi=uniform_tac.timepoints, yi=zeta,
            x=source_tac.timepoints
        )
        target_tac_fits[:, i] = pchip_interpolate(
            xi=uniform_tac.timepoints, yi=target_tac_fit_uniform,
            x=source_tac.timepoints
        )

    return impulse_response_func, target_tac_fits


def solve_stttm(
        x, i, source_tac, uniform_tac,
        region_weights, tac_weights, target_tacs, ki_peaks
):
    """ Calculate s2ttm from (regions * 3)-length array of parameters
    """

    # Extract parameter estimates for the source region, these end up scalars
    # x is a 1D [regions * ks] array, with 3 ks
    num_regions = int(len(x) / 3)
    k_1_s = x[i]
    k_2_s = x[i + num_regions]
    k_3_s = x[i + num_regions + num_regions]
    k_i_s = k_1_s * (k_3_s / (k_2_s + k_3_s))
    source_parameters = np.array([k_1_s, k_2_s, k_3_s, k_i_s])

    # Target parameters, strip out the source region's ks, keep the rest
    xt = np.delete(
        x, (i, i + num_regions, i + num_regions + num_regions)
    )
    k_1_t = xt[0:num_regions - 1]
    k_2_t = xt[num_regions - 1:(num_regions - 1) * 2]
    k_3_t = xt[(num_regions - 1) * 2:]
    k_i_t = k_1_t * (k_3_t / (k_2_t + k_3_t))
    target_parameters = np.array([k_1_t, k_2_t, k_3_t, k_i_t])

    k_i = np.insert(k_i_t, i, k_i_s)
    k_i_penalty = np.sum(abs(k_i - ki_peaks))

    impulse_response_func, target_tac_fits = source_to_target_tissue_model(
        source_parameters, target_parameters, source_tac, uniform_tac
    )
    cost = np.sum(
        region_weights *
        np.sum(tac_weights[:, np.newaxis] * (target_tacs - target_tac_fits)**2)
    ) + k_i_penalty

    return impulse_response_func, target_tac_fits, k_i, k_i_penalty, cost


class TwoTissueCompartmentModel:
    """ Save parameters for a 2TC model """

    def __init__(self, k1, k2, k3, desc=""):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.ki = self.k1 * (self.k3 / (self.k2 + self.k3))
        self.desc = desc

    def __str__(self):
        return (
            "'{desc}': ki = k1 * (k3 / (k2 + k3)) = "
            "{k1:0.2f} * ({k3:0.2f} / ({k2:0.2f} + {k3:0.2f})) = "
            "{ki:0.2f}".format(
                desc=self.desc, ki=self.ki, k1=self.k1, k2=self.k2, k3=self.k3
            )
        )
