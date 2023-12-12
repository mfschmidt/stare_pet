import numpy as np
import pickle
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde
from datetime import datetime

from .util import get_kde_fwhm_points
from .util import from_cache, to_cache
from .fitting_models import decay_model, find_curve_fits, func2tc_model
from .plotting import plot_bootstrap_constant, plot_bootstrap_curves
from .mp_queues import run_in_mp_queue


class CurveGenerator:
    """ An object that generates random curves

        Instantiate the object, then call object.new() any time you want
        a new curve based on the mean, sd, and distribution you used to
        initialize it.
    """

    def __init__(self, mean, sd, distribution='uniform', seed=999):
        assert (len(mean) == len(sd))
        self._mean = mean
        self._sd = sd
        self._distribution = distribution
        if self._distribution == 'normal':
            self.randomizer = np.random.default_rng(seed).normal
        else:
            # Default to uniform
            self.randomizer = np.random.default_rng(seed).random

    def new(self):
        # IDEA: More realistic, also, would be to restrict how far
        #       a given point @t can be from the previous point @t-1.
        random_noise = 2.0 * (self.randomizer(len(self._mean)) - 0.5)
        scaled_deviation = random_noise * self._sd
        return self._mean + scaled_deviation


def fit_curve_to_exponential(bootstrap_curve, vascular_tac, uniform_tac):
    """ Fit just one curve, generator-style """

    successes, failures = find_curve_fits(
        decay_model,
        vascular_tac.post_peak_timepoints(),
        bootstrap_curve,
        sigmas=vascular_tac.post_peak_sigmas(),
        success_limit=1
    )
    if len(successes) == 0:
        return None, failures
    else:
        fit = successes[0]
        post_peak_boot_curve_fit_uniform = decay_model(
            uniform_tac.post_peak_timepoints(),
            *fit['parameters']
        )
        full_boot_curve_fit_uniform = np.concatenate([
            uniform_tac.pre_peak_activity(),
            post_peak_boot_curve_fit_uniform
        ])
        if np.any(full_boot_curve_fit_uniform < 0.0):
            failures.append({
                "code": 1,
                "fit": "exp",
                "desc": "good fit, but had negatives",
                "p0": fit['p0'],
            })
            return None, failures
        else:
            fit['curve'] = full_boot_curve_fit_uniform

        return fit, failures


def fit_curves_mp(
        results, num_curves=100, num_2tc_params=3,
):
    """ Generate and fully fit n curves. """

    # Stretch fit_tac's timepoints out to be evenly spaced at 0.10 seconds.
    # This is just a template to fit upcoming curves onto.
    # NOTE: pvc_mean_tac is the best estimate of pre-peak activity so far.
    #       vascular_tac has been interpolated to high-res and back.
    #       xp & fp must have same # of samples, so align both to time_curve.
    uniform_tac = results.pvc_mean_vascular_tac.get_uniform_time_curve(
        spacing=0.10, interpolation='linear',
    )

    # Support random curve generation, on the fly, as many times as necessary,
    # based on actual TAC plus random noise
    # curve_generator = CurveGenerator(
    #     results.pvc_mean_vascular_tac.post_peak_activity(),
    #     results.pvc_mean_vascular_tac.post_peak_sd(),
    #     distribution='uniform', seed=998
    # )
    # Using one curve generator seems to have created a few very similar
    # curves. So below, there's a separate Generator, each with its own
    # seed, for each curve desired. There are many ways to do this, and
    # this is probably not the most efficient, but it does work.

    list_of_args = [(
        i,
        CurveGenerator(
            results.pvc_mean_vascular_tac.post_peak_activity(),
            results.pvc_mean_vascular_tac.post_peak_sd(),
            distribution='uniform', seed=998 + i
        ),
        results.pvc_mean_vascular_tac,
        uniform_tac,
        results.corrected_tacs,
        num_2tc_params,
        results.args.vasc_corr_pct
    ) for i in range(num_curves)]

    curve_fit_results = run_in_mp_queue(
        worker, list_of_args, results.args.num_cpus, results.logger
    )

    # Count up failures from fitting all the curves and report them.
    num_failures = np.sum([len(r['errors']) for r in curve_fit_results])
    results.logger.info(f"Fitting {len(curve_fit_results)} curves required "
                        f"{num_failures} failures.")
    return curve_fit_results


def fit_curve_to_regional_tacs(
        good_curve, vascular_tac, uniform_tac, corrected_regional_tacs,
        num_2tc_params, vasc_corr_pct,
):
    """ """

    # Create a random number generator
    rng = np.random.default_rng(999)

    lower_bounds = np.zeros((1, num_2tc_params))
    upper_bounds = np.ones((1, num_2tc_params))

    regions = corrected_regional_tacs.columns

    # Assess each fit curve as an adjuster to regional tacs
    # Reset rate constants and bounds for each curve
    rate_constants = np.zeros((len(regions), num_2tc_params))

    # Get boot curve down-sampled to original mid_times
    curve_for_original_t = pchip_interpolate(
        xi=uniform_tac.timepoints,
        yi=good_curve,
        x=vascular_tac.timepoints,
    )

    fit_errors = []
    fit_successes = []
    for j, region in enumerate(regions):
        # Adjust regional TAC by bootstrapped vascular activity
        raw_activity = corrected_regional_tacs[region].values
        vc_pct = vasc_corr_pct / 100.0
        adjustment = curve_for_original_t * vc_pct
        vasc_corr_tac = (raw_activity - adjustment) / (1.0 - vc_pct)

        # Score adjusted TAC for 2TCM fit
        successes, failures = 0, 0
        while successes < 1 and failures < 10:
            # Generate random rate constants between lower & upper_bounds
            two_sd = upper_bounds - lower_bounds
            x0 = lower_bounds + (
                    two_sd * [rng.random() for _ in range(num_2tc_params)]
            )
            # Use weights, not sigmas, for func2tc_model. It weights
            # residuals within the function, not depending on the
            # curve fitting library to do so.
            try:
                ls_result = least_squares(
                    func2tc_model,
                    x0.ravel(),
                    bounds=(lower_bounds.ravel(), upper_bounds.ravel()),
                    kwargs={
                        "uniform_mid_times": uniform_tac.timepoints,
                        "mid_times": vascular_tac.timepoints,
                        "full_boot_curve_fit_uniform": good_curve,
                        "tac": vasc_corr_tac,
                        "weights": vascular_tac.weights(),
                        "tracer": 'FDG'
                    }
                )
                if ls_result.success:
                    # This should only happen once per j, and not overwrite rcs
                    successes += 1
                    rate_constants[j, :] = np.real(ls_result.x)
                    fit_successes.append({
                        "code": 0,
                        "fit": "tac",
                        "desc": "successful fit",
                        "p0": list(x0.ravel()),
                    })
                else:
                    # This can happen repeatedly, no harm in overwriting nans
                    fit_errors.append({
                        "code": 21,
                        "fit": "tac",
                        "desc": "least squares failure to fit",
                        "p0": list(x0.ravel()),
                    })
                    failures += 1
            except ValueError:
                fit_errors.append({
                    "code": 99,
                    "fit": "tac",
                    "desc": "ValueError in least_squares",
                    "p0": list(x0.ravel()),
                })
                failures += 1
    # If any of the fits, for any of the regions is near 0 or 1,
    # invalidate the whole thing. 2TCM should not be 0 or 1
    if len(fit_successes) > 0:
        p0 = fit_successes[-1]['p0']
    else:
        p0 = []
    if np.any(np.abs(rate_constants.ravel() < 0.0001)):
        rate_constants[:, :] = np.nan
        fit_errors.append({
            "code": 22,
            "fit": "tac",
            "desc": "least squares produced a zero rate constant",
            "p0": p0,
        })
    elif np.any(np.abs(1.0 - rate_constants.ravel()) < 0.0001):
        rate_constants[:, :] = np.nan
        fit_errors.append({
            "code": 23,
            "fit": "tac",
            "desc": "least squares produced a one rate constant",
            "p0": p0,
        })

    return rate_constants, fit_errors


def find_2tc_bounds(rate_constants):
    """ """

    # We have up to 1000 sets of rate constants (dim 0 of rate_constants).
    # Use them to generate probability density functions,
    # then take the full width at half maximum (FWHM) of the PDF
    # to get the range of free parameters in STARE, and the penalty
    # in the cost function.

    # Assign STARE upper and lower bounds as either side of ksdensity FWHM
    # of the bootstrap samples.

    # Bounds for k1, k2, k3 (for constraining STARE search space)
    # dim_a is usually 6 regions
    # dim_b is usually 3 constants
    # In matlab, this would just be an 18-len vector, 6 k1s 6 k2s 6 k3s.
    dim_a, dim_b = rate_constants[0].shape
    # bounds = np.zeros((dim_a, dim_b, 2))
    peaks = np.zeros((dim_a, dim_b, 2))
    fwhm = np.zeros((dim_a, dim_b, 3, 2))

    lower_bounds = np.zeros(dim_a * dim_b)
    upper_bounds = np.zeros(dim_a * dim_b)

    for i in range(dim_a):
        for k in range(dim_b):
            param_values = rate_constants[:, i, k]
            kde = gaussian_kde(param_values)
            new_density_x = np.linspace(
                np.min(param_values), np.max(param_values), num=1000
            )
            new_density_y = kde(new_density_x)
            half_max = np.max(new_density_y) / 2
            xs_over_half_max = [val for idx, val in enumerate(new_density_x)
                                if new_density_y[idx] > half_max]

            # Store the full-width-half-max values, and the peak with its index
            lower_bounds[(k * dim_a) + i] = np.min(xs_over_half_max)
            upper_bounds[(k * dim_a) + i] = np.max(xs_over_half_max)
            # bounds[i, k] = np.array(
            #     [np.min(xs_over_half_max), np.max(xs_over_half_max), ]
            # )
            peak_idx = np.argmax(new_density_y)
            peaks[i, k] = np.array(
                [new_density_x[peak_idx], new_density_y[peak_idx], ]
            )
            fwhm[i, k], _x, _y = get_kde_fwhm_points(rate_constants[:, i, k])

    return lower_bounds, upper_bounds, peaks, fwhm


def worker(arg_tuple):
    """ Bootstrap a random starting point, fit to exponential, fit to data

        This worker can be launched in a separate process to do all three
        of the necessary steps for bootstrap anchoring. It will first
        generate a random curve as a starting vector. It will then fit
        that curve to a stack of three exponentials. It will then fit
        that result to real data. Errors at any step will be logged and
        the process will start over with new seed parameters.
    """

    # Workers get a single argument, so the caller must pack things into a tuple
    # and the worker must unpack them.
    # This order must match exactly the order where they're packed in.
    (i, curve_generator, vascular_tac, uniform_tac, corrected_regional_tacs,
        num_2tc_params, vasc_corr_pct) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting worker for curve {i} "
          f"at {worker_start.strftime('%m/%d %I:%M')}", flush=True)

    cumulative_errors = []
    failed_bootstrap_curves = []
    bootstrap_curve, rate_constants, exp_fit = None, None, None
    final_curve = None
    while final_curve is None:

        # First, generate a random curve to serve as a starting point.
        bootstrap_curve = curve_generator.new()

        # Second, fit this curve to three stacked exponentials.
        exp_fit, fit_exp_errors = fit_curve_to_exponential(
            bootstrap_curve, vascular_tac, uniform_tac
        )
        cumulative_errors = cumulative_errors + fit_exp_errors
        if exp_fit is None:
            failed_bootstrap_curves.append(bootstrap_curve)
            continue

        # Third, fit the exponentials onto actual data.
        rate_constants, fit_tac_errors = fit_curve_to_regional_tacs(
            exp_fit['curve'], vascular_tac, uniform_tac,
            corrected_regional_tacs, num_2tc_params, vasc_corr_pct
        )
        good_rate_constants = rate_constants[[
            ~np.any(np.isnan(rate_constants[_].ravel()))
            for _ in range(rate_constants.shape[0])
        ]]
        cumulative_errors = cumulative_errors + fit_tac_errors
        if len(good_rate_constants) == 0:
            failed_bootstrap_curves.append(bootstrap_curve)
            continue
        else:
            final_curve = good_rate_constants  # triggers break from while loop

    worker_end = datetime.now()
    print(f"    Finished worker for region {i} "
          f"at {worker_end.strftime('%m/%d %I:%M')} "
          f"after {worker_end - worker_start}.", flush=True)

    return {
        "i": i,
        "failed_bootstrap_curves": failed_bootstrap_curves,
        "bootstrap_curve": bootstrap_curve,
        "fit_exp": exp_fit,
        "rate_constants": rate_constants,
        "errors": cumulative_errors,
        "elapsed_time": worker_end - worker_start,
    }


def boot_anchor(results):
    """
        :param results: An object containing lots of data from the pipeline
        :return:
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Bootstrap anchoring")

    # Manual configuration
    # Originally, we aimed for 1000 boostrap seed curves,
    # which resulted in about 500-600 "good" curves,
    # which resulted in about 100-200 "fit" curves.
    # We are now targeting 500 fit curves, however many seeds that takes.
    bootstrap_iterations = results.args.bootstrap_iterations

    # Currently, only FDG is supported. Other tracers would require
    # changes to this 2TCirr. We would then have to check the tracer.
    num_2tc_params = 3
    if results.args.tracer != 'FDG':
        # Perhaps, in the future, reversible tracers can have 4 parameters.
        # raise ValueError("Tracer must be FDG. No others are yet supported.")
        print("Tracer should be FDG. No others are officially supported.")

    # Stretch fit_tac's timepoints out to be evenly spaced at 0.10 seconds.
    # This is just a template to fit upcoming curves onto.
    # NOTE: pvc_mean_tac is the best estimate of pre-peak activity so far.
    #       vascular_tac has been interpolated to high-res and back.
    #       xp & fp must have same # of samples, so align both to time_curve.
    uniform_tac = results.pvc_mean_vascular_tac.get_uniform_time_curve(
        spacing=0.10, interpolation='linear',
    )

    # Attempt to fit each bootstrap curve to the stacked exponential decay model
    # Find all workable parameters for fitting boostrap curves to regional tacs
    # If prior curves were saved to disk, load them rather than running.
    cache_file = f"sub-{results.args.subject}_step-4_bootstrap_curve_fits.pkl"
    good_curves_fits = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if good_curves_fits is None:
        # pvc_mean_tac is only used for timepoints and weights, NOT activity
        good_curves_fits = fit_curves_mp(
            results, bootstrap_iterations, num_2tc_params
        )
        to_cache(good_curves_fits, results.args.cache_path, cache_file)
    else:
        logger.info("  loaded cached step 4 curve fits to save time")

    num_failures = np.sum([len(r['errors']) for r in good_curves_fits])
    num_total_bootstraps = len(good_curves_fits) + num_failures

    if len(good_curves_fits) == 0:
        logger.error("FAILURE: No curves could be fit!!")
        rpt_sect.add_line("No curves were fit during bootstrapping!")
        rpt_sect.end()
        return results

    good_rate_constants = np.asarray(
        [c['rate_constants'] for c in good_curves_fits]
    )
    # Get upper and lower bounds for 2TCM parameters
    # This wants n x 6 region x 3 parameter
    kde_lower_bounds, kde_upper_bounds, kde_peaks, kde_fwhm = find_2tc_bounds(
        good_rate_constants,
    )
    # flattened_bounds = np.concatenate(
    #     [kde_bounds[:, :, 0].ravel(), kde_bounds[:, :, 1].ravel(), ]
    # )

    # Split discovered boot constants into three parameters
    k1 = good_rate_constants[:, :, 0]
    k2 = good_rate_constants[:, :, 1]
    k3 = good_rate_constants[:, :, 2]
    # solve all good rate constants, element-wise
    kis = np.multiply(k1, np.divide(k3, (k2 + k3)))

    ki_fwhm = np.zeros((len(results.corrected_tacs.columns), 3, 2))
    for i in range(len(results.regions)):
        ki_fwhm[i], _x, _y = get_kde_fwhm_points(kis[:, i])

    # For recursive plotting fixes:
    pickle_name = f"sub-{results.args.subject}_boot_anchor_data.pkl"
    with open(results.args.debug_path / pickle_name, "wb") as f:
        pickle.dump(
            {
                "regional_tacs": results.corrected_tacs,
                "good_rate_constants": good_rate_constants,
                "kis": kis,
                "kde_lower_bounds": kde_lower_bounds,
                "kde_upper_bounds": kde_upper_bounds,
                "kde_peaks": kde_peaks,
                "kde_fwhm": kde_fwhm,
                "ki_fwhm": ki_fwhm,
                "uniform_tac": uniform_tac,
                "curve_fits": good_curves_fits,
            },
            f
        )

    results.kde_lower_bounds = kde_lower_bounds
    results.kde_upper_bounds = kde_upper_bounds
    results.bootstrap_curves = [
        c['fit_exp']['curve'] for c in good_curves_fits
    ]
    results.bootstrap_rate_constants = good_rate_constants
    results.bootstrap_kis = kis
    results.bootstrap_ki_fwhm = ki_fwhm

    # Plot all bootstrap curves
    fig_all_curves = plot_bootstrap_curves(
        results.bootstrap_curves, results.fitted_hires_tac,
        results.pvc_mean_vascular_tac, results.args.subject,
        skip_outliers=False,
    )
    figure_name = "sub-{}_step-4_all_bootstrap_curves.png".format(
        results.args.subject,
    )
    fig_all_curves.savefig(results.args.fig_path / figure_name)

    # Plot most bootstrap curves
    fig_most_curves = plot_bootstrap_curves(
        results.bootstrap_curves, results.fitted_hires_tac,
        results.pvc_mean_vascular_tac, results.args.subject,
        skip_outliers=True,
    )
    figure_name = "sub-{}_step-4_most_bootstrap_curves.png".format(
        results.args.subject,
    )
    fig_most_curves.savefig(results.args.fig_path / figure_name)
    caption = "Bootstrap curves used for fitting the stacked exponentials"
    # A 'right_fig' image should come before everything else,
    # so text flows alongside.
    rpt_sect.add_figure(results.args.fig_path / figure_name, caption,
                        css_class='right_fig')

    rpt_sect.add_line(f"From {num_total_bootstraps} random bootstrap curves, "
                      f"{len(good_curves_fits)} good curves were found.")

    # Write plots to visualize the bootstrapping
    # Plot densities of each constant
    for i, k_const in enumerate(["k1", "k2", "k3", "ki"]):
        if k_const == "ki":
            rate_consts_for_plot = kis
        else:
            rate_consts_for_plot = results.bootstrap_rate_constants[:, :, i]
        fig = plot_bootstrap_constant(
            results.corrected_tacs.columns,
            rate_consts_for_plot,
            k_const,
            subject=results.args.subject,
            tracer=results.args.tracer,
        )
        figure_name = "sub-{}_step-4_bootstrap_{}_density_by_region.png".format(
            results.args.subject, k_const
        )
        fig.savefig(results.args.fig_path / figure_name)
        caption = f"Bootstrapped rate constant boundaries for {k_const}"
        rpt_sect.add_figure(results.args.fig_path / figure_name, caption)

    if results.args.debug:
        pickle_name = f"sub-{results.args.subject}_step-4_results.pkl"
        with open(results.args.cache_path / pickle_name, "wb") as f:
            pickle.dump(results, f)

    rpt_sect.end()
    return results
