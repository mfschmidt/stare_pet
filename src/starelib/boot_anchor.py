import numpy as np
import pickle
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde

from .timeactivitycurve import TimeActivityCurve
from .util import get_kde_fwhm_points
from .util import from_cache, to_cache
from .fitting_models import decay_model, find_curve_fits, func2tc_model
from .plotting import plot_bootstrap_constant, plot_bootstrap_curves


def gen_bootstrap_curves(
        sample_mean, sample_sd, n=1000,
        distribution='uniform', seed=999
):
    """ Generate n curves within 1SD of sample for bootstrapping """

    # Ensure the data we're given make sense
    assert(len(sample_mean) == len(sample_sd))

    # Seed the random number generator
    rng = np.random.default_rng(seed)
    if distribution == 'uniform':
        randomizer = rng.random
    elif distribution == 'normal':
        randomizer = rng.normal
    else:
        # Default to uniform
        randomizer = rng.random

    # Generate a thousand curves, based on actual TAC plus random noise
    boot_curves = []
    for i in range(n):
        # TODO: This is a uniform distribution, but perhaps a normal
        #       distribution would be better weighted?
        # TODO: More realistic, also, would be to restrict how far
        #       a given point @t can be from the previous point @t-1.
        random_noise = 2.0 * (randomizer(len(sample_mean)) - 0.5)
        scaled_deviation = random_noise * sample_sd
        generated_curve = sample_mean + scaled_deviation
        boot_curves.append(generated_curve)

    return np.array(boot_curves)


def make_uniform_time_curve(pvc_mean_tac, spacing=0.10):
    """ Evenly space timepoints from uneven sampling

        Betsy's matlab version only stored a peak_index value
        in the fit vascular tac, not the pvc-corrected vascular tac. So we
        needed two tacs to piece together a higher resolution interpolation.
        Because we use a TimeActivityCurve object where every TAC has both
        activity and timepoints and the ability to calculate its own peak,
        only one TAC is necessary here.
    """

    # Interpolate a higher-resolution x-axis time data from TAC data
    # In matlab test, results in a 551-length vector from 0.0 to 55.0
    # from 11 pre-peak 0.0 to 1.0 and 540 post-peak 1.1 to 55.0
    pre_peak_time_uniform = np.arange(
        start=0.0,
        stop=round(pvc_mean_tac.post_peak_timepoints()[0], 1),
        step=spacing,
    )
    post_peak_time_uniform = np.arange(
        start=round(pvc_mean_tac.post_peak_timepoints()[0], 1),
        stop=pvc_mean_tac.timepoints[-1] + spacing,
        step=spacing,
    )
    boot_curve_time_uniform = np.concatenate([
        pre_peak_time_uniform, post_peak_time_uniform,
    ])

    # Interpolate higher-resolution y-axis activity from TAC data
    # Interpolate values from sparse to hi-res, then clip low end to 0.0.
    # DIFF: Numpy's interpolator flattens at the end; matlab's shoots higher.
    # NOTE: pvc_mean_tac is the best estimate of pre-peak activity so far.
    #       vascular_tac has been interpolated to high-res and back.
    #       xp & fp must have same # of samples, so align both to time_curve.
    pre_peak_vasctac_uniform = np.interp(
        pre_peak_time_uniform,
        pvc_mean_tac.pre_peak_timepoints(),
        pvc_mean_tac.activity[:pvc_mean_tac.peak_index],
    )
    # Clean up any errant points
    pre_peak_vasctac_uniform[(
        (pre_peak_vasctac_uniform < 0) | np.isnan(pre_peak_vasctac_uniform)
    )] = 0.0
    num_post_peak = len(boot_curve_time_uniform) - len(pre_peak_vasctac_uniform)
    # We model only the pre-peak data, leave post-peak for later
    post_peak_vasctac_uniform = np.array([np.nan, ] * num_post_peak)
    boot_curve_activity_uniform = np.concatenate([
        pre_peak_vasctac_uniform, post_peak_vasctac_uniform,
    ])

    # Ensure the fit is uniformly sampled.
    deltas = []
    last_t = 0.0
    for j, t in enumerate(boot_curve_time_uniform):
        if j > 0:
            deltas.append(t - last_t)
        last_t = t
    if (np.max(np.array(deltas)) - np.min(np.array(deltas))) > 0.00001:
        raise ValueError("Impossibly, the predetermined times are nonuniform!")

    return TimeActivityCurve(
        activity=np.array(boot_curve_activity_uniform),
        timepoints=np.array(boot_curve_time_uniform),
        source="uniform_interpolator",
        name="uniform_time_only",
    )


def fit_curves(curves, vascular_tac, uniform_tac, verbose=True):
    """ Fit each curve to triple-stack exponential. """

    good_curves = []
    counts = {
        "fit": 0,
        "skipped_for_negatives": 0,
    }
    for i, curve in enumerate(curves):

        # TODO: Put each curve in a queue, let multiple Processes handle them.

        # Let the function handle successes and failures and fitting.
        # It will return a list of 'success_limit' fits, so we [0] the only one.
        fits = find_curve_fits(
            decay_model, vascular_tac.post_peak_timepoints(), curve,
            sigmas=vascular_tac.post_peak_sigmas(),
            success_limit=1, failure_limit=256
        )
        if len(fits) > 0:
            counts["fit"] += 1
            first_fit = fits[0]

            # Combine interpolated pre-peak blood data with post-peak fit
            post_peak_boot_curve_fit_uniform = decay_model(
                uniform_tac.post_peak_timepoints(),
                *first_fit['parameters']
            )
            full_boot_curve_fit_uniform = np.concatenate([
                uniform_tac.pre_peak_activity(),
                post_peak_boot_curve_fit_uniform
            ])

            # Store fits that seem OK
            if np.any(full_boot_curve_fit_uniform < 0.0):
                counts["skipped_for_negatives"] += 1
                if verbose:
                    print(f"SKIPPING: Solution {i} contains negatives.")
            else:
                # save y; x is the same boot_curve_time_uniform for every curve.
                good_curves.append(full_boot_curve_fit_uniform)

    return good_curves


def fit_curves_to_regional_tacs(
        good_curves, vascular_tac, uniform_tac, corrected_regional_tacs,
        num_2tc_params, vasc_corr_pct, verbose=True,
):
    """ """

    # Create a random number generator
    rng = np.random.default_rng(999)

    lower_bounds = np.zeros((1, num_2tc_params))
    upper_bounds = np.ones((1, num_2tc_params))

    regions = corrected_regional_tacs.columns
    bootstrap_rate_constants = np.zeros(
        (len(good_curves), len(regions), num_2tc_params)
    )

    # Assess each fit curve as an adjuster to regional tacs
    num_good_rate_constants = 0
    for i, curve in enumerate(good_curves):

        # TODO: Put each curve in a queue, let multiple Processes handle them.

        # Reset rate constants and bounds for each curve
        rate_constants = np.zeros((len(regions), num_2tc_params))

        # Get boot curve down-sampled to original mid_times
        curve_for_original_t = pchip_interpolate(
            xi=uniform_tac.timepoints,
            yi=curve,
            x=vascular_tac.timepoints,
        )

        for j, region in enumerate(regions):
            # Adjust regional TAC by bootstrapped vascular activity
            raw_activity = corrected_regional_tacs[region].values
            vc_pct = vasc_corr_pct / 100.0
            adjustment = curve_for_original_t * vc_pct
            vasc_corr_tac = (raw_activity - adjustment) / (1.0 - vc_pct)

            # Score adjusted TAC for 2TCM fit
            successes, failures = 0, 0
            while successes < 1 and failures < 10:
                # Generate random rate constants between lower_ & upper_bounds
                two_sd = upper_bounds - lower_bounds
                x0 = lower_bounds + two_sd * rng.random()
                # Use weights, not sigmas, for func2tc_model. It weights
                # residuals within the function, not depending on the
                # curve fitting library to do so.
                ls_result = least_squares(
                    func2tc_model,
                    x0.ravel(),
                    bounds=(lower_bounds.ravel(), upper_bounds.ravel()),
                    kwargs={
                        "uniform_mid_times": uniform_tac.timepoints,
                        "mid_times": vascular_tac.timepoints,
                        "full_boot_curve_fit_uniform": curve,
                        "tac": vasc_corr_tac,
                        "weights": vascular_tac.weights(),
                        "tracer": 'FDG'
                    }
                )
                if ls_result.success:
                    # This should only happen once per j, and not overwrite rcs
                    successes += 1
                    rate_constants[j, :] = np.real(ls_result.x)
                else:
                    # This can happen repeatedly, no harm in overwriting nans
                    failures += 1

        # If any of the fits, for any of the regions is near 0 or 1,
        # invalidate the whole thing. 2TCM should not be 0 or 1
        if np.any(np.abs(rate_constants.ravel() < 0.0001)):
            rate_constants[:, :] = np.nan
            status = "cancelled due to a zero rate constant"
        elif np.any(np.abs(1.0 - rate_constants.ravel()) < 0.0001):
            rate_constants[:, :] = np.nan
            status = "cancelled due to a one rate constant"
        else:
            status = "good"
            num_good_rate_constants += 1

        if verbose:
            print(f"{i}/{len(good_curves)}. {status}")

        # Update the full collection of rate_constants
        bootstrap_rate_constants[i] = rate_constants

    print(f"Found {num_good_rate_constants} rate constants "
          f"from {len(good_curves)} curves.")

    return bootstrap_rate_constants


def find_2tc_bounds(rate_constants):
    """ """

    # We have up to 1000 sets of rate constants.
    # Use them to generate probability density functions,
    # then take the full width at half maximum (FWHM) of the PDF
    # to get the range of free parameters in STARE, and the penalty
    # in the cost function.

    # Assign STARE upper and lower bounds as either side of ksdensity FWHM
    # of the bootstrap samples.

    # Bounds for k1, k2, k3 (for constraining STARE search space
    dim_a, dim_b = rate_constants[0].shape
    bounds = np.zeros((dim_a, dim_b, 2))
    peaks = np.zeros((dim_a, dim_b, 2))
    fwhm = np.zeros((dim_a, dim_b, 3, 2))

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
            bounds[i, k] = np.array(
                [np.min(xs_over_half_max), np.max(xs_over_half_max), ]
            )
            peak_idx = np.argmax(new_density_y)
            peaks[i, k] = np.array(
                [new_density_x[peak_idx], new_density_y[peak_idx], ]
            )
            fwhm[i, k], _x, _y = get_kde_fwhm_points(rate_constants[:, i, k])

    return bounds, peaks, fwhm


def boot_anchor(results):
    """
        :param results: An object containing lots of data from the pipeline
        :return:
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Bootstrap anchoring")

    # Manual configuration
    bootstrap_iterations = 1000

    # Currently, only FDG is supported. Other tracers would require
    # changes to this 2TCirr. We would then have to check the tracer.
    num_2tc_params = 3
    if results.args.tracer != 'FDG':
        # Perhaps, in the future, reversible tracers can have 4 parameters.
        # raise ValueError("Tracer must be FDG. No others are yet supported.")
        print("Tracer should be FDG. No others are officially supported.")

    # Generate a thousand curves, based on actual TAC plus random noise
    bootstrap_curves = gen_bootstrap_curves(
        results.pvc_mean_vascular_tac.post_peak_activity(),
        results.pvc_mean_vascular_tac.post_peak_sd(),
        n=bootstrap_iterations, distribution='uniform', seed=999
    )
    if results.args.debug_path is not None and results.args.debug_path.exists():
        pickle.dump(
            bootstrap_curves,
            open(results.args.debug_path / "boot_curve_permutations.pkl", "wb")
        )

    # Stretch fit_tac's timepoints out to be evenly spaced at 0.10 seconds.
    uniform_tac = make_uniform_time_curve(
        results.pvc_mean_vascular_tac, spacing=0.10
    )

    # Attempt to fit each bootstrap curve to the stacked exponential decay model
    # Find all workable parameters for fitting boostrap curves to regional tacs
    # If prior curves were saved to disk, load them rather than running.
    cache_file = "step_4_good_curves.pkl"
    good_curves = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if good_curves is None:
        # pvc_mean_tac is only used for timepoints and weights, NOT activity
        good_curves = fit_curves(
            bootstrap_curves, results.pvc_mean_vascular_tac, uniform_tac,
            verbose=results.args.verbose
        )
        to_cache(good_curves, results.args.cache_path, cache_file)
    else:
        logger.info("  loaded cached step 4a curve fits to save time")

    # Attempt to extract reasonable rate constants from each good curve
    # If prior rate constants were cached to disk, load them rather than running
    cache_file = "step_4_rate_constants.pkl"
    good_rate_constants = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if good_rate_constants is None:
        rate_constants = fit_curves_to_regional_tacs(
            good_curves, results.pvc_mean_vascular_tac, uniform_tac,
            results.corrected_tacs, num_2tc_params, results.args.vasc_corr_pct,
            verbose=results.args.verbose
        )
        good_rate_constants = rate_constants[[
            ~np.any(np.isnan(rate_constants[_].ravel()))
            for _ in range(rate_constants.shape[0])
        ]]
        to_cache(good_rate_constants, results.args.cache_path, cache_file)
    else:
        logger.info("  loading cached step 4b rate constants to save time")

    if len(good_rate_constants) == 0:
        logger.error("FAILURE: No curves could be fit!!")
        rpt_sect.add_line("No curves were fit during bootstrapping!")
        rpt_sect.end()
        return results

    # Get upper and lower bounds for 2TCM parameters
    kde_bounds, kde_peaks, kde_fwhm = find_2tc_bounds(
        good_rate_constants,
    )
    flattened_bounds = np.concatenate(
        [kde_bounds[:, :, 0].ravel(), kde_bounds[:, :, 1].ravel(), ]
    )

    # Split discovered boot constants into three parameters
    k1 = good_rate_constants[:, :, 0]
    k2 = good_rate_constants[:, :, 1]
    k3 = good_rate_constants[:, :, 2]
    # solve all good rate constants, element-wise
    kis = np.multiply(k1, np.divide(k3, (k2 + k3)))

    ki_fwhm = np.zeros((len(results.corrected_tacs.columns), 3, 2))
    for i in range(len(results.corrected_tacs.columns)):
        ki_fwhm[i], _x, _y = get_kde_fwhm_points(kis[:, i])

    # For recursive plotting fixes:
    pickle.dump(
        {
            "regional_tacs": results.corrected_tacs,
            "good_rate_constants": good_rate_constants,
            "kis": kis,
            "bounds": flattened_bounds,
            "kde_bounds": kde_bounds,
            "kde_peaks": kde_peaks,
            "kde_fwhm": kde_fwhm,
            "ki_fwhm": ki_fwhm,
            "uniform_tac": uniform_tac,
            "uniform_curves": good_curves,
        },
        open(results.args.debug_path / "boot_anchor_data.pkl", "wb")
    )

    results.bootstrap_curves = good_curves
    results.bootstrap_rate_constants = good_rate_constants
    results.bootstrap_ki_fwhm = ki_fwhm

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
        fig.savefig(
            results.args.fig_path /
            f"step_4_bootstrap_{k_const}_density_by_region.png"
        )

    # Plot all bootstrap curves
    fig = plot_bootstrap_curves(
        results.bootstrap_curves, results.fitted_hires_tac,
        results.pvc_mean_vascular_tac, results.args.subject
    )
    fig.savefig(results.args.fig_path / f"step_4_bootstrap_curves.png")

    rpt_sect.end()
    return results
