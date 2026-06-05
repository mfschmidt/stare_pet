import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from datetime import datetime
import pickle

from .fitting_models import solve_stttm, TwoTissueCompartmentModel
from .timeactivitycurve import TimeActivityCurve
from .plotting import plot_all_stare_tac_fits
from .util import from_cache, to_cache
from .mp_queues import run_in_mp_queue


def cost_function(
        x, src_idx, source_tac, uniform_tac,
        region_weights, weights, target_tacs, boot_ki_ks_density_peak
):
    """ Run the s2ttm model and return a measure for poorness of fit

        :param np.array x: parameter estimates, in a 1D [regions * ks] array
        :param int src_idx: which column of x0 is source, all else target
        :param source_tac: Corrected Time Activity Curve for source
        :param uniform_tac: Time Activity Curve at uniform hi-res
        :param np.array region_weights: Weights to modify each region's cost
        :param np.array weights: Weights relative to each time point's duration
        :param pd.DataFrame target_tacs: Corrected TACs for each region
        :param np.array boot_ki_ks_density_peak: Previously calculated kis
    """

    try:
        irf, target_tac_fits, ki, ki_penalty, cost = solve_stttm(
            x, src_idx, source_tac, uniform_tac, region_weights, weights,
            target_tacs, boot_ki_ks_density_peak,
        )
    except ValueError:
        # In the case of an irrecoverable error while solving,
        # just return a cost so high that this non-solution can't possibly be
        # selected as legitimate.
        # Reasonable solutions have costs below 0.010
        return 999.0

    return cost


def worker(arg_tuple):
    """ the wrapper function designed to be run in parallel pools """

    # Provide a place for the callback to store intermediate annealing data
    global_annealer_data = []

    def annealer_callback(x, cur_cost, context):
        """ Monitor annealer progress and decide when to quit.

            :param x: best parameters so far
            :param cur_cost: cost of result from the best parameters so far
            :param context: why the callback was called
        """

        if len(global_annealer_data) > 0:
            delta = global_annealer_data[-1].get('cur_cost', 0.000) - cur_cost
        else:
            delta = cur_cost
        data_dict = {
            "cur_cost": cur_cost,
            "delta": delta,
            "context": context,
            "t": datetime.now(),
        }
        for _, p in enumerate(x):
            data_dict[f"x{_ + 1:00d}"] = p
        global_annealer_data.append(data_dict)

        # Return false to continue, true to stop with adequate results
        # TODO: Determine a reasonable stopping condition
        return False

    # Workers get a single argument, so we must pack things into a tuple.
    # This order must match exactly the order where they're packed in.
    (i, sa_bounds, src_tac, hires_tac, region_weights, w, target_tacs,
     ki_peaks, max_iter, init_temp, no_local_search, x0) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting worker for region {i} "
          f"at {worker_start.strftime('%m/%d %I:%M')}", flush=True)

    # Scipy's dual_annealing function requires that the 'cost_function'
    # function must accept x as its first argument as a 1D array.
    # The bounds must be a (lo, hi) pair for each element of that array.
    optimization_result = dual_annealing(
        func=cost_function,
        bounds=sa_bounds,
        args=(i, src_tac, hires_tac, region_weights, w, target_tacs, ki_peaks),
        maxiter=max_iter,  # matlab uses Inf, we can maybe adjust this later
        initial_temp=init_temp,  # matlab default is 100, python default is 5230
        no_local_search=no_local_search,  # see stats.stackexchange.com
        callback=annealer_callback,  # Monitors and decides when to quit
        x0=x0,  # Our suggestion for a starting point
    )

    # Each iteration of the annealer solved this, then only used the cost.
    # We solve it one more time with final parameters to save full results.
    irf, target_tac_fits, ki, ki_penalty, cost = solve_stttm(
        optimization_result.x, i, src_tac, hires_tac,
        region_weights, w, target_tacs, ki_peaks
    )

    worker_end = datetime.now()
    print(f"    Finished worker for region {i} "
          f"at {worker_end.strftime('%m/%d %I:%M')} "
          f"after {worker_end - worker_start}.", flush=True)

    return {
        "i": i,
        "x0": x0,
        "result": optimization_result,
        "source_tac": src_tac,
        "ki_peaks": ki_peaks,
        "irf": irf,
        "tgt_tac_fits": target_tac_fits,
        "ki": ki,
        "ki_penalty": ki_penalty,
        "cost": cost,
        "sa_bounds": sa_bounds,
        "hires_tac": hires_tac,
        "region_weights": region_weights,
        "w": w,
        "target_tacs": target_tacs,
        "max_iter": max_iter,
        "init_temp": init_temp,
        "no_local_search": no_local_search,
        "sa_progress": global_annealer_data,
        "elapsed_time": worker_end - worker_start,
    }


def package_rate_constants(annealer_results, regions):
    """ Pack rate constants into a simple dataframe """

    final_rate_header = []
    for k in ["1", "2", "3", "i", ]:
        # PET nomenclature has capital K for 1, i, lowercase k for 2, 3
        if k in ["1", "i", ]:
            letter_k = "K"
        else:
            letter_k = "k"
        for i, region in enumerate(regions):
            final_rate_header.append(f"{letter_k}{k}_{region}")
    final_rate_data = np.ndarray((len(regions), 4 * len(regions)))
    for i, region in enumerate(regions):
        final_rate_data[i, :] = np.concatenate(
            [annealer_results[i]["result"]["x"], annealer_results[i]["ki"], ]
        )
    return pd.DataFrame(
        data=final_rate_data, columns=final_rate_header
    )


def minimize_cost_function(results, x0=None):
    """ Find parameters for each region in corrected_tacs

        :param StareResults results: An object containing pipeline data
        :param x0: x0 is randomized between bounds, but can be overridden here.
        :return: results, with additional data
    """

    # Manual configuration
    no_local_search = False

    # If we don't have a uniform set of timepoints for interpolation,
    # don't bother with the rest of the function; it won't work.
    # It's far better to check here once rather than every single iteration
    # of the source-to-target tissue model (s2ttm).
    if not results.fitted_hires_tac.has_uniform_time_delta:
        raise ValueError("Interpolated hi-res TAC does not have uniform"
                         "time deltas, and interpolation will not work.")

    # According to STARE matlab code, we could weight the target TACS by region,
    # but no advantage to that approach was found during validation.
    # So we will create a weight vector for potential future use, but
    # we set its weights to all ones.
    if results.region_weights is None:
        results.region_weights = np.ones(
            len(results.corrected_tacs.columns) - 1
        )
    # Weights by TR have already been calculated
    w = results.pvc_mean_vascular_tac.weights()

    # If x0 was overridden in the function call, we can use it.
    if x0 is None:
        # By default, we expect to generate our own randomized starting params.
        randomness = np.random.random(size=results.kde_lower_bounds.shape)
        bounds_ranges = results.kde_upper_bounds - results.kde_lower_bounds
        x0 = results.kde_lower_bounds + bounds_ranges * randomness

    sa_bounds = [(results.kde_lower_bounds[i], results.kde_upper_bounds[i])
                 for i in range(len(results.kde_lower_bounds))]

    mp_args_list = []
    for i, region in enumerate(results.corrected_tacs.columns):
        print(f"  set up simulated annealing '{region}' at {datetime.now()}")
        # Separate source and target TACs
        source_tac = TimeActivityCurve(
            activity=results.corrected_tacs[region].values,
            timepoints=results.mid_times,
            source='corrected regional tacs df',
            name=region,
        )
        target_tacs = results.corrected_tacs.drop(region, axis='columns')
        ki_peaks = [
            results.bootstrap_ki_fwhm[m, 1, 0]
            for m in range(len(results.bootstrap_ki_fwhm))
        ]

        # For multiprocessing, just set up arguments to call later in the pool
        mp_args_list.append(
            (
                i, sa_bounds, source_tac, results.fitted_hires_tac,
                results.region_weights, w, target_tacs, ki_peaks,
                results.args.annealer_iterations, 5230, no_local_search, x0
            )
        )

        # fig, axes = plot_stare_tac_fits(optimization_result)
        # fig.savefig(out_path / f"fit_tac_{region.name}_via_sa.png")

    annealer_results = run_in_mp_queue(
        worker, mp_args_list, results.args.num_cpus, results.logger
    )
    if results.args.debug and results.args.debug_path.exists():
        pickle.dump(
            mp_args_list,
            open(results.args.debug_path / "sa_args_list.pickle", "wb")
        )
        pickle.dump(
            annealer_results,
            open(results.args.debug_path / "sa_results.pickle", "wb")
        )

    # Save the rate constants in an accessible csv format.
    final_rate_df = package_rate_constants(
        annealer_results,
        regions=results.corrected_tacs.columns,
    )

    return sa_bounds, annealer_results, final_rate_df


def minimize_parameter_cost(results):
    """ Find parameters for each region in corrected_tacs

        :param StareResults results: An object containing pipeline data
        :return: results, with additional data
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Simulated Annealing")

    start_time = datetime.now()
    logger.info(f"Starting {results.args.subject} simulated annealing "
                f"at {start_time}")

    cache_file = f"sub-{results.args.subject}_step-5_minimized_params.pickle"
    sa_results = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if sa_results is None:
        # pvc_mean_tac is only used for timepoints and weights, NOT activity
        if len(results.bootstrap_rate_constants) == 0:
            logger.info("Parameter cost minimization has nothing to work "
                        "with and is not being run.")
            rpt_sect.add_line("Parameter cost minimization has nothing to work"
                              " with and is not being run.")
        else:
            sa_results = minimize_cost_function(results)
            to_cache(sa_results, results.args.cache_path, cache_file)
    else:
        logger.info("  loaded cached step 5 curve fits to save time")

    # Store annealer results in the global results object
    results.annealer_bounds = sa_results[0]
    results.annealer_results = sa_results[1]
    results.final_rate_df = pd.DataFrame(sa_results[2])

    if results.args.debug and results.args.debug_path.exists():
        with open(
            results.args.debug_path /
            f"sub-{results.args.subject}_step-5_results.pickle",
            "wb"
        ) as f:
            pickle.dump(results, f)

    print("Finished simulated annealing at {}, after {}".format(
        datetime.now(), datetime.now() - start_time
    ))

    results.final_rate_df.to_csv(
        results.args.output_path /
        f"sub-{results.args.subject}_final_stare_all_rate_constants.csv",
        index=False
    )
    final_rate_mean_df = pd.DataFrame(
        data=np.mean(results.final_rate_df.values, axis=0).reshape(
            (4, len(results.corrected_tacs.columns))
        ).T,
        columns=["K1", "k2", "k3", "Ki"],
        index=results.corrected_tacs.columns,
    )
    final_rate_mean_df.to_csv(
        results.args.output_path /
        f"sub-{results.args.subject}_final_stare_mean_rate_constants.csv",
        index=True
    )

    caption = "Final TACs plotted by source region"
    fig = plot_all_stare_tac_fits(
        results.corrected_tacs, results.mid_times, results.annealer_results,
        title=f"{results.args.subject} final TAC fits by source region"
    )
    filename = f"sub-{results.args.subject}_step-5_final_fits.png"
    fig.savefig(results.args.fig_path / filename, bbox_inches='tight')
    rpt_sect.add_figure(results.args.fig_path / filename, caption)

    rpt_sect.add_line(final_rate_mean_df.to_html(
        float_format=lambda _: f"{_:0.4f}"
    ), log=False)

    rpt_sect.add_link(
        results.args.output_path /
        f"sub-{results.args.subject}_final_stare_all_rate_constants.csv",
        text="Table of all source/target rate constants"
    )

    rpt_sect.end()
    results.write_report()
    return results
