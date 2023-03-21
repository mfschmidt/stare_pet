import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from datetime import datetime
import multiprocessing as mp

from .fitting_models import solve_stttm
from .timeactivitycurve import TimeActivityCurve
from .plotting import plot_all_stare_tac_fits
from .util import from_cache, to_cache


def cost_function(
        x, src_idx, source_tac, uniform_tac,
        region_weights, weights, target_tacs, boot_ki_ks_density_peak
):
    """ Run the s2ttm model and return a measure for poorness of fit

        :param np.array x: parameter estimates, in a 1D [regions * ks] array
        :param int src_idx: which column of x0 is source, all else target
        :param source_tac: Corrected Time Activity Curve for source
        :param uniform_tac: Time Activity Curve at uniform hi res
        :param np.array region_weights: Weights to modify each region's cost
        :param np.array weights: Weights relative to each time point's duration
        :param pd.DataFrame target_tacs: Corrected TACs for each region
        :param np.array boot_ki_ks_density_peak: Previously calculated kis
    """

    irf, target_tac_fits, ki, ki_penalty, cost = solve_stttm(
        x, src_idx, source_tac, uniform_tac, region_weights, weights,
        target_tacs, boot_ki_ks_density_peak,
    )
    return cost


def worker(arg_tuple):
    """ the wrapper function designed to be run in parallel pools """

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
    }


def queue_consumer(task_q, rslt_q, pid):
    print(f"  process {pid} is alive and checking the queue.")
    while True:
        # Get the next set of parameters to minimize
        msg = task_q.get()
        if msg is None:
            break
        else:
            # Save results to the result queue,
            print(f"  process {pid} sending worker tuple "
                  f"for region {msg[0]}...", flush=True)
            rslt_q.put(worker(msg))

    print(f"  process {pid} consumed a None and is exiting.")


def minimize_cost_function(results):
    """ Find parameters for each region in corrected_tacs

        :param StareResults results: An object containing pipeline data
        :return: results, with additional data
    """

    # Manual configuration
    # annealer always goes to max_iter, needs a callback for setting limits
    max_iter = 5000  # 20k is better, but is slow, about an hour per 4 thousand.
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

    # Generate candidate parameters within bootstrap ranges
    # These are unraveled in 'F' Fortran order to match matlab's 18-item arrays.
    # So x0 will have a 1D 18-item array of 6 k1, 6 k2, 6 k3
    lo_bounds = np.ravel(
        np.min(results.bootstrap_rate_constants, axis=0), order='F'
    )
    hi_bounds = np.ravel(
        np.max(results.bootstrap_rate_constants, axis=0), order='F'
    )
    randomness = np.random.random(size=lo_bounds.shape)
    x0 = lo_bounds + (hi_bounds - lo_bounds) * randomness
    sa_bounds = [(lo_bounds[i], hi_bounds[i]) for i in range(len(lo_bounds))]
    # x0 is now a [num_regions x num_ks] ndarray of candidate parameters

    annealer_results = {}
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
                max_iter, 5230, no_local_search, x0
            )
        )

        # fig, axes = plot_stare_tac_fits(optimization_result)
        # fig.savefig(out_path / f"fit_tac_{region.name}_via_sa.png")

    # Create the process pool and launch processes to deal with it.
    print(f"Creating MP Queue at {datetime.now().strftime('%Y-%m-%d %I:%M')}")
    processes = []

    # Fill the queue with jobs
    task_queue = mp.Queue()
    rslt_queue = mp.Queue()
    for argument_tuple in mp_args_list:
        task_queue.put(argument_tuple)
    print(f"  queue has {task_queue.qsize()} jobs")
    for _ in range(results.args.num_cpus):
        task_queue.put(None)  # to kill each worker when real jobs are complete
    print(f"  queue has {task_queue.qsize()} (jobs + Nones)")

    # Create processes to handle the jobs
    for pid in range(results.args.num_cpus):
        proc = mp.Process(
            target=queue_consumer, args=(task_queue, rslt_queue, pid)
        )
        # proc.daemon = True  # process run in background and clean up its mess
        print(f"  start process {pid}")
        proc.start()
        processes.append(proc)
    # All processes are now running separately.
    # This process will continue without waiting until rslt_queue.get() below.

    print(f"  queue has {rslt_queue.qsize()} results")

    # Results are stuck back onto the queue, so we need to get them.
    # This rslt_queue.get() should wait for a result,
    # pausing this thread until processes finish.
    for _ in mp_args_list:
        annealer_result = rslt_queue.get()
        print(f"  GOT a result")
        annealer_results[annealer_result["i"]] = annealer_result

    print(f"Completed MP Queue at {datetime.now().strftime('%Y-%m-%d %I:%M')}")
    print(f"  queue has {rslt_queue.qsize()} results")

    # Save the rate constants in an accessible csv format.
    final_rate_header = []
    for k in ["1", "2", "3", "i", ]:
        for i, region in enumerate(results.corrected_tacs.columns):
            final_rate_header.append(f"k{k}_{region}")
    final_rate_data = np.ndarray((
        len(results.corrected_tacs.columns),
        4 * len(results.corrected_tacs.columns)
    ))
    for i, region in enumerate(results.corrected_tacs.columns):
        final_rate_data[i, :] = np.concatenate(
            [annealer_results[i]["result"]["x"], annealer_results[i]["ki"], ]
        )
    final_rate_df = pd.DataFrame(
        data=final_rate_data, columns=final_rate_header
    )
    final_rate_df.to_csv(
        results.args.output_path / "final_stare_all_rate_constants.csv",
        index=False
    )
    final_rate_mean_df = pd.DataFrame(
        data=np.mean(final_rate_data, axis=0).reshape(
            (1, 4 * len(results.corrected_tacs.columns))
        ),
        columns=final_rate_header,
        index=pd.Index(
            ["n/a" if results.args.subject is None else results.args.subject, ],
            name="subject"
        ),
    )
    final_rate_mean_df.to_csv(
        results.args.output_path / "final_stare_mean_rate_constants.csv",
        index=False
    )

    fig = plot_all_stare_tac_fits(
        results.corrected_tacs, results.mid_times, annealer_results
    )
    fig.savefig(results.args.fig_path / f"step_5_final_fits.png")

    results.annealer_bounds = sa_bounds
    results.annealer_results = annealer_results
    results.final_rate_df = final_rate_df

    return results


def minimize_parameter_cost(results):
    """ Find parameters for each region in corrected_tacs

        :param StareResults results: An object containing pipeline data
        :return: results, with additional data
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Simulated Annealing")

    start_time = datetime.now()
    logger.info(f"Starting simulated annealing at {start_time}")

    cache_file = "step_5_minimized_params.pkl"
    annealer_results = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if annealer_results is None:
        # pvc_mean_tac is only used for timepoints and weights, NOT activity
        annealer_results = minimize_cost_function(results)
        to_cache(annealer_results, results.args.cache_path, cache_file)
    else:
        logger.info("  loaded cached step 4a curve fits to save time")

    print("Finished simulated annealing at {}, after {}".format(
        datetime.now(), datetime.now() - start_time
    ))

    rpt_sect.end()
    return results
