import numpy as np
import pandas as pd
import pickle
from scipy.optimize import dual_annealing
from datetime import datetime

from .fitting_models import source_to_target_tissue_model
from .timeactivitycurve import TimeActivityCurve


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

    # Extract parameter estimates for the source region, these end up scalars
    # x is a 1D [regions * ks] array, with 3 ks
    num_regions = int(len(x) / 3)
    k_1_s = x[src_idx]
    k_2_s = x[src_idx + num_regions]
    k_3_s = x[src_idx + num_regions + num_regions]
    k_i_s = k_1_s * (k_3_s / (k_2_s + k_3_s))
    source_parameters = np.array([k_1_s, k_2_s, k_3_s, k_i_s])

    # Target parameters, strip out the source region's ks, keep the rest
    xt = np.delete(
        x,
        (src_idx, src_idx + num_regions, src_idx + num_regions + num_regions)
    )
    k_1_t = xt[0:num_regions - 1]
    k_2_t = xt[num_regions - 1:(num_regions - 1) * 2]
    k_3_t = xt[(num_regions - 1) * 2:]
    k_i_t = k_1_t * (k_3_t / (k_2_t + k_3_t))
    target_parameters = np.array([k_1_t, k_2_t, k_3_t, k_i_t])

    k_i = np.insert(k_i_t, src_idx, k_i_s)

    # Calculate the s2ttm
    impulse_response_func, target_tac_fits = source_to_target_tissue_model(
        source_parameters, target_parameters, source_tac, uniform_tac
    )

    # Differences between s2ttm k_i and previous peak k_i are BAD
    ki_penalty = np.sum(abs(k_i - boot_ki_ks_density_peak))

    cost = np.sum(
        region_weights *
        np.sum(weights[:, np.newaxis] * (target_tacs - target_tac_fits) ** 2)
    ) + ki_penalty

    return cost


def minimize_cost_function(
        k_constants, mid_times, corrected_tacs,
        weights, region_weights, uniform_vasc_tac,
        bootstrap_ki_peaks, out_path,
        debug_path=None, no_local_search=False, subject=None, max_iter=20000,
):
    """ Find parameters for each region in corrected_tacs

        :param k_constants: 1D array of
        :param mid_times: 1D array of timepoints midway through each scan
        :param corrected_tacs: timepoint x region dataframe, tac in each col
        :param weights: 1D array with a duration-based weight for each timepoint
        :param region_weights: 1D array with a weight for each region
        :param uniform_vasc_tac: TAC with a uniform time axis for interpolation
        :param bootstrap_ki_peaks: array
        :param out_path: Path for writing results
        :param debug_path: Path for writing debug data, optional
        :param no_local_search: set to True for classic brute force annealing
        :param subject: if provided, subject will be added to the final csv
        :param max_iter: stop annealing after this many iterations
    """

    start_time = datetime.now()
    print(f"Starting simulated annealing at {start_time}")
    # If we don't have a uniform set of timepoints for interpolation,
    # don't bother with the rest of the function; it won't work.
    # It's far better to check here once rather than every single iteration
    # of the s2ttm.
    if not uniform_vasc_tac.has_uniform_time_delta:
        raise ValueError("Interpolated hi-res TAC does not have uniform"
                         "time deltas, and interpolation will not work.")

    # According to STARE matlab code, we could weight the target TACS by region,
    # but no advantage to that approach was found during validation.
    # So we will create a weight vector for potential future use, but
    # we set its weights to all ones.
    if region_weights is None:
        region_weights = np.ones(len(corrected_tacs.columns) - 1)

    # Generate candidate parameters within bootstrap ranges
    # These are unraveled in 'F' Fortran order to match matlab's 18-item arrays.
    # So x0 will have a 1D 18-item array of 6 k1, 6 k2, 6 k3
    lo_bounds = np.ravel(np.min(k_constants, axis=0), order='F')
    hi_bounds = np.ravel(np.max(k_constants, axis=0), order='F')
    randomness = np.random.random(size=lo_bounds.shape)
    x0 = lo_bounds + (hi_bounds - lo_bounds) * randomness
    sa_bounds = [(lo_bounds[i], hi_bounds[i]) for i in range(len(lo_bounds))]
    # x0 is now a [num_regions x num_ks] ndarray of candidate parameters

    results = {}
    for i, region in enumerate(corrected_tacs.columns):
        print(f"  starting {region} simulated annealing at {datetime.now()}")
        # Separate source and target TACs
        source_tac = TimeActivityCurve(
            activity=corrected_tacs[region].values,
            timepoints=mid_times,
            source='corrected regional tacs df',
            name=region,
        )
        target_tacs = corrected_tacs.drop(region, axis='columns')
        ki_peaks = [
            bootstrap_ki_peaks[m, 1, 0]
            for m in range(len(bootstrap_ki_peaks))
        ]

        # Scipy's dual_annealing function requires that the 'cost_function'
        # function must accept x as its first argument as a 1D array.
        # The bounds must be a (lo, hi) pair for each element of that array.
        optimization_result = dual_annealing(
            func=cost_function,
            bounds=sa_bounds,
            args=(
                i, source_tac, uniform_vasc_tac,
                region_weights, weights,
                target_tacs, ki_peaks,
            ),
            maxiter=max_iter,  # matlab uses Inf, we can maybe adjust this later
            initial_temp=5230,  # matlab default is 100, python default is 5230
            no_local_search=no_local_search,  # see stats.stackexchange.com
            x0=x0,  # Our suggestion for a starting point
        )

        # After annealing, apply the best fit parameters.
        optx = optimization_result.x
        num_regions = int(len(optx) / 3)
        k_1_s = optx[i]
        k_2_s = optx[i + num_regions]
        k_3_s = optx[i + num_regions + num_regions]
        source_parameters = np.array([k_1_s, k_2_s, k_3_s, ])

        # Target parameters, strip out the source region's ks, keep the rest
        optxt = np.delete(
            optx, (i, i + num_regions, i + num_regions + num_regions)
        )
        k_1_t = optxt[0:num_regions - 1]
        k_2_t = optxt[num_regions - 1:(num_regions - 1) * 2]
        k_3_t = optxt[(num_regions - 1) * 2:]
        target_parameters = np.array([k_1_t, k_2_t, k_3_t, ])

        irf, tgt_fits = source_to_target_tissue_model(
            source_parameters, target_parameters, source_tac, uniform_vasc_tac
        )

        # TODO: Track ki vars; where are they needed and where are they not?
        k_i_s = k_1_s * (k_3_s / (k_2_s + k_3_s))
        k_i_t = k_1_t * (k_3_t / (k_2_t + k_3_t))
        k_i = np.insert(k_i_t, i, k_i_s)
        ki_penalty_term = np.sum(abs(k_i - ki_peaks))
        cost_term = np.sum(
            region_weights *
            np.sum(weights[:, np.newaxis] * (target_tacs - tgt_fits) ** 2)
        ) + ki_penalty_term

        results[i] = {
            "x0": x0, "result": optimization_result,
            "source_tac": source_tac, "ki_peaks": ki_peaks,
            "irf": irf, "tgt_fits": tgt_fits, "ki": k_i,
            "cost": cost_term, "ki_penalty": ki_penalty_term,
            "cost_penalty_ratio": cost_term / ki_penalty_term
        }

        # fig, axes = plot_stare_tac_fits(optimization_result)
        # fig.savefig(out_path / f"fit_tac_{region.name}_via_sa.png")

    if debug_path is not None:
        pickle.dump(
            results, open(debug_path / f"sa_results.pkl", "wb")
        )

    # Save the rate constants in an accessible csv format.
    final_rate_header = []
    for k in ["1", "2", "3", "i", ]:
        for i, region in enumerate(corrected_tacs.columns):
            final_rate_header.append(f"k{k}_{region}")
    final_rate_data = np.ndarray(
        (len(corrected_tacs.columns), 4 * len(corrected_tacs.columns))
    )
    for i, region in enumerate(corrected_tacs.columns):
        final_rate_data[i, :] - np.concatenate(
            [results[i]["result"]["x"], results[i]["ki"], ]
        )
    final_rate_df = pd.DataFrame(
        data=final_rate_data, columns=final_rate_header
    )
    final_rate_df.to_csv(
        out_path / "final_stare_all_rate_constants.csv", index=False
    )
    final_rate_mean_df = pd.DataFrame(
        data=np.mean(final_rate_data, axis=0).reshape(
            (1, 4 * len(corrected_tacs.columns))
        ),
        columns=final_rate_header,
        index=pd.Index(
            ["n/a" if subject is None else subject, ], name="subject"
        ),
    )
    final_rate_mean_df.to_csv(
        out_path / "final_stare_mean_rate_constants.csv", index=False
    )

    print("Finished simulated annealing at {}, after {}".format(
        datetime.now(), datetime.now() - start_time
    ))

    return results
