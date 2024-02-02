import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans

from .centroid import Centroid
from .mp_queues import run_in_mp_queue


def likely_irreversible(c):
    """Return true if centroid appears irreversible.

    :param Centroid c: The centroid to assess
    :return: True if irreversible, False otherwise
    """

    # If the highest value in the timeseries is the last one,
    # this voxel is likely irreversible
    return c.activity[-1] == max(c.activity)


def likely_irreversible_linear(c, return_features=False, skip_t0=False):
    """ Return true if centroid has a positive slope.

    :param Centroid c: The centroid to assess
    :param return_features: Return (slope, intercept) rather than a boolean
    :param skip_t0: Calculate slope from all but the first point
    :return: True if centroid has a positive slope.
    """

    if skip_t0:
        slope, intercept = np.polyfit(c.timepoints[1:], c.activity[1:], 1)
    else:
        slope, intercept = np.polyfit(c.timepoints, c.activity, 1)

    if return_features:
        return slope, intercept
    else:
        return slope > 0.0


def likely_noise(c):
    """Return true if centroid appears to just be noise.

    :param Centroid c: The centroid to assess
    :return: True if noise, False otherwise
    """

    # If activity at any point after the first one is negative,
    # this voxel is likely noise
    return np.any(c.activity[1:] < 0)


def likely_vascular(c):
    """Return true if centroid appears vascular.

    :param Centroid c: The centroid to assess
    :return: True if vascular, False otherwise
    """

    # If this centroid is reversible signal,
    # it is probably vascular
    return not likely_noise(c) and not likely_irreversible(c)


def likely_peripheral(c):
    """Return true if centroid appears peripheral.

    :param Centroid c: The centroid to assess
    :return: True if peripheral, False otherwise
    """

    # If this centroid represents peripheral areas,
    # it is probably peripheral
    # NOTE: c is a centroid, defined in centroid.py
    # NOTE: c.activity is the timeseries you probably care about.
    # NOTE: Use functions above as examples.
    # NOTE: This should always return True; return something more useful.
    return len(c.activity) > 0


def k_means_worker(arg_tuple):
    """ A worker function to calculate k-means for one k

        This worker can be launched in a separate process to calculate k
        clusters, and save k-means results in a returnable dict.

    """

    # Workers get a single argument, so the caller must pack arguments
    # into a tuple and the worker (this function) must unpack them.
    # This order must match exactly the order where they're packed.
    (k, data, verbose) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting k-means worker for k={k} "
          f"at {worker_start.strftime('%m/%d %I:%M')}", flush=True)
    log_messages = []

    k_means = KMeans(
        init="k-means++",
        n_clusters=k,
        n_init=3,
        max_iter=1024 ** 2,
        random_state=42,
        verbose=verbose,
    )
    k_means.fit(data)

    log_messages.append(
        f"  lowest inertia == {k_means.inertia_:0.0f}"
        f" after {k_means.n_iter_} iterations"
        f" in {datetime.now() - worker_start}."
    )

    worker_end = datetime.now()
    print(f"    Finished k-means worker for k={k} "
          f"at {worker_end.strftime('%m/%d %I:%M')} "
          f"after {worker_end - worker_start}.", flush=True)

    return {
        "k": k,
        "k_means": k_means,
        "log_messages": log_messages,
    }


def find_centroids(
        data,
        ks,
        features,
        mid_times=None,
        num_cpus=1,
        verbose=0,
        logger=None,
):
    """Step 1. From all PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param features: A dict of functions to assign features to centroids
    :param iterable mid_times: will be stored alongside activity in TACs
    :param num_cpus: How many CPUs to deploy on multiprocessing
    :param int verbose: Set non-zero to increase logging, higher is more
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger
    print(f"Setting up {len(ks)} K-means values across {num_cpus} cpus.")

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    list_of_args = []
    for k in ks:
        list_of_args.append((k, data, verbose))
    # Run each tuple of arguments in a separate process to save time.

    k_means_results = run_in_mp_queue(
        k_means_worker, list_of_args, num_cpus, logger
    )

    # Retrieve and organize k-means results
    k_means_fits = {}
    all_centroids = []
    for kmeans_result in k_means_results:
        k = kmeans_result['k']
        k_means_fits[k] = kmeans_result['k_means']

        # Make a place to store counts while we look through clusters
        feature_counts = {"total": 0}
        for feature_label in features.keys():
            feature_counts[feature_label] = 0

        # Go through clusters, creating a centroid object for each one.
        for i in range(k):
            cc = kmeans_result['k_means'].cluster_centers_[i]
            this_centroid = Centroid(
                activity=cc,
                timepoints=mid_times,
                label=i + 1,  # should be non-zero as zero indicates background
                k=k,
                name=f"centroid {i + 1}/{k}",
                source="k-means",
                # labels=k_means.labels_ + 1,
                # blob_count=len(blob_ids),
                # voxels_per_blob=np.mean(voxel_counts),
            )
            this_centroid.features = {}

            # Count features for reporting and
            # Save features of this centroid, like whether it is
            # noise, vascular, peripheral, etc. using functions provided.
            for feature_label, fxn in features.items():
                this_centroid.features[feature_label] = fxn(this_centroid)
                if this_centroid.features[feature_label]:
                    feature_counts[feature_label] += 1
            feature_counts["total"] += 1
            all_centroids.append(this_centroid)

        if verbose:
            for label, count in feature_counts.items():
                if label != "total":
                    kmeans_result['log_messages'].append(
                        f"  {count:03d} / "
                        f"{feature_counts['total']:03d} are {label}"
                    )

        # Rather than logging them out of order, we pool all messages from
        # a given k, hold them, and we emit them all in one chunk here.
        logger.info(f" - K-Means for {kmeans_result['k']} complete.")
        for message in kmeans_result['log_messages']:
            logger.info(message)

    post_k_timestamp = datetime.now()
    logger.info(
        f"All {len(ks)} k-means finished in "
        f"{post_k_timestamp - pre_k_timestamp}"
    )

    return all_centroids, k_means_fits


def find_vascular_centroids(
        data,
        ks,
        allow_override=True,
        mid_times=None,
        num_cpus=1,
        verbose=0,
        logger=None,
):
    """Step 1. From all PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for k-means clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param bool allow_override: Check for a higher cluster one t later
    :param iterable mid_times: will be stored alongside activity in TACs
    :param int num_cpus: how many processes to use on finding centroids
    :param int verbose: Set non-zero to increase logging, higher is more
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # Run k-means, and label centroids with features
    vascular_features = {
        "likely_noise": likely_noise,
        "likely_irreversible": likely_irreversible,
        "likely_irreversible_linear": likely_irreversible_linear,
        "likely_vascular": likely_vascular,
    }
    all_centroids, k_means_fits = find_centroids(
        data,
        ks,
        vascular_features,
        mid_times=mid_times,
        num_cpus=num_cpus,
        verbose=verbose,
    )

    for k in ks:
        # Split all centroids into vascular and other, for this value of k
        vascular_centroids = []
        other_centroids = []
        for centroid in [c for c in all_centroids if c.k == k]:
            if centroid.features["likely_vascular"]:
                vascular_centroids.append(centroid)
            else:
                other_centroids.append(centroid)

        for i, vc in enumerate(vascular_centroids):
            logger.debug(f"  {vc.peak_value:0.3f} at {vc.peak_index}")

        # Label the top candidate for a vascular cluster from this k value.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous or sinus
        if len(vascular_centroids) > 0:
            # Select the 'best' from among all vascular centroids.
            # Do this two ways, find the earliest peak and the highest peak,
            # then sort the earliest by height and the highest by onset.
            peak_idxs = np.array([c.peak_index for c in vascular_centroids])
            peak_vals = np.array([c.peak_value for c in vascular_centroids])
            earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
            highest_peak_idxs = np.where(peak_vals == np.max(peak_vals))[0]
            # Of the vascular centroids peaking at the same earliest time,
            # which is highest?
            highest_early_peak_idx = earliest_peak_idxs[np.argmax([
                vascular_centroids[i].peak_value for i in earliest_peak_idxs
            ])]
            # Of the vascular centroids with the highest peaks,
            # which is earliest?
            earliest_high_peak_idx = highest_peak_idxs[np.argmax([
                vascular_centroids[i].peak_index for i in highest_peak_idxs
            ])]
            # Label this centroid as best, at least for this value of k
            vascular_centroids[highest_early_peak_idx].best_in_k = True
            # vascular_centroids[earliest_high_peak_idx].best_in_k = True
            logger.debug(
                "  Early centroid [{}/{}] has peak of {:0.3f} at t {}".format(
                    vascular_centroids[highest_early_peak_idx].label,
                    vascular_centroids[highest_early_peak_idx].k,
                    vascular_centroids[highest_early_peak_idx].peak_value,
                    vascular_centroids[highest_early_peak_idx].peak_index,
                )
            )
            logger.debug(
                "  High centroid [{}/{}] has peak of {:0.3f} at t {}".format(
                    vascular_centroids[earliest_high_peak_idx].label,
                    vascular_centroids[earliest_high_peak_idx].k,
                    vascular_centroids[earliest_high_peak_idx].peak_value,
                    vascular_centroids[earliest_high_peak_idx].peak_index,
                )
            )

        plural_string = "" if len(vascular_centroids) == 1 else "s"
        logger.info(
            f"  found {len(vascular_centroids)} potential vascular"
            f" cluster{plural_string} with k={k}."
        )

    # Which cluster-centroid timeseries has the highest peak?
    # And where is that peak?
    # from Matlab vascClust.m:160:174
    best_in_k_centroids = [c for c in all_centroids if c.best_in_k]
    top_indices, top_frequencies = np.unique(
        [c.peak_index for c in best_in_k_centroids], return_counts=True
    )
    # Which time point is most likely to have the highest value?
    """
    # This is an alternative way to select the best centroid from among those
    # NOT selected as the best in their individual k pool. When it was tested,
    # it did not perform as well as the original, but it remains here in case
    # someone else has the same idea and wants to test it.
    vascular_peaks = [
        (c.k, np.argmax(c.activity), np.max(c.activity))
        for c in all_centroids
        if c.features["likely_vascular"]
    ]
    values_per_peak = [
        (idx, np.mean([vp[2] for vp in vascular_peaks if vp[1] == idx]))
        for idx in np.unique([vp[1] for vp in vascular_peaks])
    ]
    best_centroid_idx = values_per_peak[
        np.argmax([vp[1] for vp in values_per_peak])
    ][0]
    centroids_with_best_idx = [
        c for c in all_centroids
        if (c.peak_index == best_centroid_idx)
    ]
    """
    # This is the most likely time point to have the best vascular peak,
    # but it is only about 90% accurate in our tests. So we'll also consider
    # the next time point, but only if it has both a higher peak than our
    # current best centroid and a more spatially concise clustering.
    if len(top_frequencies) == 0:
        logger.error(f"None of the {len(all_centroids)} clusters appear "
                     "vascular. There's nothing more to be done.")
        raise TypeError("No vascular clusters available.")
    best_centroid_idx = top_indices[np.argmax(top_frequencies)]

    # Make a list of best-in-k centroids that peak at the same,
    # most common, time point
    centroids_with_best_idx = [
        c for c in best_in_k_centroids if (c.peak_index == best_centroid_idx)
    ]
    # Of those centroids peaking together, which one peaks highest?
    first_choice_centroid = centroids_with_best_idx[
        np.argmax([c.peak_value for c in centroids_with_best_idx])
    ]
    best_centroid = first_choice_centroid

    # We probably have the best centroid, but if the next time point contains
    # a centroid with a higher value AND a more spatially concise clustering,
    # we should consider the runner-up time point a better bet. Even if that
    # centroid was not 'best_in_k', because 'best_in_k' was also
    # restricted to this same earliest peak. We explicitly want to see
    # if the peak being too early caused us to miss a better option here.
    if allow_override:
        alt_centroid_idx = best_centroid_idx + 1
        centroids_with_alt_idx = [
            c for c in all_centroids
            if ((c.peak_index == alt_centroid_idx) &
                (c.features['likely_vascular']))
        ]
        # Of those centroids peaking together, which one peaks highest?
        if len(centroids_with_alt_idx) > 0:
            alt_centroid = centroids_with_alt_idx[
                np.argmax([c.peak_value for c in centroids_with_alt_idx])
            ]
            if alt_centroid.peak_value > first_choice_centroid.peak_value:
                # For speed, we opted not to calculate spatial clustering on
                # every cluster; we only do this now if we want to compare them.
                if alt_centroid.blob_count == 0:
                    labels = k_means_fits[alt_centroid.k].labels_ + 1
                    alt_centroid.update_spatial_clusters(
                        labels, verbose=verbose, logger=logger,
                    )
                if first_choice_centroid.blob_count == 0:
                    labels = k_means_fits[first_choice_centroid.k].labels_ + 1
                    first_choice_centroid.update_spatial_clusters(
                        labels, verbose=verbose, logger=logger,
                    )
                if alt_centroid.blob_count < first_choice_centroid.blob_count:
                    # We have an alternate centroid with a higher peak and a
                    # less spatially sparse clustering. We will use it.
                    logger.info(
                        f"Overriding the cluster selection with an alternate!!"
                        f" original best {first_choice_centroid.description()};"
                        f" new best is {alt_centroid.description()}."
                    )
                    best_centroid.source = ", ".join([
                        best_centroid.source, "original best, overridden"
                    ])
                    alt_centroid.source = ", ".join([
                        alt_centroid.source, "overrides first choice"
                    ])
                    best_centroid = alt_centroid
                else:
                    logger.info(f"An alternate, {alt_centroid.description()}, "
                                "was considered and dropped.")
            else:
                logger.info("No alternate centroids had higher peaks.")
        else:
            logger.info("No alternate centroids were considered.")

    # Label the centroid with the highest peak value
    best_centroid.best_overall = True
    logger.info(
        f"The very best cluster is {best_centroid.description()}."
    )

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits


def find_peripheral_centroids(
        data,
        ks,
        num_cpus=1,
        mid_times=None,
        verbose=0,
        logger=None,
):
    """Step 1. From all PET data, find a peripheral cluster.

    Loop over all values for k in ks, looking for clusters that
    exhibit peripheral-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param int num_cpus: How many processes to use finding centroids
    :param iterable mid_times: will be stored alongside activity in TACs
    :param int verbose: Set non-zero to increase logging, higher is more
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    vascular_features = {
        "likely_noise": likely_noise,
        "likely_irreversible": likely_irreversible,
        "likely_vascular": likely_vascular,
    }
    all_centroids, k_means_fits = find_centroids(
        data,
        ks,
        vascular_features,
        mid_times=mid_times,
        num_cpus=num_cpus,
        verbose=verbose,
    )

    for k in ks:
        vascular_centroids = []
        other_centroids = []
        for centroid in [c for c in all_centroids if c.k == k]:
            if centroid.features["likely_vascular"]:
                vascular_centroids.append(centroid)
            else:
                other_centroids.append(centroid)

        for i, vc in enumerate(vascular_centroids):
            logger.debug(f"  {vc.peak_value:0.3f} at {vc.peak_index}")

        # Label the top candidate for a vascular cluster from this clustering.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous or sinus
        if len(vascular_centroids) > 0:
            peak_idxs = np.array([c.peak_index for c in vascular_centroids])
            earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
            highest_early_peak_idx = earliest_peak_idxs[
                np.argmax(
                    [vascular_centroids[i].peak_value for i in
                     earliest_peak_idxs]
                )
            ]
            # Label this centroid as best, at least for this value of k
            vascular_centroids[highest_early_peak_idx].best_in_k = True
            logger.debug(
                "  Best centroid [{}] has peak of {:0.3f} at t idx {}".format(
                    vascular_centroids[highest_early_peak_idx].label,
                    vascular_centroids[highest_early_peak_idx].peak_value,
                    vascular_centroids[highest_early_peak_idx].peak_index,
                )
            )

        plural_string = "" if len(vascular_centroids) == 1 else "s"
        logger.info(
            f"  found {len(vascular_centroids)} potential vascular"
            f" cluster{plural_string} with k={k}."
        )

    # Which cluster-centroid timeseries has the highest peak?
    # And where is that peak?
    # from Matlab vascClust.m:160:174
    best_in_k_centroids = [c for c in all_centroids if c.best_in_k]
    top_indices, top_frequencies = np.unique(
        [c.peak_index for c in best_in_k_centroids], return_counts=True
    )
    # Which time point is most likely to have the highest value?
    best_centroid_idx = top_indices[np.argmax(top_frequencies)]

    # Make a list of centroids that peak at the same, most common, time point
    centroids_with_best_idx = [
        c
        for c in all_centroids
        if ((c.peak_index == best_centroid_idx) and c.best_in_k)
    ]
    # Of those centroids peaking together, which one peaks highest?
    best_centroid = centroids_with_best_idx[
        np.argmax([c.peak_value for c in centroids_with_best_idx])
    ]
    # Label the centroid with the highest peak value
    best_centroid.best_overall = True
    logger.info(
        f"The very best cluster is label {best_centroid.label} "
        f"from k {best_centroid.k}."
    )
    logger.info(
        f"It peaked at frame {best_centroid.peak_index + 1} "
        f"to a value of {best_centroid.peak_value}."
    )

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits
