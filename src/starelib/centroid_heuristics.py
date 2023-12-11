import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans
import pandas as pd

from .centroid import Centroid
from .util import reshape_labels_to_3d
from .mp_queues import run_in_mp_queue


def likely_irreversible(c):
    """Return true if centroid appears irreversible.

    :param Centroid c: The centroid to assess
    :return: True if irreversible, False otherwise
    """

    # If the highest value in the timeseries is the last one,
    # this voxel is likely irreversible
    return c.activity[-1] == max(c.activity)


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


def worker(arg_tuple):
    """ A worker function to calculate k-means and blobs for one k

        This worker can be launched in a separate process to calculate k
        clusters, and describe their centroids and characteristics in a
        Centroid object for each cluster.

    """

    # Workers get a single argument, so the caller must pack arguments
    # into a tuple and the worker (this function) must unpack them.
    # This order must match exactly the order where they're packed in.
    (k, data, features, vol_shape, mid_times, verbose) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting k-means worker for k {k} "
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

    # Count features for reporting, not necessary for execution
    feature_counts = {"total": 0}
    for feature_label in features.keys():
        feature_counts[feature_label] = 0

    # Find reasonable timeseries in the cluster means.
    # count_irreversible, count_noise = 0, 0
    centroids = []
    for i in range(k_means.n_clusters):
        blob_df, blob_ids, voxel_counts = get_cluster_blobs(
            reshape_labels_to_3d(k_means.labels_, vol_shape),
            label=i, verbose=verbose,
        )
        cc = k_means.cluster_centers_[i]
        this_centroid = Centroid(
            activity=cc,
            timepoints=mid_times,
            label=i + 1,  # should be non-zero as zero indicates background
            k=k,
            labels=k_means.labels_ + 1,
            name=f"centroid {i + 1}/{k}",
            blob_count=len(blob_ids),
            voxels_per_blob=np.mean(voxel_counts),
        )
        # Save features of this centroid, like whether it is
        # noise, vascular, peripheral, etc. using functions provided.
        for feature_label, fxn in features.items():
            this_centroid.features[feature_label] = fxn(this_centroid)
            if this_centroid.features[feature_label]:
                feature_counts[feature_label] += 1
        feature_counts["total"] += 1

        centroids.append(this_centroid)

    if verbose:
        for label, count in feature_counts.items():
            if label != "total":
                log_messages.append(
                    f"  {count:03d} / {feature_counts['total']:03d} are {label}"
                )

    worker_end = datetime.now()
    print(f"    Finished k-means worker for k {k} "
          f"at {worker_end.strftime('%m/%d %I:%M')} "
          f"after {worker_end - worker_start}.", flush=True)

    return {
        "k": k,
        "k_means": k_means,
        "centroids": centroids,
        "log_messages": log_messages,
    }


def find_centroids(
    data,
    vol_shape,
    ks,
    features,
    mid_times=None,
    num_cpus=-1,
    verbose=0,
    logger=None,
):
    """Step 1. From all PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param tuple vol_shape: The original 3d shape of each column vector in data
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param features: A dict of functions to assign features to centroids
    :param iterable mid_times: will be stored alongside activity in TACs
    :param num_cpus: How many CPUs to deploy on multiprocessing
    :param int verbose: Set non-zero to increase logging, higher is more
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    k_means_fits = {}
    all_centroids = []
    list_of_args = []
    for k in ks:
        list_of_args.append((k, data, features, vol_shape, mid_times, verbose))
    # Run each tuple of arguments in a separate process to save time.
    kmeans_results = run_in_mp_queue(
        worker, list_of_args, num_cpus, logger
    )
    for kmeans_result in kmeans_results:
        for centroid in kmeans_result['centroids']:
            all_centroids.append(centroid)
        k_means_fits[kmeans_result['k']] = kmeans_result['k_means']
        # Rather than logging them out of order, we pool all messages from
        # a given k, hold them, and we emit them all in one chunk here.
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
    vol_shape,
    ks,
    mid_times=None,
    num_cpus=-1,
    verbose=0,
    logger=None,
):
    """Step 1. From all PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for k-means clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param tuple vol_shape: The original 3d shape of each column vector in data
    :param iterable ks: Iterable of integers, each used as a k in k-means
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
        "likely_vascular": likely_vascular,
    }
    all_centroids, k_means_fits = find_centroids(
        data,
        vol_shape,
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
    alt_centroid_idx = best_centroid_idx + 1
    centroids_with_alt_idx = [
        c for c in all_centroids
        if (c.peak_index == alt_centroid_idx) & (c.features['likely_vascular'])
    ]
    # Of those centroids peaking together, which one peaks highest?
    alt_centroid = centroids_with_alt_idx[
        np.argmax([c.peak_value for c in centroids_with_alt_idx])
    ]
    if alt_centroid.peak_value > first_choice_centroid.peak_value:
        if alt_centroid.blob_count < first_choice_centroid.blob_count:
            # We have an alternate centroid with a higher peak and a more
            # spatially concise clustering. We will use it.
            logger.info(
                f"Overriding the best cluster selection with an alternate!! "
                f"The original best was {first_choice_centroid.description()}. "
                f"The new best is {alt_centroid.description()}. "
            )
            best_centroid = alt_centroid
        else:
            logger.info(f"An alternate centroid, {alt_centroid.description()}, "
                        "was considered and dropped.")
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
    vol_shape,
    ks,
    num_cpus=-1,
    mid_times=None,
    verbose=0,
    logger=None,
):
    """Step 1. From all PET data, find a peripheral cluster.

    Loop over all values for k in ks, looking for clusters that
    exhibit peripheral-like properties. Return the best possible
    cluster.

    :param ndarray data: Array of timeseries
    :param tuple vol_shape: The original 3d shape of each column vector in data
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
        vol_shape,
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
                    [vascular_centroids[i].peak_value for i in earliest_peak_idxs]
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


def get_cluster_blobs(array_3d, label=1, max_gap=1, verbose=0):
    """Find connected blobs in array_3d"""

    _voxels_in_mask = []
    _blobs = {}
    voxels_added_by_scan = 0
    voxels_added_recursively = 0

    def add_voxel(loc):
        """for any voxel, find which blob it's in, then add it to the list"""
        nonlocal voxels_added_recursively

        if loc in _blobs:
            print("false alarm")
            return

        # First pass through the searchlight, are we near an existing blob?
        # If a nearby mask member is labeled, adopt this label
        still_looking, current_blob_id = True, None
        for _x in range(loc[0] - max_gap, loc[0] + max_gap + 1):
            for _y in range(loc[1] - max_gap, loc[1] + max_gap + 1):
                for _z in range(loc[2] - max_gap, loc[2] + max_gap + 1):
                    if (
                        still_looking
                        and ((_x, _y, _z) in _blobs)
                        and (_x >= 0)
                        and (_x < array_3d.shape[0])
                        and (_y >= 0)
                        and (_y < array_3d.shape[1])
                        and (_z >= 0)
                        and (_z < array_3d.shape[2])
                    ):
                        current_blob_id = _blobs[(_x, _y, _z)]
                        # We know our blob; we can stop cycling through
                        still_looking = False

        if still_looking:
            # No neighbors are yet recorded; this is a new blob
            if len(_blobs) == 0:
                max_blob = 0
            else:
                max_blob = np.max([v for k, v in _blobs.items()])
            current_blob_id = max_blob + 1
            # if verbose:
            #     print(f" new blob, #{current_blob_id}")

        # label the voxel we've been asked to add
        _blobs[loc] = current_blob_id

        # Second pass, label all in-mask voxels
        for _x in range(loc[0] - max_gap, loc[0] + max_gap + 1):
            for _y in range(loc[1] - max_gap, loc[1] + max_gap + 1):
                for _z in range(loc[2] - max_gap, loc[2] + max_gap + 1):
                    try:
                        if (
                            ((_x, _y, _z) not in _blobs)
                            and (array_3d[_x, _y, _z] == label)
                            and (_x >= 0)
                            and (_x < array_3d.shape[0])
                            and (_y >= 0)
                            and (_y < array_3d.shape[1])
                            and (_z >= 0)
                            and (_z < array_3d.shape[2])
                        ):
                            # _blobs[(_x, _y, _z)] = current_blob_id
                            # This voxel is in the mask, but not yet labeled
                            # expand outward, seeking more voxels within-blob
                            voxels_added_recursively += 1
                            add_voxel((_x, _y, _z))
                    except IndexError:
                        # No problem, we're searching beyond the array
                        # boundaries and don't need to look here anyway
                        voxels_added_recursively -= 1
                        pass
                    except RecursionError:
                        # We got pretty deep following this voxel's trail.
                        # Pick it up on the next one.
                        voxels_added_recursively -= 1
                        pass
        return  # from add_voxel, not get_cluster_blobs

    # Run through every voxel, adding it to the list if it's in the mask
    for x in range(array_3d.shape[0]):
        for y in range(array_3d.shape[1]):
            for z in range(array_3d.shape[2]):
                if array_3d[x, y, z] == label:
                    _voxels_in_mask.append((x, y, z))

    # Run through only in-mask voxels, adding them to a numbered blob.
    print(f"Label {label} has {len(_voxels_in_mask):,} voxels.")
    for x, y, z in _voxels_in_mask:
        if (x, y, z) not in _blobs:
            voxels_added_by_scan += 1
            # if verbose:
            #     print(f"Adding {voxels_added_by_scan}. ({x}, {y}, {z})")
            add_voxel((x, y, z))

    # All in-mask voxels have been added,
    # now organize them into a DataFrame for easy analyses
    blob_data = pd.DataFrame(
        [
            {
                "blob": blob,
                "gap": max_gap,
                "x": locus[0],
                "y": locus[1],
                "z": locus[2],
            }
            for locus, blob in _blobs.items()
        ]
    )
    blob_ids, voxel_counts = np.unique(blob_data["blob"], return_counts=True)
    if verbose > 0:
        print(
            f"Label {label}: {len(blob_ids):,} blobs "
            f"with {np.mean(voxel_counts):0,.1f} voxels each"
        )
    if verbose > 1:
        print(
            f"  found {len(blob_data):,} voxels, grouped them into "
            f"{len(blob_data['blob'].unique()):,} blobs "
            f"with max gap of {max_gap}."
        )
        print(
            f"  {voxels_added_by_scan:,} voxels were added while scanning, "
            f"{voxels_added_recursively:,} were added recursively."
        )

    return blob_data, blob_ids, voxel_counts
