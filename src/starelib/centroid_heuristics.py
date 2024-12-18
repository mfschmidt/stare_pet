import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans
import pandas as pd

from .centroid import Centroid
from .mp_queues import run_in_mp_queue
from .util import (
    dice_coef, flatten_4d_to_2d, reshape_labels_to_3d, get_s_i_axis
)


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
    return (
        not likely_noise(c) and
        not likely_irreversible(c) and
        not likely_background(c)
    )


def likely_background(c):
    """Return true if centroid appears to be background.

    :param Centroid c: The centroid to assess
    :return: True if background, False otherwise
    """

    # If this centroid is larger than 1/5 of the image,
    # it is much more likely to be non-brain background than vascular.
    if isinstance(c, Centroid):
        print(f"C {c.label}/{c.k} has {c.voxel_count:,}/{c.voxels_in_img:,} voxels.")
        return c.voxel_count > c.voxels_in_img / 5.0
    else:
        return False


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


def calculate_k_stability(centroids, similarity_matrix):
    """ Is a variant of the 'best' centroid found at multiple k's?
    """

    unique_k_values = sorted(set([c.k for c in centroids]))

    # Let each centroid pick out its relevant coefficients and summarize them.
    for c1 in centroids:
        c1_idx = f"{c1.label:02d}-{c1.k:02d}"
        c1.features['stability'] = 'Not implemented'
        c1.features['overlapping'] = list()
        c1.features['matches_best'] = 0.0
        best_matches = list()
        top_dices = list()
        all_dices = dict()
        for k in unique_k_values:
            if c1.k != k:
                all_dices[k] = list()
        for c2 in centroids:
            c2_idx = f"{c2.label:02d}-{c2.k:02d}"
            dc = similarity_matrix.loc[c1_idx, c2_idx]
            if c1.k != c2.k:
                all_dices[c2.k].append(dc)
                if dc > 0.50:
                    c1.features['overlapping'].append(c2_idx)
            if c2.best_in_k:
                best_matches.append(dc)
        for k, dices in all_dices.items():
            if len(dices) > 0:
                top_dices.append(max(dices))
            else:
                top_dices.append(0.0)
        if len(top_dices) > 0:
            c1.features['stability'] = np.mean(top_dices)
        else:
            c1.features['stability'] = 0.0
        if len(best_matches) > 0:
            c1.features['matches_best'] = np.mean(best_matches)
        else:
            c1.features['matches_best'] = 0.0


def calculate_sparsity(centroids, threshold=95):
    """ With each centroid, calculate sparsity of blob distribution.

    :param centroids: List of centroids
    :param threshold: Sparsity threshold, and integer between 0 and 100
    :returns: nothing; centroids are updated internally
    """

    real_threshold = threshold / 100.0
    for c in centroids:
        counts = c.blob_data.groupby("blob")['blob'].agg('count').sort_values(
            ascending=False
        )
        blobs_consumed, voxels_consumed = 0, 0
        for idx, voxels in counts.items():
            blobs_consumed += 1
            voxels_consumed += voxels
            ratio = voxels_consumed / c.voxel_count
            if ratio > real_threshold:
                c.features[f"{threshold}_in_blobs"] = blobs_consumed
                c.features[f"{threshold}_in_voxels"] = voxels_consumed
                c.features[f"{threshold}_out_blobs"] = c.blob_count - blobs_consumed
                c.features[f"{threshold}_out_voxels"] = c.voxel_count - voxels_consumed
                print(f"First {blobs_consumed} used {voxels_consumed} voxels "
                      f"({ratio:0.1%}) - {c.voxel_count - voxels_consumed} voxels "
                      f"remain in {c.blob_count - blobs_consumed} blobs.")
                break


def calculate_axis_weights(centroids):
    """ How heavily weighted are the cluster masks along x, y, z axes?
    """

    # Collapse axes of each centroid to determine how heavily a single
    # slice is represented by this cluster mask in each dimension
    for c in centroids:
        img_shape = c.original_shape
        img_affine = c.original_affine
        ax, direction = get_s_i_axis(img_shape, img_affine)

        # Figure out which slice is the lowest axial neck slice.
        mask = (reshape_labels_to_3d(c.labels, c.original_shape) == c.label)
        if ax == 2:
            # feature_name = 'k_density'
            density = np.sum(np.sum(mask, axis=0), axis=0)
        elif ax == 1:
            # feature_name = 'j_density'
            density = np.sum(np.sum(mask, axis=0), axis=1)
        else:  # ax == 0
            # feature_name = 'i_density'
            density = np.sum(np.sum(mask, axis=1), axis=1)

        if direction > 0.0:
            start, stop = 0.0, len(density)
        else:
            start, stop = len(density), 0.0

        """
        # labels already had +1, so they're 1-indexed
        # We don't have the affine and are ignorant of real-world directions,
        # so avoid world x,y,z and use array i,j,k
        # Collapse i, leaving [j,k]
        jk_sums_2d = np.sum(mask, axis=0)
        # Collapse j, leaving only [k]
        c.features['k_density'] = np.sum(jk_sums_2d, axis=0)
        # Collapse k, leaving only [j]
        c.features['j_density'] = np.sum(jk_sums_2d, axis=1)
        # Collapse j, leaving [i, k]
        ik_sums_2d = np.sum(mask, axis=1)
        # Collapse k, leaving only [i]
        c.features['i_density'] = np.sum(ik_sums_2d, axis=1)
        """

        # Calculate a score to indicate the cluster is dominated by neck noise.
        # These weights roughly add the proportion of voxels in the lowest 12 slices,
        # and ignore everything else. So a score of 0.50 would indicate that about
        # half the voxels are in the bottom 12 axial slices.
        def custom_inverse_sigmoid(x):
            return 1.0 - (1 / (1 + 3 ** (-1 * (x - 12)))) ** (1 / 2)
        weights = custom_inverse_sigmoid(np.linspace(start, stop, len(density)))
        ratios = density / np.sum(density)
        # ratios = ratios - np.mean(ratios) * np.std(ratios)
        c.features['neck_noise_score'] = np.sum(ratios * weights)


def calculate_spatial_info(centroids, step, logger, num_cpus=-1):
    """ Calculate centroid stats.
    """

    logger.debug(
        f"Analyzing spatial clusters for step {step}"
    )
    centroid_tuples = [
        (str(c), c) for c in centroids
    ]
    # Do spatial info calculations in as many threads as we can.
    list_of_results = run_in_mp_queue(
        spatial_info_worker,
        centroid_tuples,
        num_cpus=num_cpus,
        logger=logger,
    )
    # Multi-processing works on copies of the original data, so return
    # updated data, losing our original centroids
    returned_centroids = list()
    for rslt in list_of_results:
        returned_centroids.append(rslt['c'])
        for msg in rslt['log_messages']:
            logger.info(msg)
    return returned_centroids


def spatial_info_worker(arg_tuple):
    """ A worker function to calculate spatial information for centroids

        This worker can be launched in a separate process to calculate
        spatial information on a cluster/centroid.
    """

    (desc, c) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting spatial information worker for centroid {str(c)} "
          f"at {worker_start.strftime('%m/%d %I:%M')}", flush=True)
    log_messages = list()
    c.update_spatial_clusters(message_list=log_messages, verbose=True)

    if c.blob_data is None:
        print(f"{str(c)} just finished update_spatial_clusters; still no blob_data!!!")
    else:
        print(f"{str(c)} got {c.blob_data.shape}-shaped blob_data.")
    log_messages.append(
        f"  For {str(c)}, found {c.blob_count} blobs with "
        f"{c.voxels_per_blob:0.2f} voxels each in "
        f"{datetime.now() - worker_start}."
    )
    worker_end = datetime.now()
    print(f"    Finished spatial information worker for centroid {str(c)} "
          f"at {worker_end.strftime('%m/%d %I:%M')} "
          f"after {worker_end - worker_start}.", flush=True)

    return {
        "c": c,
        "log_messages": log_messages,
    }


def k_means_worker(arg_tuple):
    """ A worker function to calculate k-means for one k

        This worker can be launched in a separate process to calculate k
        clusters, and save k-means results in a returnable dict.

    """

    # Workers get a single argument, so the caller must pack arguments
    # into a tuple and the worker (this function) must unpack them.
    # This order must match exactly the order where they're packed.
    (desc, k, data, random_seed, verbose) = arg_tuple

    worker_start = datetime.now()
    print(f"    Starting k-means worker for k={k} "
          f"at {worker_start.strftime('%m/%d %I:%M')}", flush=True)
    log_messages = []

    k_means = KMeans(
        init="k-means++",
        n_clusters=k,
        n_init=3,
        max_iter=1024 ** 2,
        random_state=random_seed,
        verbose=verbose,
    )
    k_means.fit(data)

    log_messages.append(
        f"  data mean {np.mean(data):0.2f}, sd {np.std(data):0.2f}\n"
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
        pet_4d_img,
        ks,
        features,
        mid_times=None,
        num_cpus=1,
        verbose=0,
        random_seed=42,
        logger=None,
):
    """Step 1. From all PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray pet_4d_img: Nifti with 3D Array of timeseries, 4D overall
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param features: A dict of functions to assign features to centroids
    :param iterable mid_times: will be stored alongside activity in TACs
    :param num_cpus: How many CPUs to deploy on multiprocessing
    :param int verbose: Set non-zero to increase logging, higher is more
    :param int random_seed: Allow setting the random seed, if desired
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger
    logger.info(f"Setting up {len(ks)} K-means values across {num_cpus} cpus.")

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    list_of_args = []
    for k in ks:
        list_of_args.append(
            (f"k {k}", k, flatten_4d_to_2d(pet_4d_img.get_fdata(), zxy=True),
             random_seed, verbose)
        )
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
                original_shape=pet_4d_img.shape,
                original_affine=pet_4d_img.affine,
                voxel_count=np.sum(kmeans_result['k_means'].labels_ == i),
                voxels_in_img=len(kmeans_result['k_means'].labels_),
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


def label_best_centroid(centroids, best_label="best_overall"):
    """ From a list of centroids, go through and label the best. """

    best_centroid = None
    if best_label == 'best_in_k':
        # Select the 'best' from among all vascular centroids.
        # Do this by finding the earliest peak, then among centroids
        # peaking there, find the highest. "highest-of-earliest"
        peak_idxs = np.array([c.peak_index for c in centroids])
        earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
        highest_early_peak_idx = earliest_peak_idxs[np.argmax([
            centroids[i].peak_value for i in earliest_peak_idxs
        ])]

        # We once tried "earliest-of-highest", but it didn't work as well.
        """
        peak_vals = np.array([c.peak_value for c in centroids])
        highest_peak_idxs = np.where(peak_vals == np.max(peak_vals))[0]
        earliest_high_peak_idx = highest_peak_idxs[np.argmin([
            centroids[i].peak_value for i in highest_peak_idxs
        ])]
        """

        best_centroid = centroids[highest_early_peak_idx]
        best_centroid.best_in_k = True

    elif best_label == 'best_overall':
        top_indices, top_frequencies = np.unique(
            [c.peak_index for c in centroids], return_counts=True
        )

        # This is the most likely time point to have the best vascular peak,
        # but it is only about 90% accurate in our tests. So we'll also consider
        # the next time point, but only if it has both a higher peak than our
        # current best centroid and a more spatially concise clustering.
        if len(top_frequencies) == 0:
            raise TypeError(f"None of the {len(centroids)} clusters appear "
                            "vascular. There's nothing more to be done.")
        best_centroid_idx = top_indices[np.argmax(top_frequencies)]

        # Make a list of best-in-k centroids that peak at the same,
        # most common, time point
        centroids_with_best_idx = [
            c for c in centroids if (c.peak_index == best_centroid_idx)
        ]
        # Of those centroids peaking together, which one peaks highest?
        best_centroid = centroids_with_best_idx[
            np.argmax([c.peak_value for c in centroids_with_best_idx])
        ]
        best_centroid.best_overall = True

    return best_centroid


def find_vascular_centroids(
        pet_4d_img,
        ks,
        step,
        mid_times=None,
        num_cpus=1,
        verbose=0,
        logger=None,
):
    """ From PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for k-means clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray pet_4d_img: Nifti with 3D array of timeseries, 4D overall
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param int step: Which step of k-means, used for labeling centroids.
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
        "likely_background": likely_background,
    }
    all_centroids, k_means_fits = find_centroids(
        pet_4d_img,
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
            if centroid.features.get("likely_vascular", False):
                vascular_centroids.append(centroid)
            else:
                other_centroids.append(centroid)

        for i, vc in enumerate(vascular_centroids):
            logger.debug(f"  {vc.peak_value:0.3f} at {vc.peak_index}")

        # Label the top candidate for a vascular cluster from this k value.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous
        if len(vascular_centroids) > 0:
            top_c = label_best_centroid(vascular_centroids, 'best_in_k')
            logger.debug(
                "  Early centroid [{}/{}] has peak of {:0.3f} at t {}".format(
                    top_c.label, top_c.k, top_c.peak_value, top_c.peak_index,
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
    for c in best_in_k_centroids:
        c.name = f"Best step {step}. {c.name}"
    label_best_centroid(best_in_k_centroids, 'best_overall')

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits


def build_similarity(centroids, order_by=None, logger=None):
    """ Calculate Dice similarity between centroids
    """

    num_dices_calculated = 0
    num_dices_duped = 0
    num_diagonals = 0
    mat_len = len(centroids)
    dice_matrix = np.zeros((mat_len, mat_len))
    row_index = list()
    col_index = list()

    for i, c1 in enumerate(centroids):
        row_index.append(f"{c1.label:02d}-{c1.k:02d}")
        for j, c2 in enumerate(centroids):
            if i == 0:
                col_index.append(f"{c2.label:02d}-{c2.k:02d}")
            if str(c1) == str(c2):
                dice_matrix[i, j] = 1.0
                num_diagonals += 1
            elif dice_matrix[i, j] != 0.0:
                dice_matrix[j, i] = dice_matrix[i, j]
                num_dices_duped += 1
            elif dice_matrix[j, i] != 0.0:
                dice_matrix[i, j] = dice_matrix[j, i]
                num_dices_duped += 1
            else:
                dice_matrix[i, j] = dice_coef(
                    np.ravel(c1.labels == c1.label),
                    np.ravel(c2.labels == c2.label),
                )
                # dice_matrix[i, j] = dice_matrix[j, i]
                num_dices_calculated += 1
    _msg = (f"Cluster similarity: calculated {num_dices_calculated:,} / "
            f"{len(centroids) ** 2:,}, duped {num_dices_duped:,}, "
            f"{num_diagonals:,} diagonals.")
    if logger is None:
        print(_msg)
    else:
        logger.info(_msg)

    return pd.DataFrame(dice_matrix, index=row_index, columns=col_index)


def consider_alternate_clusters(
        centroids, k_means_fits, source_4d_image, verbose=0, logger=None
):
    """ After selecting the best cluster, look again for a better alternate.

        We probably have the best centroid, but if the next time point contains
        a centroid with a higher value AND a more spatially concise clustering,
        we should consider the runner-up time point a better bet. Even if that
        centroid was not 'best_in_k', because 'best_in_k' was also
        restricted to this same earliest peak. We explicitly want to see
        if the peak being too early caused us to miss a better option here.

        :param list centroids: List of centroids
        :param list k_means_fits: List of fitted K-means calculations
        :param Nifti1Image source_4d_image: The image used to compute the KMeans
        :param int verbose: How verbose should we be with our output?
        :param logger: Logger instance
        :return: True if we changed the best cluster, False otherwise
    """

    logger = logging.getLogger("STARE") if logger is None else logger
    html_lines = list()

    # Before beginning, locate the current selection of "best" centroid.
    # While doing this, count vascular centroids for a narrative report.
    first_choice_centroid = None
    vascular_centroids = list()
    num_vascular_total = 0
    num_vascular_per_k = dict()
    for c in centroids:
        if c.features.get("likely_vascular", False):
            vascular_centroids.append(c)
            num_vascular_total += 1
            if c.k in num_vascular_per_k.keys():
                num_vascular_per_k[c.k] += 1
            else:
                num_vascular_per_k[c.k] = 1
        c.original_shape = source_4d_image.shape
        if c.best_overall:
            first_choice_centroid = c
    if first_choice_centroid is None:
        raise ValueError("No centroid is selected as 'best'.")

    html_lines.append(f"Overall, {num_vascular_total:,} of {len(centroids):,} "
                      f"centroids are likely to be vascular.")
    html_lines.append("There are " + ", ".join([
        f"{v} from k={k}" for k, v in sorted(num_vascular_per_k.items())
    ]) + ".")

    # Figure out how many alternatives we really have to choose from,
    # and report the narrative. We don't do anything with it, yet.
    for delay in (0, 1, 2, ):
        html_table = list()
        peak_idx = first_choice_centroid.peak_index + delay
        peak_t = first_choice_centroid.timepoints[peak_idx]
        centroids_at_peak = [
            c for c in vascular_centroids if c.peak_index == peak_idx
        ]
        if len(centroids_at_peak) < 1:
            # Hopefully, this will prevent extra empty tables for step two.
            break
        elif len(centroids_at_peak) == 1:
            plural_str, conj_str = "", "s"
        else:
            plural_str, conj_str = "s", ""
        html_table.append("<table>")
        html_table.append("<thead>")
        html_table.append(f"<tr><th colspan=\"9\">{len(centroids_at_peak)} "
                          f"centroid{plural_str} peak{conj_str} at index "
                          f"{peak_idx} (best + {delay}, t = {peak_t:0.3f})"
                          "</th></tr>")
        html_table.append(
            "<tr>"
            "<th>Cluster</th>"
            "<th>Peak</th>"
            "<th>Stability across Ks</th>"
            "<th>Clusters with >50% overlap</th>"
            "<th>Similarity to best</th>"
            "<th>Neck noise</th>"
            "<th>Sparsity</th>"
            "<th>Voxels</th>"
            "<th>Notes</th>"
            "</tr>"
        )
        html_table.append("</thead>")
        html_table.append("<tbody>")
        for c in sorted(
                centroids_at_peak, key=lambda pv: pv.peak_value, reverse=True
        ):
            stability_str = "n/a"
            if 'stability' in c.features:
                stability_str = f"{c.features['stability']:0.1%}"
            overlap_str = "n/a"
            if 'overlapping' in c.features:
                overlap_str = f"{len(c.features['overlapping'])}"
            match_best_str = "n/a"
            if 'matches_best' in c.features:
                match_best_str = f"{c.features['matches_best']:0.1%}"
            neck_noise_str = "n/a"
            if 'neck_noise_score' in c.features:
                neck_noise_str = f"{c.features['neck_noise_score']:0.2f}"
            reduced_str = "Δ" if c.source == "sparsity reduction" else ""
            html_table.append(
                "<tr>"
                f"<td>{c.label:02d}/{c.k:02d}</td>"
                f"<td>{c.peak_value:0.4f}</td>"
                f"<td>{stability_str}</td>"
                f"<td>{overlap_str}</td>"
                f"<td>{match_best_str}</td>"
                f"<td>{neck_noise_str}</td>"
                f"<td>{c.sparsity}</td>"
                f"<td>{c.voxel_count:,}</td>"
                f"<td>{'*' if c.best_in_k else ''}"
                f"{'+' if c.best_overall else ''}{reduced_str}</td>"
                "</tr>"
            )
        html_table.append("</tbody>")
        html_table.append("<tfoot>")
        html_table.append("<tr><th colspan=\"9\">* indicates best-of-k; "
                          "+ indicates best-overall; "
                          "Δ indicates sparsity-reduction.<br />"
                          "'Similarity to best' indicates how similar a "
                          "given cluster is to the best clusters from other"
                          " ks.</th></tr>")
        html_table.append("</tfoot>")
        html_table.append("</table>")
        html_lines.append("\n".join(html_table))

    # The best centroid has the highest peak out of all centroids peaking
    # at the same, earliest time. For candidates to replace it,
    # only consider peaks at one time point later.
    centroids_with_alt_idx = [
        c for c in vascular_centroids
        if ((c.peak_index == first_choice_centroid.peak_index + 1) &
            (c.features['likely_vascular']))
    ]

    # Of those centroids peaking together, which one peaks highest?
    if len(centroids_with_alt_idx) > 0:
        alt_centroid = centroids_with_alt_idx[
            np.argmax([c.peak_value for c in centroids_with_alt_idx])
        ]
        if alt_centroid.peak_value > first_choice_centroid.peak_value:
            # For speed, we opted not to calculate spatial clustering on
            # every cluster; but we must do this now if we want to compare them.
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
                html_lines.append(
                    f"Overriding the cluster selection with an alternate!!"
                    f" original best {first_choice_centroid.description()};"
                    f" new best is {alt_centroid.description()}."
                )
                """ TODO: In this version, we don't actually override yet.
                first_choice_centroid.name = " ".join([
                    "Original", first_choice_centroid.name,
                ])
                first_choice_centroid.best_overall = False
                alt_centroid.name = " ".join([
                    "Best by override.", alt_centroid.name,
                ])
                alt_centroid.best_overall = True
                """
                return html_lines
            else:
                logger.info(f"An alternate, {alt_centroid.description()}, "
                            "was considered and dropped.")
        else:
            logger.info("No alternate centroids had higher peaks.")
    else:
        logger.info("No alternate centroids were considered.")

    return html_lines


def find_peripheral_centroids(
        pet_4d_img,
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

    :param ndarray pet_4d_img: Nifti containing timeseries
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
        pet_4d_img,
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
            if centroid.features.get("likely_vascular", False):
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
