import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans

from .centroid import Centroid


def likely_irreversible(c):
    """ Retrun true if centroid appears irreversible.

        :param Centroid c: The centroid to assess
        :return: True if irreversible, False otherwise
    """

    # If the highest value in the timeseries is the last one,
    # this voxel is likely irreversible
    return c.activity[-1] == max(c.activity)


def likely_noise(c):
    """ Retrun true if centroid appears to just be noise.

        :param Centroid c: The centroid to assess
        :return: True if noise, False otherwise
    """

    # If activity at any point after the first one is negative,
    # this voxel is likely noise
    return np.any(c.activity[1:] < 0)


def likely_vascular(c):
    """ Retrun true if centroid appears vascular.

        :param Centroid c: The centroid to assess
        :return: True if vascular, False otherwise
    """

    # If this centroid is reversible signal,
    # it is probably vascular
    return not likely_noise(c) and not likely_irreversible(c)


def likely_peripheral(c):
    """ Retrun true if centroid appears peripheral.

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


def find_centroids(data, ks, features, mid_times=None, verbose=0):
    """ Step 1. From all PET data, find a vascular cluster.

        Loop over all values for k in ks, looking for clusters that
        exhibit vascular-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means
        :param features: A dict of functions to assign features to centroids
        :param iterable mid_times: will be stored alongside activity in TACs
        :param int verbose: Set non-zero to increase logging, higher is more

        :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE")

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    k_means_fits = {}
    all_centroids = []
    for k in ks:
        logger.info(f"K-means (k={k})")
        pre_1k_timestamp = datetime.now()
        k_means = KMeans(init="k-means++", n_clusters=k,
                         n_init=3, max_iter=1024**2, random_state=42,
                         verbose=verbose, )
        k_means.fit(data)
        k_means_fits[k] = k_means
        post_1k_timestamp = datetime.now()
        logger.info(f"  lowest inertia == {k_means.inertia_:0.0f}"
                    f" after {k_means.n_iter_} iterations"
                    f" in {post_1k_timestamp - pre_1k_timestamp}.")

        # Count features for reporting, not necessary for execution
        feature_counts = {"total": 0}
        for feature_label in features.keys():
            feature_counts[feature_label] = 0

        # Find reasonable timeseries in the cluster means.
        # count_irreversible, count_noise = 0, 0
        for i in range(k_means.n_clusters):
            cc = k_means.cluster_centers_[i]
            this_centroid = Centroid(
                activity=cc,
                timepoints=mid_times,
                label=i + 1,  # should be non-zero as zero indicates background
                k=k,
                labels=k_means.labels_ + 1,
                name=f"centroid {i + 1}/{k}"
            )
            # Save features of this centroid, like whether it is
            # noise, vascular, peripheral, etc. using functions provided.
            for feature_label, fxn in features.items():
                this_centroid.features[feature_label] = fxn(this_centroid)
                if this_centroid.features[feature_label]:
                    feature_counts[feature_label] += 1
            feature_counts["total"] += 1

            all_centroids.append(this_centroid)

        for label, count in feature_counts.items():
            if label != "total":
                logger.debug(
                    f"  {count:03d} / {feature_counts['total']:03d} are {label}"
                )

    post_k_timestamp = datetime.now()
    logger.info(f"All {len(ks)} k-means finished in "
                f"{post_k_timestamp - pre_k_timestamp}")

    return all_centroids, k_means_fits


def find_vascular_centroids(data, ks, mid_times=None, verbose=0):
    """ Step 1. From all PET data, find a vascular cluster.

        Loop over all values for k in ks, looking for clusters that
        exhibit vascular-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means
        :param iterable mid_times: will be stored alongside activity in TACs
        :param int verbose: Set non-zero to increase logging, higher is more

        :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE")

    vascular_features = {
        "likely_noise": likely_noise,
        "likely_irreversible": likely_irreversible,
        "likely_vascular": likely_vascular,
    }
    all_centroids, k_means_fits = find_centroids(
        data, ks, vascular_features, mid_times=mid_times, verbose=verbose
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
            peak_idxs = np.array([c.peak_index for c in vascular_centroids])
            earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
            # Of the vascular centroids peaking at the same earliest time, which is highest?
            highest_early_peak_idx = earliest_peak_idxs[
                np.argmax([vascular_centroids[i].peak_value
                           for i in earliest_peak_idxs])
            ]
            # Label this centroid as best, at least for this value of k
            vascular_centroids[highest_early_peak_idx].best_in_k = True
            logger.debug(
                "  Best centroid [{}] has peak of {:0.3f} at time idx {}".format(
                    vascular_centroids[highest_early_peak_idx].label,
                    vascular_centroids[highest_early_peak_idx].peak_value,
                    vascular_centroids[highest_early_peak_idx].peak_index,
                )
            )

        plural_string = "" if len(vascular_centroids) == 1 else "s"
        logger.info(f"  found {len(vascular_centroids)} potential vascular"
                    f" cluster{plural_string} with k={k}.")

    # Which cluster-centroid timeseries has the highest peak?
    # And where is that peak?
    # from Matlab vascClust.m:160:174
    best_in_k_centroids = [c for c in all_centroids if c.best_in_k]
    top_indices, top_frequencies = np.unique(
        [c.peak_index for c in best_in_k_centroids], return_counts=True
    )
    # Which time point is most likely to have the highest value?
    best_centroid_idx = top_indices[np.argmax(top_frequencies)]

    # Make a list of best-in-k centroids that peak at the same, most common, time point
    centroids_with_best_idx = [
        c for c in best_in_k_centroids
        if (c.peak_index == best_centroid_idx)
    ]
    # Of those centroids peaking together, which one peaks highest?
    best_centroid = centroids_with_best_idx[
        np.argmax([c.peak_value for c in centroids_with_best_idx])
    ]
    # Label the centroid with the highest peak value
    best_centroid.best_overall = True
    logger.info(f"The very best cluster is label {best_centroid.label} "
                f"from k {best_centroid.k}.")
    logger.info(f"It peaked at frame {best_centroid.peak_index + 1} "
                f"to a value of {best_centroid.peak_value}.")

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits


def find_peripheral_centroids(data, ks, mid_times=None, verbose=0):
    """ Step 1. From all PET data, find a peripheral cluster.

        Loop over all values for k in ks, looking for clusters that
        exhibit peripheral-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means
        :param iterable mid_times: will be stored alongside activity in TACs
        :param int verbose: Set non-zero to increase logging, higher is more

        :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE")

    vascular_features = {
        "likely_noise": likely_noise,
        "likely_irreversible": likely_irreversible,
        "likely_vascular": likely_vascular,
    }
    all_centroids, k_means_fits = find_centroids(
        data, ks, vascular_features, mid_times=mid_times, verbose=verbose
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
                np.argmax([vascular_centroids[i].peak_value
                           for i in earliest_peak_idxs])
            ]
            # Label this centroid as best, at least for this value of k
            vascular_centroids[highest_early_peak_idx].best_in_k = True
            logger.debug(
                "  Best centroid [{}] has peak of {:0.3f} at time idx {}".format(
                    vascular_centroids[highest_early_peak_idx].label,
                    vascular_centroids[highest_early_peak_idx].peak_value,
                    vascular_centroids[highest_early_peak_idx].peak_index,
                )
            )

        plural_string = "" if len(vascular_centroids) == 1 else "s"
        logger.info(f"  found {len(vascular_centroids)} potential vascular"
                    f" cluster{plural_string} with k={k}.")

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
        c for c in all_centroids
        if ((c.peak_index == best_centroid_idx) and c.best_in_k)
    ]
    # Of those centroids peaking together, which one peaks highest?
    best_centroid = centroids_with_best_idx[
        np.argmax([c.peak_value for c in centroids_with_best_idx])
    ]
    # Label the centroid with the highest peak value
    best_centroid.best_overall = True
    logger.info(f"The very best cluster is label {best_centroid.label} "
                f"from k {best_centroid.k}.")
    logger.info(f"It peaked at frame {best_centroid.peak_index + 1} "
                f"to a value of {best_centroid.peak_value}.")

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits
