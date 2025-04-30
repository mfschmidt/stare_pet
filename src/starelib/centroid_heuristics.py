import numpy as np
import logging
from datetime import datetime
from sklearn.cluster import KMeans
import pandas as pd
import nilearn as nil
import nibabel as nib
from nibabel import affines
from scipy.ndimage import center_of_mass
from scipy.spatial import distance
from nilearn.image import coord_transform

from .centroid import Centroid
from .mp_queues import run_in_mp_queue
from .util import (
    dice_coef, flatten_4d_to_2d, unflatten_2d_to_4d, reshape_labels_to_3d,
    get_s_i_axis, get_s_i_density
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

    # If activity at any point after the first two are negative,
    # this voxel is likely noise. This used to be after the first one,
    # but we found a subject with a negative value at the second time
    # point and a perfectly good TAC otherwise.
    return np.any(c.activity[2:] < 0.0)


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
        # print(f"C {c.label}/{c.k} has {c.voxel_count:,}/{c.voxels_in_img:,} voxels.")
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


def calculate_old_confetti_score(mask, pet_img):
    """ Score how likely the mask is to represent confetti-like clusters.

        A positive score indicates a high likelihood of confetti-like clusters.
        A negative score indicates a low likelihood of confetti-like clusters.

        :param mask: A 3D-shaped cluster mask
        :param pet_img: Nifti PET image for shape and affine, not data
    """

    # All centroids were from the same image; which axis points inferiorly?
    ax, direction = get_s_i_axis(pet_img.shape[:3], pet_img.affine)
    density = get_s_i_density(pet_img, mask)
    ratios = density / np.sum(density)

    # Calculate a score to indicate the cluster is dominated by neck noise.
    num_penalized_slices = int(len(ratios) / 4.0)
    weights = np.ones(len(density)) * -1.0
    # This weight scheme was calibrated to maximize the confetti-like
    # clusters in CerePET scans to be >0 and the good clusters to be <0
    leading_weights = [3.0 / (2**(i / 3.0)) for i in range(num_penalized_slices)]
    if direction > 0.0:
        # start, stop = 0.0, len(density)
        penalized_idx = list(range(num_penalized_slices))
    else:
        # start, stop = len(density), 0.0
        penalized_idx = [(i + 1) * -1 for i in range(num_penalized_slices)]
    weights[penalized_idx] = np.array(leading_weights)

    # These weights add the proportion of voxels in the lowest slices,
    # and penalize everything in head space. So a score of 1.0 would
    # indicate very heavy influence of voxels in the bottom-most slice.
    # A good cluster will probably have a negative score.
    return np.sum(ratios * weights)


## Experimental modifications of the above
def calculate_confetti_score(mask, pet_img):
    """ Score how likely the mask is to represent confetti-like clusters.

        A positive score indicates a high likelihood of confetti-like clusters.
        A negative score indicates a low likelihood of confetti-like clusters.

        :param mask: A 3D-shaped cluster mask
        :param pet_img: Nifti PET image for shape and affine, not data
    """

    # All centroids were from the same image; which axis points inferiorly?
    axis, direction = get_s_i_axis(pet_img.shape[:3], pet_img.affine)
    density = get_s_i_density(pet_img, mask)
    ratios = density / np.sum(density)
    com_vox = center_of_mass(pet_img.get_fdata())
    com_world = coord_transform(*com_vox, pet_img.affine)
    # print(f"Center of mass in world space: [{com_world[0]:.1f}, {com_world[1]:.1f}, {com_world[2]:.1f}]")
    # print(f"Center of mass in voxel space: [{com_vox[0]:.1f}, {com_vox[1]:.1f}, {com_vox[2]:.1f}]")

    # This weight scheme was calibrated to maximize the confetti-like
    # clusters in CerePET scans to be >0 and the good clusters to be <0
    # It is based on empirical manual measurements of where good clusters
    # lie along the z axis and where noise typically starts in CerePET images.
    # The top of the superior sagittal sinus (and brain) is 53-70 slices (63.6-84.0mm) above the COM
    # The bottom of the transverse sinus is up to 28 slices (33.6mm) below the COM.
    # Vascular clusters are very likely to be in this range,
    # and confetti is not, so include slices liberally, and penalize it strongly.
    hi_vasc_z = com_world[axis] + 84.0  # mm, world space
    lo_vasc_z = com_world[axis] - 33.6  # mm, world space
    # print(f"Vasc range in world space: [{lo_vasc_z:.1f}, {hi_vasc_z:.1f}]")
    hi_vasc_z = int(coord_transform(0, 0, hi_vasc_z, np.linalg.inv(pet_img.affine))[axis])
    lo_vasc_z = int(coord_transform(0, 0, lo_vasc_z, np.linalg.inv(pet_img.affine))[axis])
    # print(f"Vasc range in voxel space: [{lo_vasc_z}, {hi_vasc_z}]")

    # Confetti noise is pretty universal in the lower 12 slices, regardless of
    # positioning. And it's common in 44.6 to 84.9 slices (53.5-101.9mm) below the COM. Because
    # inferior slices may still contain carotid artery or jugular vein, and
    # because this noise may co-occur in a mask that also contains vascular
    # clusters, we are more conservative with inclusion, and less punitive
    # with scoring. We also use a gradient to score more inferior slices more
    # heavily. We will additively penalize voxels in the cluster for being below
    # each of the three thresholds below, so the bottom slice usually gets a triple whammy.
    hi_confetti_slice = 10  # scaled back from 12 to stay conservative
    universal_noise_z = int(coord_transform(0, 0, hi_confetti_slice, pet_img.affine)[axis])
    # print(f"Absolute noise range below {hi_confetti_slice} slices, {universal_noise_z:.1f}mm")
    sup_noise_z = com_world[axis] - 31.9
    inf_noise_z = com_world[axis] - 67.7
    # print(f"Noise range in world space: weak below {sup_noise_z:.1f}, strong below {inf_noise_z:.1f}")
    sup_noise_z = int(coord_transform(0, 0, sup_noise_z, np.linalg.inv(pet_img.affine))[axis])
    inf_noise_z = int(coord_transform(0, 0, inf_noise_z, np.linalg.inv(pet_img.affine))[axis])
    # print(f"Noise range in voxel space: weak below {sup_noise_z:.1f}, strong below {inf_noise_z:.1f}")

    # Construct a confetti score mask from low neck to above the head
    df_rows = list()
    weights = np.zeros(len(ratios))
    for i in range(len(weights)):
    # Check both directions because S+ encoding not 100% certain.
        if direction > 0.0:  # inferior to superior
            # start, stop = 0.0, len(density)
            if hi_vasc_z >= i >= lo_vasc_z:
                weights[i] -= 5.0
            if i < sup_noise_z:
                weights[i] += 0.5
            if i < inf_noise_z:
                weights[i] += 1.0
            if i < universal_noise_z:
                weights[i] += 1.0
        else:  # superior to inferior, unlikely
            # start, stop = len(density), 0.0
            if hi_vasc_z <= i <= lo_vasc_z:
                weights[i] -= 5.0
            if i > sup_noise_z:
                weights[i] += 0.5
            if i > inf_noise_z:
                weights[i] += 1.0
            if i > universal_noise_z:
                weights[i] += 1.0
        valence = 'neutral'
        score = ratios[i] * weights[i]
        if score > 0.0:
            valence = 'positive'
        elif score < 0.0:
            valence = 'negative'
        df_rows.append({'z': i, 'var': 'weight', 'val': weights[i], 'valence': valence})
        df_rows.append({'z': i, 'var': 'ratio', 'val': ratios[i], 'valence': valence})
        df_rows.append({'z': i, 'var': 'score', 'val': score, 'valence': valence})

    plottable_data = pd.DataFrame(df_rows)

    # These weights add the proportion of voxels in the lowest slices,
    # and penalize everything in head space. So a score of 1.0 would
    # indicate very heavy influence of voxels in the bottom-most slice.
    # A good cluster will probably have a negative score.
    return {
        'ratios': ratios,
        'weights': weights,
        'score': np.sum(ratios * weights),
        'data': plottable_data,
    }


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
        mask_img,
        ks,
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
    :param ndarray mask_img: Nifti with 3D Array of bool to mask pet_4d_img
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param iterable mid_times: will be stored alongside activity in TACs
    :param num_cpus: How many CPUs to deploy on multiprocessing
    :param int verbose: Set non-zero to increase logging, higher is more
    :param int random_seed: Allow setting the random seed, if desired
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """
    # TODO: the mask already IS the masked 4D data; we could just use it. Or we could redo these calculations on a proper boolean mask; Maybe just do it how we used to, but provide the original 4D for COM calculations
    logger = logging.getLogger("STARE") if logger is None else logger
    logger.info(f"Setting up {len(ks)} K-means values across {num_cpus} cpus.")

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    # For step one, there's no mask, just the whole image.
    # For step two, we mask in only the step one data.

    list_of_args = []
    # We have a full image and a mask, cluster with masked data
    if mask_img is None:
        clusterable_data = flatten_4d_to_2d(pet_4d_img.get_fdata(), zxy=True)
    else:
        logger.info(f"masking {pet_4d_img.shape} PET data by {mask_img.shape} mask")
        mask = np.sum(flatten_4d_to_2d(mask_img.get_fdata().astype(bool), zxy=True), axis=1).astype(bool)
        pet_2d_data = flatten_4d_to_2d(pet_4d_img.get_fdata(), zxy=True)
        best_centroid_masked_data = np.zeros(pet_2d_data.shape)
        best_centroid_masked_data[mask] = pet_2d_data[mask, :]
        img_to_cluster = nib.nifti1.Nifti1Image(
            unflatten_2d_to_4d(
                best_centroid_masked_data,
                pet_4d_img.shape
            ),
            affine=pet_4d_img.affine,
        )
        clusterable_data = flatten_4d_to_2d(
            img_to_cluster.get_fdata(), zxy=True,
        )
    # Run each tuple of arguments in a separate process to save time.
    for k in ks:
        list_of_args.append(
            (f"k {k}", k, clusterable_data, random_seed, verbose)
        )
    k_means_results = run_in_mp_queue(
        k_means_worker, list_of_args, num_cpus, logger
    )

    # Calculate underlying image's center of mass for all clusters
    pet_mean_img = nil.image.mean_img(pet_4d_img, copy_header=True)
    pet_vox_com = center_of_mass(pet_mean_img.get_fdata())
    pet_world_com = affines.apply_affine(pet_mean_img.affine, pet_vox_com)

    # Retrieve and organize k-means results
    k_means_fits = {}
    all_centroids = []
    for kmeans_result in k_means_results:
        k = kmeans_result['k']
        k_means_fits[k] = kmeans_result['k_means']

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
                # labels=kmeans_result['k_means'].labels_ + 1,
                # blob_count=len(blob_ids),
                # voxels_per_blob=np.mean(voxel_counts),
            )
            this_centroid.features = dict()

            this_centroid.features["likely_noise"] = likely_noise(this_centroid)
            this_centroid.features["likely_irreversible"] = likely_irreversible(this_centroid)
            this_centroid.features["likely_irreversible_linear"] = likely_irreversible_linear(this_centroid)
            this_centroid.features["likely_background"] = likely_background(this_centroid)
            this_centroid.features["likely_vascular"] = likely_vascular(this_centroid)

            # Add confetti score feature
            mask_3d = reshape_labels_to_3d(
                np.array(kmeans_result['k_means'].labels_ == i, dtype=np.uint8),
                pet_mean_img.shape,
            )  # TODO: Cache the 3D and 2D versions of this someplace to avoid reordering them. This takes way too long, especially in debug!
            confetti_dict = calculate_confetti_score(
                mask_3d, pet_mean_img
            )
            this_centroid.features["confetti_data"] = confetti_dict['data']
            this_centroid.features["confetti_score"] = confetti_dict['score']
            this_centroid.features['likely_confetti'] = \
                this_centroid.features['confetti_score'] > 0.0

            # Add centroid and underlying image's center of mass as features
            this_centroid.features["pet_com_x"] = float(pet_world_com[0])
            this_centroid.features["pet_com_y"] = float(pet_world_com[1])
            this_centroid.features["pet_com_z"] = float(pet_world_com[2])
            vox_com = center_of_mass(mask_3d)
            world_com = affines.apply_affine(pet_mean_img.affine, vox_com)
            diff_com = np.subtract(world_com, pet_world_com)
            this_centroid.features["com_x"] = float(world_com[0])
            this_centroid.features["com_y"] = float(world_com[1])
            this_centroid.features["com_z"] = float(world_com[2])
            this_centroid.features["com_shift_x"] = diff_com[0]
            this_centroid.features["com_shift_y"] = diff_com[1]
            this_centroid.features["com_shift_z"] = diff_com[2]
            this_centroid.features["com_shift"] = distance.euclidean(
                pet_world_com, world_com
            )

            # If used, clear the huge labels matrix to avoid duplicating info
            # this_centroid.labels = None

            all_centroids.append(this_centroid)

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


def label_best_centroid(centroids, best_label="best_overall", keep_confetti=False):
    """ From a list of centroids, go through and label the best. """

    logger = logging.getLogger("STARE")

    best_centroid = None
    if not keep_confetti:
        centroids = [c for c in centroids
                     if not c.features.get("likely_confetti", False)]
    if best_label == 'best_in_k':
        # Select the 'best' from among all vascular centroids.
        # Do this by finding the earliest peak, then among centroids
        # peaking there, find the highest. "highest-of-earliest"
        peak_idxs = np.array([c.peak_index for c in centroids])
        if len(peak_idxs) > 0:
            earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
            highest_early_peak_idx = earliest_peak_idxs[np.argmax([
                centroids[i].peak_value for i in earliest_peak_idxs
            ])]
        else:
            return None

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
        # Do an additional filter on 'likely_vascular' criteria
        # to exclude peaks later than 10 minutes, a very conservative threshold
        early_peaking_centroids = list()
        for c in centroids:
            if c.time_units.lower().startswith("s"):
                time_threshold = 600
            else:
                time_threshold = 10
            if c.timepoints[c.peak_index] < time_threshold:
                early_peaking_centroids.append(c)
            else:
                logger.info(f"Centroid {c.name} excluded for peaking too late "
                            f"({c.activity[c.peak_index]:0.2f} @ "
                            f"t={c.timepoints[c.peak_index]:0.1f} c.time_units)")
        top_indices, top_frequencies = np.unique(
            [c.peak_index for c in early_peaking_centroids],
            return_counts=True
        )
        logger.debug("Selecting best centroid of all best-in-k centroids:")
        for ti, tf in zip(top_indices, top_frequencies):
            logger.debug(f"  {tf} clusters peaked at index {ti}")

        # This is the most likely time point to have the best vascular peak,
        # but it is only about 90% accurate in our tests. So we'll also consider
        # the next time point, but only if it has both a higher peak than our
        # current best centroid and a more spatially concise clustering.
        if len(top_frequencies) == 0:
            # We were passed an empty list for centroids
            # Maybe later, we could allow overriding clusters to loosen thresholds.
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
        mask_img,
        ks,
        step,
        mid_times=None,
        num_cpus=1,
        keep_confetti=False,
        verbose=0,
        logger=None,
):
    """ From PET data, find a vascular cluster.

    Loop over all values for k in ks, looking for k-means clusters that
    exhibit vascular-like properties. Return the best possible
    cluster.

    :param ndarray pet_4d_img: Nifti with 3D array of timeseries, 4D overall
    :param ndarray mask_img: Nifti with 3D array of zero/non-zero values
    :param iterable ks: Iterable of integers, each used as a k in k-means
    :param int step: Which step of k-means, used for labeling centroids.
    :param iterable mid_times: will be stored alongside activity in TACs
    :param int num_cpus: how many processes to use on finding centroids
    :param bool keep_confetti: Whether to keep the confetti-like clusters
    :param int verbose: Set non-zero to increase logging, higher is more
    :param logging.logger logger: output comments here if available

    :returns tuple: The best TAC, and all the TACs
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # Run k-means, and label centroids with features.
    all_centroids, k_means_fits = find_centroids(
        pet_4d_img,
        mask_img,
        ks,
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
                # if (keep_confetti or
                #         centroid.features.get('confetti_score', 0.0) <= 0.0
                # ):
                vascular_centroids.append(centroid)
                # else:
                #     other_centroids.append(centroid)
            else:
                other_centroids.append(centroid)

        for i, vc in enumerate(vascular_centroids):
            logger.debug(f"  {vc.label}/{vc.k}: {vc.peak_value:0.3f} at {vc.peak_index} "
                         f"(confetti score: {vc.features.get('confetti_score', 0.0):0.3f})")

        # Label the top candidate for a vascular cluster from this k value.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous
        if len(vascular_centroids) > 0:
            top_c = label_best_centroid(vascular_centroids, 'best_in_k', keep_confetti)
            if top_c:
                logger.debug(
                    "  Early centroid [{}/{}] has peak of {:0.3f} at t {}".format(
                        top_c.label, top_c.k, top_c.peak_value, top_c.peak_index,
                    )
                )
            else:
                logger.debug("  Vascular centroids were all confetti-like.")
        else: logger.debug(f"  No vascular centroids found for k={k}.")
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
    very_top_c = label_best_centroid(best_in_k_centroids, 'best_overall')
    if very_top_c:
        logger.debug(
            f"  From {len(best_in_k_centroids)} best-in-k options, "
            f"selected centroid [{very_top_c.label}/{very_top_c.k}] has peak of "
            f"{very_top_c.peak_value:0.3f} at t {very_top_c.peak_index}"
        )
    else:
        logger.debug("  Improbably, all best-in-k centroids were confetti-like.")

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits


def build_similarity(centroids, order_by=None, logger=None):
    """ Calculate Dice similarity between centroids
    """

    num_dices_calculated = 0
    num_dices_duped = 0
    num_diagonals = 0
    mat_len = len(centroids)
    dice_matrix = np.ones((mat_len, mat_len)) * 100.0
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
            elif dice_matrix[i, j] < 99.0:
                dice_matrix[j, i] = dice_matrix[i, j]
                num_dices_duped += 1
            elif dice_matrix[j, i] < 99.0:
                dice_matrix[i, j] = dice_matrix[j, i]
                num_dices_duped += 1
            else:
                dice_matrix[i, j] = dice_coef(
                    np.ravel(c1.labels == c1.label),
                    np.ravel(c2.labels == c2.label),
                )
                dice_matrix[j, i] = dice_matrix[i, j]
                num_dices_calculated += 1
    _msg = (f"Cluster similarity: calculated {num_dices_calculated:,} / "
            f"{len(centroids) ** 2:,}, duped {num_dices_duped:,}, "
            f"{num_diagonals:,} diagonals.")
    if logger is None:
        print(_msg)
    else:
        logger.info(_msg)

    # No 100's should be left, but clear them out just in case.
    dice_matrix[dice_matrix > 1.0] = 0.0
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
            inf_weighted_score_str = "n/a"
            if 'inf_weighted_score' in c.features:
                inf_weighted_score_str = f"{c.features['inf_weighted_score']:0.2f}"
            reduced_str = "Δ" if c.source == "sparsity reduction" else ""
            html_table.append(
                "<tr>"
                f"<td>{c.label:02d}/{c.k:02d}</td>"
                f"<td>{c.peak_value:0.4f}</td>"
                f"<td>{stability_str}</td>"
                f"<td>{overlap_str}</td>"
                f"<td>{match_best_str}</td>"
                f"<td>{inf_weighted_score_str}</td>"
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
                    "NOT REALLY IN THIS VERSION: "
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

    all_centroids, k_means_fits = find_centroids(
        pet_4d_img,
        None,
        ks,
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
