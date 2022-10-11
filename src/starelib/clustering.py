from pathlib import Path
import numpy as np
import logging
import nibabel as nib
from sklearn.cluster import KMeans
from datetime import datetime
import pickle

from .util import flatten_4d_to_2d, reshape_labels_to_3d
from .plotting import tacs_to_plottable_dataframe, plot_simple_tacs
from .centroid import Centroid


def make_atlas_and_mask(centroid, template_img, pad_inferior=0, out_path=None):
    """ Save a centroid's cluster as a mask.

        This function intentionally saves the mask as-is, meaning if
        axial slices were cropped, they are still cropped here. In
        this version of stare_pet, each mask may or may not overlay
        the original PET image correctly.

    :param Centroid centroid: A dict containing centroid data and metadata
    :param Nifti1Image template_img: Image to use as a template for mask data
    :param int pad_inferior: Add this number of axial slices to the inferior
                             edge of the volume, for reversing the crop
    :param Path out_path: If provided, directory for writing out atlas and mask
                          By default, these are not written to disk
    :return: paths to atlas image and mask image
    """

    logger = logging.getLogger("STARE")

    # Shape the voxel labels into a 3d matrix to match the template image.
    cluster_atlas_data = reshape_labels_to_3d(
        centroid.labels,
        (template_img.shape[0], template_img.shape[1],
         template_img.shape[2] - pad_inferior)
    )

    # If requested, add back the cropped axial slices
    if pad_inferior > 0:
        replacement_slices = np.zeros(
            (template_img.shape[0], template_img.shape[1], pad_inferior)
        )
        # It is important to add replacement slices first, followed by atlas
        # I-S coordinates in this array are from inferior 0 to + superior
        cluster_atlas_data = np.concatenate(
            (replacement_slices, cluster_atlas_data), axis=2
        )

    # Build an atlas and a mask, based on these labels.
    cluster_atlas_img = nib.Nifti1Image(
        cluster_atlas_data.astype(int),
        affine=template_img.affine, header=template_img.header
    )
    cluster_atlas_img.update_header()
    cluster_mask_img = nib.Nifti1Image(
        (cluster_atlas_data == centroid.label).astype(int),
        affine=template_img.affine, header=template_img.header
    )
    cluster_mask_img.update_header()

    # Write out images if they don't already exist.
    k = centroid.k
    label = centroid.label
    space = "_orig" if pad_inferior > 0 else ""
    atlas_filename = f"cluster_k-{k:02d}_atlas{space}.nii.gz"
    mask_filename = f"cluster_k-{k:02d}_label-{label:02d}_mask{space}.nii.gz"
    if out_path is None:
        logger.debug(f"Made atlas & mask for k={k:02d}, label={label:02d}")
    else:
        if not (out_path / atlas_filename).exists():
            nib.save(cluster_atlas_img, out_path / atlas_filename)
        if not (out_path / mask_filename).exists():
            nib.save(cluster_mask_img, out_path / mask_filename)
        logger.debug(f"Made and saved atlas & mask for k={k:02d}, "
                     f"label={label:02d}")

    return out_path / mask_filename


def best_of(centroids):
    """ Return the centroid labeled best_overall from centroids.

        :param list centroids: A list of Centroid objects

        :return: Centroid object labeled "best_overall"
    """

    for centroid in centroids:
        if centroid.best_overall:
            return centroid


def save_centroid_masks(centroids, mask_output_path,
                        current_template, original_template,
                        axial_slices_to_clip=0, verbose=0):
    """ Save centroid masks to disk, return the best one

        :param list centroids: list of Centroid objects to write to disk
        :param Path mask_output_path: The path for writing out masks
        :param Nifti1Image current_template: An image in cropped clustering space
        :param Nifti1Image original_template: An image in original space
        :param int axial_slices_to_clip: how many axial slices to remove
        :param int verbose: higher numbers indicate more verbosity

        :return: Best atlas image, in original uncropped space
    """

    best_atlas = None
    for centroid in centroids:
        if centroid.best_overall:
            # Write the best atlas and mask to disk regardless of verbosity.
            # Specifying out_path causes masks to be written to disk.
            make_atlas_and_mask(
                centroid, current_template, out_path=mask_output_path
            )
            # Add back the cropped axial slices and save image in original space
            if axial_slices_to_clip > 0:
                best_atlas = make_atlas_and_mask(
                    centroid, original_template,
                    pad_inferior=axial_slices_to_clip, out_path=mask_output_path
                )
        if verbose > 1:  # for all centroids, not just the best one
            # Specifying out_path causes masks to be written to disk.
            # Write EVERY cluster to disk for future debugging
            if (mask_output_path.parent / "debug").exists():
                make_atlas_and_mask(
                    centroid, current_template,
                    out_path=mask_output_path.parent / "debug"
                )

    return best_atlas


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
        # vascular_centroids = []
        # other_centroids = []
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
            for feature_label, fxn in features.items():
                this_centroid.features[feature_label] = fxn(this_centroid)
                if this_centroid.features[feature_label]:
                    feature_counts[feature_label] += 1
            feature_counts["total"] += 1

            all_centroids.append(this_centroid)

        for label, count in feature_counts.items():
            if label != "total":
                logger.debug(f"  {count:03d} /{feature_counts['total']:03d} are {label}")

    post_k_timestamp = datetime.now()
    logger.info(f"All {len(ks)} k-means finished in "
                f"{post_k_timestamp - pre_k_timestamp}")

    return all_centroids, k_means_fits


def two_step_clustering(
        image, step_one_ks, step_two_ks, cluster_function, output_path,
        mid_times, force=False, verbose=0
):
    """ Perform two step clustering of PET data

        This step is the first of six in the STARE process.
        It flattens the four dimensions of a series of volumes
        into a 2D matrix with a timeseries vector for each voxel represented
        in each 3D volume. Those vectors are clustered in two steps to find
        likely the "best" regions, as decided by cluster_function.
        Centroids from each cluster are then plotted
        before and after "best" centroids are recognized.

        :param Nifti1Image image: A 4d image, containing a sequence of volumes
        :param list step_one_ks: A list of ints to serve as cluster quantities
        :param list step_two_ks: A list of ints to serve as cluster quantities
        :param function cluster_function: A centroid selection function
        :param Path output_path: the main output path for one subject
        :param list mid_times: A list of images
        :param int force: If true, run everything regardless of cache
        :param int verbose: Set to non-zero to trigger logging, higher is more

        :return: None
    """

    logger = logging.getLogger("STARE")

    pre_kmc_timestamp = datetime.now()
    logger.info(f"Started two-level k-means clustering at {pre_kmc_timestamp}")

    debug_path = output_path / "debug"
    cache_path = output_path / "cache"
    fig_path = output_path / "figures"

    # -------------------------------------------------------------------------
    # Step 0. Collect the individual 3D volumes provided,
    #         and combine them into a 4D image.
    # -------------------------------------------------------------------------

    # We need a timeseries vector at each voxel.
    to_cluster = flatten_4d_to_2d(image.get_fdata(), zxy=True)

    # -------------------------------------------------------------------------
    # Step 1. Find the best candidate for a vascular cluster of voxels.
    #         The first step tries 10 values of k between 6 and 40,
    #         and selects the best cluster
    # -------------------------------------------------------------------------

    # If prior models were saved to disk, load them rather than running.
    cache_file_1 = cache_path / "step_1_centroids_and_fits.pkl"
    if cache_file_1.exists() and not force:
        logger.info("  loading cached step 1 k-means to save time")
        centroids_step_1, model_fits = pickle.load(cache_file_1.open("rb"))
    else:
        centroids_step_1, model_fits = cluster_function(
            to_cluster, step_one_ks, mid_times=mid_times, verbose=verbose
        )
        cache_file_1.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((centroids_step_1, model_fits), cache_file_1.open("wb"))
        logger.debug(f"WROTE {cache_file_1.name} (pickled all_centroids "
                     f"and model_fits tuple) to {str(cache_path)}")

    # Label the best centroid for proper figure legend
    for c in centroids_step_1:
        if c.best_in_k:
            c.name = f"Best step 1. {c.name}"

    if verbose > 1:
        # These data can be used to build custom plots or otherwise explore.
        debug_path.mkdir(parents=True, exist_ok=True)
        tacs_to_plottable_dataframe(centroids_step_1).to_csv(
            debug_path / "step_1_centroids.csv"
        )
        logger.debug(f"WROTE step_1_centroids.csv to {str(debug_path)}")

    # Plot the TACs from the first k-means step
    fig = plot_simple_tacs(centroids_step_1)
    fig.savefig(fig_path / "step_1_vascular_tacs.png")

    # -------------------------------------------------------------------------
    # Step 2. Find the best candidate from step 1 for a vascular cluster of
    #         voxels. The second step finds the best of k=4 clusters from
    #         within only the voxels discovered in step 1.
    # -------------------------------------------------------------------------

    # All of the following atlases and masks from make_atlas_and_mask are Nifti1
    # images, which are translated from Analyze images in fsleyes or freeview.
    # The affine matrices are identical, so it will take some digging to find
    # the source of the problem or to determine if it is actually a problem.
    # Overlaying HarvardOxford atlas regions in fsleyes lays them atop the
    # NEW Nifti1 space, not the original SpmAnalyze space.

    # Mask out only the voxels belonging to the best cluster from step 1.
    top_centroid_step_1 = best_of(centroids_step_1)

    # Generate the masked data, with only the best centroid's data
    top_centroid_masked_data = np.zeros(to_cluster.shape, )
    top_cluster_mask = top_centroid_step_1.labels == top_centroid_step_1.label
    top_centroid_masked_data[top_cluster_mask] = to_cluster[top_cluster_mask, :]

    # Run the second k-means, but only on the best cluster from the first.
    cache_file_2 = cache_path / "step_2_centroids_and_fits.pkl"
    if cache_file_2.exists() and not force:
        logger.info("  loading cached step 2 k-means to save time")
        centroids_step_2, second_model_fits = pickle.load(cache_file_2.open("rb"))
    else:
        centroids_step_2, second_model_fits = cluster_function(
            top_centroid_masked_data, step_two_ks, mid_times=mid_times, verbose=verbose
        )
        cache_file_2.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((centroids_step_2, second_model_fits), cache_file_2.open("wb"))
        logger.debug(f"WROTE step_2_centroids_and_fits.pkl to {str(cache_path)}")

    # Label the best centroid for proper figure legend
    for c in centroids_step_2:
        if c.best_in_k:
            c.name = f"Best step 2. {c.name}"

    fig = plot_simple_tacs(centroids_step_2)
    fig.savefig(fig_path / "step_2_vascular_tacs.png")
    if verbose > 1:
        # These data can be used to build custom plots or otherwise explore.
        debug_path.mkdir(parents=True, exist_ok=True)
        tacs_to_plottable_dataframe(centroids_step_2).to_csv(
            debug_path / "step_2_centroids.csv"
        )
        logger.debug(f"WROTE step_2_centroids.csv to {str(debug_path)}")

    return centroids_step_1, centroids_step_2
