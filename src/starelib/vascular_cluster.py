from pathlib import Path
import numpy as np
import logging
from sklearn.cluster import KMeans
import nibabel as nib
from datetime import datetime
import pickle

from .util import flatten_4d_to_2d, reshape_labels_to_3d, \
                  combine_volumes_into_4d
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

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    k_means_fits = {}
    all_centroids = []
    for k in ks:
        vascular_centroids = []
        other_centroids = []
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

        # Find reasonable timeseries in the cluster means.
        count_irreversible, count_noise = 0, 0
        for i in range(k_means.n_clusters):
            cc = k_means.cluster_centers_[i]
            this_centroid = Centroid(
                activity=cc,
                timepoints=mid_times,
                label=i + 1,  # should be non-zero as zero indicates background
                k=k,
                labels=k_means.labels_ + 1,
            )
            # Rule out timeseries that climb through the end (matlab line 127)
            probably_irreversible = cc[-1] == max(cc)
            # Rule out timeseries with negative values beyond time 0 (matlab line 128)
            probably_noise = np.any(cc[1:] < 0)
            if probably_irreversible:
                count_irreversible += 1
            if probably_noise:
                count_noise += 1
            if not probably_irreversible and not probably_noise:
                # Store the data (cc) and its metadata along with it.
                # The best* fields are unknowable now, will be updated later.
                this_centroid.vascular = True
                vascular_centroids.append(this_centroid)
            else:
                other_centroids.append(this_centroid)

        logger.debug(f"  Eliminate {count_irreversible} of {k}"
                     " cluster centers as irreversible.")
        logger.debug(f"  Eliminate {count_noise} of {k}"
                     " cluster centers as noise.")
        logger.debug(f"When k == {k},"
                     f"we find {len(vascular_centroids)} vascular centroids.")
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

        # Add centroids from this k to the larger collection.
        # We originally only returned vascular centroids, but now include
        # all centroids, with vascular ones labeled as such within-dict.
        all_centroids = all_centroids + vascular_centroids + other_centroids

    post_k_timestamp = datetime.now()
    logger.info(f"All {len(ks)} k-means finished in "
                f"{post_k_timestamp - pre_k_timestamp}")

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


def vascular_clustering(
        output_path, images, mid_times,
        pet_units='kBq', axial_slices_to_clip=0, force=False, verbose=0
):
    """ Perform vascular clustering, step 1 of 6 in the STARE process

        This step is the first of six in the STARE process.
        It concatenates and clips all pre-motion-corrected 3D volumes
        into a single 4D Nifti file. It then flattens those four dimensions
        into a 2D matrix with a timeseries vector for each voxel represented
        in each 3D volume. Those vectors are clustered in two steps to find
        likely vascular regions. Centroids from each cluster are then plotted
        before and after centroids most likely to be vascular are
        recognized.

        :param Path output_path: the main output path for one subject
        :param list images: A list of images
        :param list mid_times: A list of images
        :param str pet_units: 'kBq' or 'Bq', anything else treated as 'mCi'
        :param int axial_slices_to_clip: how many axial slices to remove
        :param int force: If true, run everything regardless of cache
        :param int verbose: Set to non-zero to trigger logging, higher is more

        :return: None
    """

    logger = logging.getLogger("STARE")

    pre_kmc_timestamp = datetime.now()
    logger.info(f"Started two-level k-means clustering at {pre_kmc_timestamp}")

    output_path = Path(output_path)  # just to make sure
    debug_path = output_path / "debug"
    cache_path = output_path / "cache"

    # -------------------------------------------------------------------------
    # Step 0. Collect the individual 3D volumes provided,
    #         and combine them into a 4D image.
    # -------------------------------------------------------------------------

    # Collect all the 3d image data into a single 4d structure.
    combined_image = combine_volumes_into_4d(
        images, output_path / "orig.nii.gz", logger=logger
    )
    combined_template = combined_image.slicer[:, :, :, 0]

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = combined_image.slicer[:, :, axial_slices_to_clip:, :]
    nib.save(cropped_image, output_path / "orig_cropped.nii.gz")
    logger.debug(f"WROTE orig_cropped.nii.gz ({cropped_image.shape}) "
                 f"to {str(output_path)}")
    cropped_template = cropped_image.slicer[:, :, :, 0]

    # PET data should be in units of 'mCi'
    # If they already are, good, but other units get converted here.
    pet_4d_data = cropped_image.get_fdata()
    if pet_units.lower() == "kbq":
        pet_4d_data = pet_4d_data / 37000
    elif pet_units.lower() == "bq":
        pet_4d_data = pet_4d_data / 37000000

    # Create a 2D array of voxel-wise 4D imaging matrix for entry into kmeans.
    # We need a timeseries vector at each voxel.
    to_cluster = flatten_4d_to_2d(pet_4d_data, zxy=True)

    fig_path = output_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)
    mask_path = output_path / "masks"
    mask_path.mkdir(parents=True, exist_ok=True)

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
        centroids_step_1, model_fits = find_vascular_centroids(
            to_cluster, range(6, 40, 4), mid_times=mid_times, verbose=verbose
        )
        cache_file_1.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((centroids_step_1, model_fits), cache_file_1.open("wb"))
        logger.debug(f"WROTE {cache_file_1.name} (pickled all_centroids "
                     f"and model_fits tuple) to {str(cache_path)}")

    # Lengthen data from wide to long for plotting, and plot TACs
    centroid_data = tacs_to_plottable_dataframe(centroids_step_1)

    if verbose > 1:
        # These data can be used to build custom plots or otherwise explore.
        debug_path.mkdir(parents=True, exist_ok=True)
        centroid_data.to_csv(debug_path / "step_1_centroids.csv")
        logger.debug(f"WROTE step_1_centroids.csv to {str(debug_path)}")

    # Plot the TACs from the first k-means step
    fig = plot_simple_tacs(centroid_data)
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
    top_centroid_step_1, top_centroid_step_2 = None, None
    for centroid in centroids_step_1:
        if centroid.best_overall:
            top_centroid_step_1 = centroid
            # Write the best atlas and mask to disk regardless of verbosity.
            # Specifying out_path causes masks to be written to disk.
            make_atlas_and_mask(
                centroid, cropped_template, out_path=mask_path
            )
            # Add back the cropped axial slices and save image in original space
            if axial_slices_to_clip > 0:
                make_atlas_and_mask(
                    centroid, combined_template,
                    pad_inferior=axial_slices_to_clip, out_path=mask_path
                )
        if verbose > 1:
            # Specifying out_path causes masks to be written to disk.
            # Write EVERY cluster to disk for future debugging
            debug_path.mkdir(parents=True, exist_ok=True)
            make_atlas_and_mask(
                centroid, cropped_template, out_path=debug_path
            )

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
        centroids_step_2, second_model_fits = find_vascular_centroids(
            top_centroid_masked_data, [4, ], mid_times=mid_times, verbose=verbose
        )
        cache_file_2.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((centroids_step_2, second_model_fits), cache_file_2.open("wb"))
        logger.debug(f"WROTE step_2_centroids_and_fits.pkl to {str(cache_path)}")

    # Lengthen data from wide to long for plotting, and plot TACs
    centroid_data_2 = tacs_to_plottable_dataframe(centroids_step_2)

    fig = plot_simple_tacs(centroid_data_2)
    fig.savefig(fig_path / "step_2_vascular_tacs.png")
    if verbose > 1:
        # These data can be used to build custom plots or otherwise explore.
        debug_path.mkdir(parents=True, exist_ok=True)
        centroid_data_2.to_csv(debug_path / "step_2_centroids.csv")
        logger.debug(f"WROTE step_2_centroids.csv to {str(debug_path)}")

    best_mask_orig = None
    for centroid in centroids_step_2:
        if centroid.best_overall:
            top_centroid_step_2 = centroid
            make_atlas_and_mask(
                centroid, cropped_template, out_path=mask_path
            )
            # Add back the cropped axial slices and save image in original space
            if axial_slices_to_clip > 0:
                best_mask_orig = make_atlas_and_mask(
                    centroid, combined_template,
                    pad_inferior=axial_slices_to_clip, out_path=mask_path
                )
        if verbose > 1:
            debug_path.mkdir(parents=True, exist_ok=True)
            make_atlas_and_mask(
                centroid, cropped_template, out_path=debug_path
            )

    return best_mask_orig, top_centroid_step_1, top_centroid_step_2
