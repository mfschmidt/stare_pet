from pathlib import Path
import numpy as np
import logging
import nibabel as nib
from datetime import datetime
import pickle

from .util import flatten_4d_to_2d, reshape_labels_to_3d
from .util import from_cache, to_cache
from .centroid import Centroid
from .plotting import tacs_to_plottable_dataframe, plot_vascular_tacs
from .centroid_heuristics import find_vascular_centroids


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
        :param Nifti1Image current_template: An image in cropped cluster space
        :param Nifti1Image original_template: An image in original space
        :param int axial_slices_to_clip: how many axial slices to remove
        :param int verbose: higher numbers indicate more verbosity

        :return nibabel.Nifti1Image: the best atlas image, in original space
    """

    best_mask_path = None
    for centroid in centroids:
        if centroid.best_overall:
            # Write the best atlas and mask to disk regardless of verbosity.
            # Specifying out_path causes masks to be written to disk.
            best_mask_path = make_atlas_and_mask(
                centroid, current_template, out_path=mask_output_path
            )
            # Add back the cropped axial slices and save image in original space
            if axial_slices_to_clip > 0:
                best_mask_path = make_atlas_and_mask(
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

    # This should be in original space, cropped sliced padded back
    return best_mask_path


def cluster(results, cluster_function, data, ks, step):
    # If prior models were saved to disk, load them rather than running.
    cache_file = f"step_{step}_centroids_and_fits.pkl"
    cached_data = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if cached_data is None:
        centroids, model_fits = cluster_function(
            data, ks, mid_times=results.mid_times,
            verbose=results.args.verbose
        )
        to_cache(
            (centroids, model_fits), results.args.cache_path, cache_file
        )
    else:
        (centroids, model_fits) = cached_data
        results.logger.info(f"  loaded cached step {step} k-means to save time")

    # Label the best centroid for proper figure legend
    for c in centroids:
        if c.best_in_k:
            c.name = f"Best step {step}. {c.name}"

    results.cluster_centroids[step] = centroids
    results.cluster_model_fits[step] = model_fits

    return centroids, model_fits


def two_step_cluster(results):
    """ Perform two-step clustering of PET data

        This step is the first of six in the STARE process.
        It flattens the four dimensions of a series of volumes
        into a 2D matrix with a timeseries vector for each voxel represented
        in each 3D volume. Those vectors are clustered in two steps to find
        likely the "best" regions, as decided by cluster_function.
        Centroids from each cluster are then plotted
        before and after "best" centroids are recognized.

        :param Results results: An object containing pipeline data
        :return: None
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Two-level k-means clustering")

    # Predetermined configuration, hardcoded here
    step_one_ks = list(range(6, 40, 4))
    step_two_ks = [4, ]
    cluster_function = find_vascular_centroids

    pre_kmc_timestamp = datetime.now()
    logger.info(f"Started two-level k-means clustering at {pre_kmc_timestamp}")

    # -------------------------------------------------------------------------
    # Step 0. Collect the individual 3D volumes provided,
    #         and combine them into a 4D image.
    # -------------------------------------------------------------------------

    # We need a timeseries vector at each voxel.
    to_cluster = flatten_4d_to_2d(results.cropped_4D.get_fdata(), zxy=True)

    # -------------------------------------------------------------------------
    # Step 1. Find the best candidate for a vascular cluster of voxels.
    #         The first step tries 10 values of k between 6 and 40,
    #         and selects the best cluster
    # -------------------------------------------------------------------------

    cluster(results, cluster_function, to_cluster, step_one_ks, 1)

    # -------------------------------------------------------------------------
    # Step 2. Find the best candidate from step 1 for a vascular cluster of
    #         voxels. The second step finds the best of k=4 clusters from
    #         within only the voxels discovered in step 1.
    # -------------------------------------------------------------------------

    # All the following atlases and masks from make_atlas_and_mask are Nifti1
    # images, which are translated from Analyze images in fsleyes or freeview.
    # The affine matrices are identical, so it will take some digging to find
    # the source of the problem or to determine if it is actually a problem.
    # Overlaying HarvardOxford atlas regions in fsleyes lays them atop the
    # NEW Nifti1 space, not the original SpmAnalyze space.

    # Generate the masked data, with only the best centroid's data from step 1.
    top_centroid_masked_data = np.zeros(to_cluster.shape, )
    best_centroid_step_1 = best_of(results.cluster_centroids[1])
    top_cluster_mask = best_centroid_step_1.labels == best_centroid_step_1.label
    top_centroid_masked_data[top_cluster_mask] = to_cluster[top_cluster_mask, :]

    # Run the second k-means, but only on the best cluster from the first.
    # If prior models were saved to disk, load them rather than running.
    cluster(results, cluster_function, top_centroid_masked_data, step_two_ks, 2)

    # Plot the TACs from the first k-means step
    for step in [1, 2, ]:
        # Plot the TACs from vascular cluster centroids.
        fig = plot_vascular_tacs(results.cluster_centroids[step])
        filename = f"step_1_{step}_vascular_tacs.png"
        fig.savefig(results.args.fig_path / filename)
        logger.info(f"WROTE {filename} to {str(results.args.fig_path)}")

        # These data can be used to build custom plots or otherwise explore.
        filename = f"step_1_{step}_kmeans_tac.csv"
        tacs_to_plottable_dataframe(results.cluster_centroids[step]).to_csv(
            results.args.output_path / filename
        )
        logger.info(f"WROTE {filename} to {str(results.args.output_path)}")

        filename = f"step_1_{step}_kmeans_centroid.pkl"
        pickle.dump(
            best_of(results.cluster_centroids[step]),
            open(results.args.output_path / "debug" / filename, "wb")
        )

        best_vascular_mask_path = save_centroid_masks(
            results.cluster_centroids[step],
            results.args.output_path / "masks",
            results.cropped_4D.slicer[:, :, :, 0],
            results.input_4D.slicer[:, :, :, 0],
            axial_slices_to_clip=results.args.axial_slices_to_clip,
            verbose=results.args.verbose
        )
        results.best_vascular_mask_path = best_vascular_mask_path

    rpt_sect.end()
    return results
