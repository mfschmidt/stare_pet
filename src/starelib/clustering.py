from pathlib import Path
import numpy as np
import logging
import nibabel as nib
from nilearn import image
from datetime import datetime
import pickle

from .util import flatten_4d_to_2d, reshape_labels_to_3d
from .util import from_cache, to_cache
from .centroid import Centroid
from .plotting import tacs_to_plottable_dataframe, plot_vascular_tacs
from .plotting import plot_top_centroids_atlas
from .centroid_heuristics import find_vascular_centroids


def make_atlas_and_mask(
        centroid, template_img,
        pad_inferior=0, out_path=None, file_desc=None, logger=None
):
    """ Save a centroid's cluster as a mask.

        This function intentionally saves the mask as-is, meaning if
        axial slices were cropped, they are still cropped here. In
        this version of stare_pet, each mask may or may not overlay
        the original PET image correctly.

    :param Centroid centroid: A Centroid object with label data
    :param Nifti1Image template_img: Image to use as a template for mask data
    :param int pad_inferior: Add this number of axial slices to the inferior
                             edge of the volume, for reversing the crop
    :param Path out_path: If provided, directory for writing out atlas and mask
                          By default, these are not written to disk
    :param str file_desc: If provided, filename is overridden
    :param logging.logger logger: If provided, write output to this logger
    :return: paths to atlas image and mask image
    """

    logger = logging.getLogger("STARE") if logger is None else logger

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
        np.array(cluster_atlas_data == centroid.label).astype(int),
        affine=template_img.affine, header=template_img.header
    )
    cluster_mask_img.update_header()

    # Write out images if they don't already exist.
    k = centroid.k
    label = centroid.label
    space = "_orig" if pad_inferior > 0 else ""
    atlas_filename = f"cluster_k-{k:02d}_atlas{space}.nii.gz"
    mask_filename = f"cluster_k-{k:02d}_label-{label:02d}_mask{space}.nii.gz"
    if file_desc is not None:
        # Override the default file names
        atlas_filename = f"cluster_{file_desc}_atlas{space}.nii.gz"
        mask_filename = f"cluster_{file_desc}_mask{space}.nii.gz"
    if out_path is None:
        logger.debug(f"Made atlas & mask for k={k:02d}, label={label:02d}")
        return cluster_atlas_img, cluster_mask_img
    else:
        if not (out_path / atlas_filename).exists():
            nib.save(cluster_atlas_img, out_path / atlas_filename)
        if not (out_path / mask_filename).exists():
            nib.save(cluster_mask_img, out_path / mask_filename)
        logger.debug(f"Made and saved atlas & mask for k={k:02d}, "
                     f"label={label:02d}")
        return out_path / atlas_filename, out_path / mask_filename


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
                        step=0, axial_slices_to_clip=0, verbose=0,
                        logger=None):
    """ Save centroid masks to disk, return the best one

        :param list centroids: list of Centroid objects to write to disk
        :param Path mask_output_path: The path for writing out masks
        :param Nifti1Image current_template: An image in cropped cluster space
        :param Nifti1Image original_template: An image in original space
        :param int step: which step generated this mask
        :param int axial_slices_to_clip: how many axial slices to remove
        :param int verbose: higher numbers indicate more verbosity
        :param logging.logger logger: write output to logger if available

        :return nibabel.Nifti1Image: the best atlas image, in original space
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    best_mask_path = None
    for centroid in centroids:
        if centroid.best_overall:
            # Write the best atlas and mask to disk regardless of verbosity.
            # Specifying out_path causes masks to be written to disk.
            best_atlas_path, best_mask_path = make_atlas_and_mask(
                centroid, current_template,
                out_path=mask_output_path,
                file_desc=f"step-{step}_best",
                logger=logger,
            )
            # Add back the cropped axial slices and save image in original space
            if axial_slices_to_clip > 0:
                best_atlas_path, best_mask_path = make_atlas_and_mask(
                    centroid, original_template,
                    pad_inferior=axial_slices_to_clip,
                    out_path=mask_output_path, file_desc=f"step-{step}_best",
                    logger=logger,
                )
        if verbose > 2:  # for all centroids, not just the best one
            # Specifying out_path causes masks to be written to disk.
            # Write EVERY cluster to disk for future debugging
            if (mask_output_path.parent / "debug").exists():
                make_atlas_and_mask(
                    centroid, current_template,
                    out_path=mask_output_path.parent / "debug",
                    logger=logger,
                )

    # This should be in original space, cropped sliced padded back
    return best_mask_path


def cluster(results, cluster_function, data, ks, step):
    # If prior models were saved to disk, load them rather than running.
    cache_file = "sub-{}_step-1-{}_centroids_and_fits.pkl".format(
        results.args.subject, step
    )
    cached_data = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    vol_shape = results.cropped_4D.shape[:3]
    if cached_data is None:
        centroids, model_fits = cluster_function(
            data, vol_shape, ks, mid_times=results.mid_times,
            num_cpus=results.args.num_cpus,
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
    for handler in logger.handlers:
        handler.flush()

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
    rpt_sect.add_line(str(best_of(results.cluster_centroids[1])))

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
    rpt_sect.add_line(str(best_of(results.cluster_centroids[2])))

    # Plot the top centroids over PET data, and add it to the report.
    pet_avg_img = image.mean_img(results.cropped_4D)
    best_atlases, best_masks, best_cs = {}, {}, {}
    for step in [1, 2]:
        best_cs[step] = best_of(results.cluster_centroids[step])
        best_atlases[step], best_masks[step] = make_atlas_and_mask(
            best_cs[step], pet_avg_img
        )

    top_centroid_fig = plot_top_centroids_atlas(
        best_masks[1], best_masks[2], pet_avg_img,
        title="\n".join([
            f"{results.args.subject}:",
            f"step 1. orange. {best_cs[1].label} of {best_cs[1].k}",
            f"step 2. red. {best_cs[2].label} of {best_cs[2].k}",
        ]),
    )
    filename = f"sub-{results.args.subject}_step-1-vascular_cluster_masks.png"
    top_centroid_fig.savefig(results.args.fig_path / filename)
    caption = "Step one (orange) and step two (red) vascular clusters"
    rpt_sect.add_figure(results.args.fig_path / filename, caption)

    # Plot the TACs from the first k-means step
    for step in [1, 2, ]:
        # Plot the TACs from vascular cluster centroids.
        fig = plot_vascular_tacs(results.cluster_centroids[step], tall=True)
        filename = f"sub-{results.args.subject}_step-1-{step}_vascular_tacs.png"
        fig.savefig(results.args.fig_path / filename)
        logger.info(f"WROTE {filename} to {str(results.args.fig_path)}")

        caption = f"See figure: K-Means Vascular Clustering, Step {step}"
        # rpt_sect.add_figure(
        #     results.args.fig_path / filename,
        #     caption,
        #     css_class={1: 'left_fig', 2: 'right_fig'}[step]
        # )
        rpt_sect.add_link(results.args.fig_path / filename, text=caption)

        # For debugging, plot all clusters for each k, to see best vs rest
        if results.save_all_cluster_masks or (results.args.verbose > 1):
            unique_ks = sorted(np.unique([
                c.k for c in results.cluster_centroids[step]
            ]))
            for k in unique_ks:
                cs_in_k = [c for c in results.cluster_centroids[step]
                           if c.k == k]
                fig_s_k = plot_vascular_tacs(
                    cs_in_k, draw_non_vascular=True, tall=True
                )
                filename_s_k = filename.replace("_vas", f"_k-{k}_vas")
                fig_s_k.savefig(results.args.debug_path / filename_s_k)

            # For debugging, draw all masks on the average PET
            for c in results.cluster_centroids[step]:
                c_fig = plot_top_centroids_atlas(
                    make_atlas_and_mask(c, pet_avg_img)[1],
                    None,
                    pet_avg_img,
                    title="\n".join([
                        f"{results.args.subject}:",
                        f"step 1. orange. {c.label} of {c.k}, "
                        f"peak {c.peak_value:0.2f} @ t # {c.peak_index}",
                    ]),
                )
                filename = (f"sub-{results.args.subject}_step-1-k-{c.k}_"
                            f"label-{c.label}_vascular_cluster_mask.png")
                c_fig.savefig(results.args.debug_path / filename)

        # These data can be used to build custom plots or otherwise explore.
        filename = f"sub-{results.args.subject}_step-1-{step}_kmeans_tac.csv"
        tacs_to_plottable_dataframe(results.cluster_centroids[step]).to_csv(
            results.args.output_path / filename
        )
        logger.info(f"WROTE {filename} to {str(results.args.output_path)}")

        filename = "sub-{}_step-1-{}_kmeans_centroid.pkl".format(
            results.args.subject, step
        )
        with open(results.args.output_path / "debug" / filename, "wb") as f:
            pickle.dump(
                best_of(results.cluster_centroids[step]), f
            )

        # Save out nifti masks (which ones conditional on verbosity)
        best_vascular_mask_path = save_centroid_masks(
            results.cluster_centroids[step],
            results.args.output_path / "masks",
            results.cropped_4D.slicer[:, :, :, 0],
            results.input_4D.slicer[:, :, :, 0],
            step=step,
            axial_slices_to_clip=results.args.axial_slices_to_clip,
            verbose=results.args.verbose,
            logger=logger,
        )
        results.best_vascular_mask_path[step] = best_vascular_mask_path

    rpt_sect.end()
    return results
