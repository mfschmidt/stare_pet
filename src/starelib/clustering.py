import stat
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
import logging
import copy
import nibabel as nib
from nilearn import image
from datetime import datetime
import pickle
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from importlib.resources import files

from .util import (
    flatten_4d_to_2d, unflatten_2d_to_4d, reshape_labels_to_3d,
    collapse_array_3d,
)
from .util import from_cache, to_cache
from .centroid import Centroid
from .plotting import tacs_to_plottable_dataframe, plot_vascular_tacs
from .plotting import plot_top_centroids_atlas
from .centroid_heuristics import (
    find_vascular_centroids, likely_irreversible_linear,
    consider_alternate_clusters, calculate_spatial_info,
    calculate_k_stability, calculate_axis_weights, build_similarity
)
from .components import decompose_components


def make_atlas_and_mask(
        centroid, template_img,
        labels=None, out_path=None, file_desc=None, logger=None,
        resample_to=None, pad_to=None,
):
    """ Save a centroid's cluster as a mask.

        This function intentionally saves the mask as-is, meaning if
        axial slices were cropped, they are still cropped here. In
        this version of stare_pet, each mask may or may not overlay
        the original PET image correctly.

    :param Centroid centroid: A Centroid object
    :param array labels: Array of labels, some matching the centroid label
    :param Nifti1Image template_img: Image to use as a template for mask data
    :param Path out_path: If provided, directory for writing out atlas and mask
                          Without it, these are not written to disk
    :param str file_desc: If provided, filename is overridden
    :param logging.logger logger: If provided, write info to this logger
    :param resample_to: If provided, resample 3D matrix to this image's space
    :param pad_to: If provided, pad 3D matrix to this image's space
    :return: paths to atlas image and mask image
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    if labels is None:
        labels = centroid.labels

    # Shape the voxel labels into a 3d matrix to match the template image.
    target_img = template_img
    cluster_atlas_data = reshape_labels_to_3d(
        labels, template_img.shape,
    )
    # Handle resampling, if requested
    if resample_to is not None:
        target_img = resample_to
        # First, create an image in native cluster resolution,
        # which is possibly down-sampled from the original
        cluster_atlas_img = nib.Nifti1Image(
            cluster_atlas_data, template_img.affine,
        )
        # Then resample it to the original space,
        # which may still be cropped from the original
        cluster_atlas_data = image.resample_img(
            cluster_atlas_img, target_affine=resample_to.affine,
            interpolation='nearest', target_shape=resample_to.shape,
        ).get_fdata()

    # If requested, add back the cropped axial slices
    if pad_to is not None:
        target_img = pad_to
        pad_inferior = pad_to.shape[2] - cluster_atlas_data.shape[2]
        replacement_slices = np.zeros(
            (cluster_atlas_data.shape[0], cluster_atlas_data.shape[1],
             pad_inferior)
        )
        # It is important to add replacement slices first, followed by atlas
        # I-S coordinates in this array are from inferior 0 to + superior
        cluster_atlas_data = np.concatenate(
            (replacement_slices, cluster_atlas_data), axis=2
        )

    # Build an atlas and a mask, based on these labels.
    # noinspection PyTypeChecker
    cluster_atlas_img = nib.Nifti1Image(
        np.array(cluster_atlas_data, dtype=int),
        affine=target_img.affine, header=target_img.header
    )
    cluster_atlas_img.update_header()
    # noinspection PyTypeChecker
    cluster_mask_img = nib.Nifti1Image(
        np.array(cluster_atlas_data == centroid.label, dtype=int),
        affine=target_img.affine, header=target_img.header
    )
    cluster_mask_img.update_header()

    # Write out images if they don't already exist.
    k = centroid.k
    label = centroid.label
    space = "_orig" if pad_to is not None else ""
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


def save_centroid_masks(centroids, fits, output_path, current_template, step=0,
                        resample_to_template=None, logger=None):
    """ Save centroid masks to disk, return the best one

        :param list centroids: list of Centroid objects to write to disk
        :param dict fits: k-means fit results for extracting labels
        :param Path output_path: The path for writing out masks
        :param Nifti1Image current_template: An image in cropped cluster space
        :param int step: Step 1 or 2 clustering
        :param Nifti1Image resample_to_template: Resample clusters to this space
        :param logging.logger logger: write output to logger if available

    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # Specifying out_path causes masks to be written to disk.
    if output_path.exists():
        mask_path = output_path / "masks"
        mask_path.mkdir(exist_ok=True)
        for centroid in centroids:
            if centroid.features.get("likely_vascular", False):
                this_mask_path = mask_path / "vascular"
            elif centroid.features.get("likely_irreversible", False):
                this_mask_path = mask_path / "irreversible"
            elif centroid.features.get("likely_noise", False):
                this_mask_path = mask_path / "noise"
            else:
                this_mask_path = mask_path / "other"
            this_mask_path.mkdir(exist_ok=True)
            atlas_nifti_file_path, mask_nifti_file_path = make_atlas_and_mask(
                centroid, current_template,
                labels=fits[centroid.k].labels_ + 1,
                out_path=this_mask_path,
                resample_to=resample_to_template,
                logger=logger,
            )
            background_template = current_template
            if resample_to_template is not None:
                background_template = resample_to_template
            filename = (f"sub-{output_path.parent.name}_step-1_"
                        f"k-{centroid.k}_label-{centroid.label}_"
                        f"vascular_cluster_mask.png")
            if step == 1:
                fig = plot_top_centroids_atlas(
                    nib.load(mask_nifti_file_path), None, background_template,
                    color_map=ListedColormap(['orange', 'red', ]),
                    title="\n".join([
                        f"{output_path.parent.name}:",
                        f"step {step}. orange. {centroid.label} of {centroid.k}"
                        f", peak {centroid.peak_value:0.2f} "
                        f"@ t # {centroid.peak_index}"
                        f"({centroid.voxel_count} voxels)",
                    ]),
                )
                fig.savefig(this_mask_path / filename)
                plt.close(fig)
            elif step == 2:
                fig = plot_top_centroids_atlas(
                    None, nib.load(mask_nifti_file_path), background_template,
                    color_map=ListedColormap(['orange', 'red', ]),
                    title="\n".join([
                        f"{output_path.parent.name}:",
                        f"step {step}. red. {centroid.label} of {centroid.k}"
                        f", peak {centroid.peak_value:0.2f} "
                        f"@ t # {centroid.peak_index} "
                        f"({centroid.voxel_count} voxels)",
                    ]),
                )
                fig.savefig(this_mask_path / filename)
                plt.close(fig)


def tabulate_centroids(centroids, added_columns=None):
    """ Make a table of all centroid information as a dataframe.

    :param centroids: an iterable of Centroid objects to describe
    :param added_columns: a dict of name: value pairs to add to the table

    :returns: a dataframe with each centroid described in each row
    """

    centroid_rows = []
    for centroid in centroids:
        centroid_rows.append(centroid.to_dict())
    df = pd.DataFrame(centroid_rows)
    df = df.drop(["timepoints", "activity"], axis=1)
    if added_columns is not None:
        for k, v in added_columns.items():
            df[k] = v
    return df.sort_values(by=['k', 'label', ])


def post_process_clusters(
        centroids,
        k_means_model_fits,
        pet_4d_img,
        results,
        step,
        rpt_sect,
        logger=None
):
    """ After k-means, handle any additional processing before use.
    """

    # If we were asked not to post-process, just pass data right back.
    if results.args.ignore_spatial_info:
        rpt_sect.add_line("Centroids were not post-processed.")
        return centroids  # untouched

    # If we were asked explicitly to use an alternative cluster, do so.
    def _override(k_step, new_mask_path):
        """ Override the k-means auto-selection with a new cluster.

            It's debatable whether this should prevent clustering from running.
            It currently depends on copying things from the calculated centroid
            so it must run after k-means.

            :param k_step: step number to use
            :param new_mask_path: new mask path
            :returns: centroid, data, path to a Nifti image of the mask
        """

        results.best_vascular_mask_path[k_step] = new_mask_path
        rpt_sect.add_line(f"Step {k_step} cluster overridden by external mask"
                          f", '{str(new_mask_path)}'.")
        _manual_centroid, _manual_data = fake_centroid_from_mask(
            new_mask_path, best_of(centroids), pet_4d_img,
        )
        _manual_centroid.name = f"Overridden best step {k_step} centroid"
        # _best_cluster_as_image = nib.nifti1.Nifti1Image(
        #     _manual_data, affine=pet_4d_img.affine,
        # )
        # Delete the cache file for step two. We're overriding it.
        (results.args.cache_path /
         f"sub-{results.args.subject}_k_step-{k_step}_centroids_and_fits.pkl"
         ).unlink(missing_ok=True)

        # Keep the old cluster, but take away its 'best_overall' flag.
        best_of(centroids).best_overall = False
        # Add the new cluster, and flag it as the 'best_overall'.
        _manual_centroid.best_overall = True
        centroids.append(_manual_centroid)

        return _manual_centroid, _manual_data

    # If asked, override the k-means decision. Centroids are modified in place,
    # so the returned objects are convenient, but not necessary.
    if (step == 1) and results.args.override_step_1_cluster:
        _ = _override(
            step, results.args.override_step_1_cluster
        )

    # If we were asked to use an alternative cluster explicitly, do so.
    if (step == 2) and results.args.override_step_2_cluster:
        _ = _override(
            step, results.args.override_step_2_cluster
        )

    # We really don't care about non-vascular centroids,
    # but we track them all so we can put them back together
    # after replacing vascular centroids with new ones including spatial info.
    vascular_centroids = [
        c for c in centroids
        if c.features.get("likely_vascular", False)
    ]
    other_centroids = [c for c in centroids if c not in vascular_centroids]

    # To calculate debug/spatial info, each centroid needs to carry a
    # reference to its 1-based labels. But only the 'best_in_k' centroids
    # were previously paired with their labels. We don't really want to save
    # 200 copies of multi-GB images when we write the results object to disk.
    # But we need to pair labels w/vascular clusters to calculate_spatial_info.
    for c in vascular_centroids:
        a = c.labels if c.labels is not None else np.zeros((0,))
        logger.debug(f"C {c.label}/{c.k} labels {a.shape}")
        c.labels = k_means_model_fits[c.k].labels_ + 1
    # Send centroids off to be run in multiple processes
    # Separate processes cannot share memory, so we do the calculations,
    # then overwrite our vascular centroids list with the new ones.
    # Then we need to weave these new centroids into the original list.
    vascular_centroids = calculate_spatial_info(
        vascular_centroids,
        step, logger, num_cpus=results.args.num_cpus
    )
    logger.info("Calculating centroid similarity and k-stability")
    sim_mat = build_similarity(vascular_centroids)
    calculate_k_stability(vascular_centroids, sim_mat)
    logger.info("Calculating axis weights for detecting neck noise.")
    calculate_axis_weights(vascular_centroids)

    if (
            (step == 1) and
            (results.args.reduce_step_one_sparsity != 0)
    ):
        # Make a copy of the current best centroid,
        # and modify the copy to remove the smallest blobs.
        # Then re-calculate spatial info and continue with it.
        original_best_centroid = best_of(vascular_centroids)
        new_mask = original_best_centroid.mask_in_3d(
            sparsity_threshold=results.args.reduce_step_one_sparsity,
            logger=results.logger,
        )
        new_best_centroid, new_manual_data = fake_centroid_from_mask(
            new_mask, original_best_centroid, pet_4d_img,
        )
        reduced_ratio = new_best_centroid.features.get('reduced_ratio', 0.0)
        new_best_centroid.name = (
            f"{original_best_centroid.name} "
            f"({reduced_ratio:0.0%} reduced)"
        )
        new_best_centroid.source = "sparsity reduction"
        # best_cluster_as_image = nib.nifti1.Nifti1Image(
        #     new_manual_data, affine=pet_4d_img.affine,
        # )
        # The former champion has been displaced. :-(
        original_best_centroid.best_overall = False
        # Provide a key for the plotter to include this TAC, too.
        original_best_centroid.features['former_champion'] = True
        vascular_centroids.append(new_best_centroid)

    # If spatial information convinces us to abandon our original k-means
    # cluster selection, override it with a new one.
    # Pass all centroids so it can report the ratio of vascular/all
    alt_cluster_html = consider_alternate_clusters(
        centroids, k_means_model_fits, pet_4d_img,
        verbose=results.args.verbose, logger=logger
    )
    for line in alt_cluster_html:
        rpt_sect.add_line(line)

    """
    updated_centroids = list()
    for i, old_centroid in enumerate(centroids):
        updated = False
        for new_centroid in vascular_centroids:
            if new_centroid.label == old_centroid.label:
                if new_centroid.k == old_centroid.k:
                    updated_centroids.append(new_centroid)
                    updated = True
        if not updated:
            updated_centroids.append(old_centroid)
    """
    all_centroids = sorted(
        vascular_centroids + other_centroids,
        key=lambda c: (c.k, c.label),
    )
    if results.args.debug:
        sim_mat.to_csv(results.args.debug_path / f"dice_step_{step}.csv")
        # noinspection PyTypeChecker
        pickle.dump(
            all_centroids,
            open(results.args.debug_path / f"step-{step}_centroids.pkl", "wb")
        )

    return all_centroids


def load_or_calculate_clusters(
        results, cluster_function, source_4d_image, ks, step, report_section, logger=None
):
    """ Treat data as either 4D Nifti1Image or 2D array

        :param results: Results object
        :param cluster_function: function that takes k and label as input and
        :param source_4d_image: 4D PET activity image, may differ from i
        :param ks: list of k values for running repeated k-means
        :param step: Which step in two-step k-means, 1 or 2
        :param report_section: Part of the report we can write to
        :param logger: where to stream info/debug/warnings
        :return: dict containing several results
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # If prior models were saved to disk, load them rather than running.
    cache_file = "sub-{}_step-1-{}_centroids_and_fits.pkl".format(
        results.args.subject, step
    )
    cached_data = from_cache(
        results.args.cache_path, cache_file, results.args.force
    )
    if cached_data is None:
        # Calculate the clusters via k-means with multiprocessing
        all_centroids, model_fits = cluster_function(
            source_4d_image, ks, step,
            mid_times=results.mid_times,
            num_cpus=results.args.num_cpus,
            verbose=results.args.verbose,
            logger=logger,
        )

        # Replace all_centroids with a similar list, but tweaked and filled in
        all_centroids = post_process_clusters(
            all_centroids, model_fits, source_4d_image, results, step,
            report_section, logger=logger
        )

        # Save the results, so we can just load them if there's a next time.
        to_cache(
            (all_centroids, model_fits),
            results.args.cache_path, cache_file
        )
    else:
        (all_centroids, model_fits) = cached_data
        results.logger.info(f"  loaded cached step {step} k-means to save time")

    # Generate the masked data, with only the best centroid's data from step 1.
    best_centroid = best_of(all_centroids)
    if best_centroid.labels is None:
        best_centroid.labels = model_fits[best_centroid.k].labels_ + 1
    top_cluster_mask = (best_centroid.labels == best_centroid.label)
    pet_2d_data = flatten_4d_to_2d(source_4d_image.get_fdata(), zxy=True)
    best_centroid_masked_data = np.zeros(pet_2d_data.shape)
    best_centroid_masked_data[top_cluster_mask] = pet_2d_data[top_cluster_mask, :]

    # Return atlas and mask in down-sampled space if we down-sampled
    best_atlas, best_mask = make_atlas_and_mask(
        best_centroid, image.mean_img(source_4d_image, copy_header=True),
        resample_to=None,
    )
    best_cluster_as_image = nib.nifti1.Nifti1Image(
        unflatten_2d_to_4d(
            best_centroid_masked_data,
            source_4d_image.shape
        ),
        affine=source_4d_image.affine,
    )

    results.cluster_centroids[step] = all_centroids
    results.cluster_model_fits[step] = model_fits
    # As far as I can tell, only 'best_centroid' and 'best_cluster_as_image' get used. So we could probably clean this up. Two lines above also cover some of it.
    return {
        'centroids': all_centroids,
        'fits': model_fits,
        'best_centroid': best_centroid,
        'best_data': best_centroid_masked_data,
        'best_atlas': best_atlas,
        'best_cluster_as_image': best_cluster_as_image,
    }


def save_table_of_centroid_stats(results, step):
    """ Tabulate centroid stats and save them to a csv file. """

    # Add on two extra features to each centroid before saving table.
    for c in results.cluster_centroids[step]:
        m1, b1 = likely_irreversible_linear(
            c, return_features=True, skip_t0=False
        )
        c.features["line_whole"] = {"slope": m1, "intercept": b1}
        m2, b2 = likely_irreversible_linear(
            c, return_features=True, skip_t0=True
        )
        c.features["line_wo_first"] = {"slope": m2, "intercept": b2}
        if results.args.debug and (c.labels is None):
            c.labels = results.cluster_model_fits[step][c.k].labels_ + 1

    # And now, build the table and write it with other results.
    centroid_table = pd.concat([
        tabulate_centroids(
            results.cluster_centroids[step],
            added_columns={'subject': results.args.subject, 'step': step},
        ),
    ], axis=0).reset_index(drop=True)

    filename = f"sub-{results.args.subject}_vasc_clust_step-{step}_metadata.csv"
    # Order columns so subject and step are first
    cols = [c for c in centroid_table.columns if c not in ['subject', 'step']]
    centroid_table[['subject', 'step', ] + cols].to_csv(
        results.args.debug_path / filename, index=False, float_format='%0.5f'
    )

    # It's also very useful to have centroid data for plotting and comparisons
    # so put that in a separate table.
    centroid_data_idx = None
    centroid_data_columns = []
    centroid_data_values = []
    for c in results.cluster_centroids[step]:
        if centroid_data_idx is None:
            centroid_data_idx = c.timepoints
        else:
            if not np.array_equal(centroid_data_idx, c.timepoints):
                print("Warning: Centroids have different timepoints!!")

        centroid_data_columns.append(f"l-{c.label}_k-{c.k}")
        centroid_data_values.append(c.activity)
        if c.best_overall:
            if c.blob_data is None:
                print("WTF!?!? The best centroid has no blob data!")
            else:
                c.blob_data.to_csv(results.args.debug_path / "best_cluster_blobs.csv")

    centroid_data = pd.DataFrame(
        data=np.vstack(centroid_data_values).T,
        index=pd.Index(centroid_data_idx),
        columns=centroid_data_columns
    )
    filename = f"sub-{results.args.subject}_vasc_clust_step-{step}_data.csv"
    centroid_data.to_csv(
        results.args.debug_path / filename,
        index=True, float_format='%0.5f'
    )


def fake_centroid_from_mask(
        new_mask,
        basis_centroid,
        pet_4d_img,
):
    """ Prepare a centroid with fake clustering data from a provided mask

        :param new_mask: Path to the mask or an ndarray containing a mask
        :param basis_centroid: Current centroid, calculated by k-means
        :param pet_4d_img: The data used to calculate basis_centroid
        :return: centroid
    """

    # Load alternate mask and use it to build a fake cluster centroid
    if isinstance(new_mask, str) or isinstance(new_mask, PurePath):
        _fake_3d_mask = nib.Nifti1Image.from_filename(new_mask)
        print(f" |fc| Loaded {_fake_3d_mask.shape}-shaped mask from disk ('{new_mask}')")
    else:
        _fake_3d_mask = nib.Nifti1Image(new_mask, pet_4d_img.affine)
        print(f" |fc| Made {new_mask.shape}-shaped new_mask into a Nifti1Image")
    if _fake_3d_mask.shape != basis_centroid.original_shape[:3]:
        _fake_3d_mask = image.resample_img(
            _fake_3d_mask,
            target_affine=basis_centroid.original_affine,
            interpolation='nearest',
            target_shape=basis_centroid.original_shape[:3],
        )
        print(f" |fc| Resampled mask to {_fake_3d_mask.shape} with "
              f"{np.sum(_fake_3d_mask):,} hot voxels.")
    _fake_flat_mask = np.squeeze(flatten_4d_to_2d(
        np.expand_dims(_fake_3d_mask.get_fdata(), 3),
    )).astype(bool)
    print(f" |fc| Flattened mask to {_fake_flat_mask.shape} with "
          f"{np.sum(_fake_flat_mask):,} hot voxels.")
    # For a pre-existing centroid, keep the same label, and match the mask.
    # _fake_flat_mask = np.multiply(
    #     _fake_flat_mask, basis_centroid.label
    # )
    # Create a centroid based on the fake override mask.
    # curr_2d_data = flatten_4d_to_2d(pet_4d_img.get_fdata())
    new_centroid = copy.deepcopy(basis_centroid)
    # masked_2d_data = np.zeros(curr_2d_data.shape)  # a millionish x 25ish
    masked_2d_data = flatten_4d_to_2d(pet_4d_img.get_fdata())[_fake_flat_mask, :]
    new_centroid.activity = np.mean(masked_2d_data, axis=0)
    new_centroid.source = "manual override"
    new_centroid.labels = _fake_flat_mask.astype(np.uint8) * new_centroid.label
    new_centroid.update_spatial_clusters(force_update=True)

    print(f" |fc| [{', '.join([f'{a:0.1f}' for a in new_centroid.activity])}]")
    print(f" |fc| {new_centroid.blob_data.shape}-shaped blob_data")

    # Return the real data, masked by the new mask, to be fed into step two.

    return new_centroid, masked_2d_data


def resample_for_clustering(original_image, resample_string, logger=None):
    """ Resample original data into new resolution specified by resample_string

    :param Nifti1Image original_image: Input image needing resampling
    :param resample_string: command line argument specifying resampling
    :param logger: logger object
    :return: resampled image

    """

    if resample_string == "":
        target_affine = None
    elif resample_string == "2mm":
        target_affine = np.diag((2.0, 2.0, 2.0, ))
    elif resample_string == "3mm":
        target_affine = np.diag((3.0, 3.0, 3.0, ))
    elif resample_string == "4mm":
        target_affine = np.diag((4.0, 4.0, 4.0, ))
    elif resample_string == "2x":
        target_affine = original_image.affine.copy()
        target_affine[:3, :3] = original_image.affine[:3, :3] * 2.0
    else:
        target_affine = None
        warning = (f"The resampling method '{resample_string}' is not "
                   "recognized or supported. STARE will continue with "
                   "the original data in its original resolution.")
        if logger:
            logger.warning(warning)
        else:
            print(warning)

    if target_affine is not None:
        if resample_string == "2x":
            # I checked with a toy dataset and discovered that nilearn's
            # resampling doesn't do any averaging or smoothing of collapsed
            # voxels. So to do a 2x resample, it just uses one of the 8
            # values in the 2x2x2 array it collapses. And it does the same
            # thing with any of the three interpolation options! With noisy
            # data, we'll get a lot more benefit from averaging the values
            # within the 2x2x2 block. So I wrote a function to do that.
            # (see adhoc_test_resampling_with_fake_data.ipynb)
            resampled_data = collapse_array_3d(original_image.get_fdata(), by=2)
            return (
                nib.nifti1.Nifti1Image(resampled_data, affine=target_affine, ),
                True
            )
        else:
            return (
                image.resample_img(original_image, target_affine=target_affine),
                True,
            )
    else:
        return original_image, False


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

    # Just for debug/curiosity, we can also cluster via PCA and ICA.
    # This just saves some component maps, doesn't affect anything else.
    if results.args.decompose_components:
        decompose_components(results, logger)

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

    # Some higher-resolution images are difficult to cluster, and take forever,
    # so they perform better down-sampled. IF resampling was requested on the
    # command line, the cluster masks will be down-sampled as requested, then
    # up-sampled back to original resolution after k-means.
    # This down-sampling is easy, done once below. The reversal back to original
    # space is done while saving out
    curr_4d_pet_img, data_are_resampled = resample_for_clustering(
        results.cropped_4D, results.args.resample_for_clustering, logger=logger
    )
    orig_3d_pet_img = image.mean_img(results.input_4D, copy_header=True)
    crop_3d_pet_img = image.mean_img(results.cropped_4D, copy_header=True)
    curr_3d_pet_img = image.mean_img(curr_4d_pet_img, copy_header=True)

    # Have somewhere to store results from step one and two clustering
    k_means_results = dict()

    # -------------------------------------------------------------------------
    # Step 1. Find the best candidate for a vascular cluster of voxels.
    #         The first step tries 10 values of k between 6 and 40,
    #         and selects the best cluster
    # -------------------------------------------------------------------------

    k_means_results[1] = load_or_calculate_clusters(
        results, cluster_function, curr_4d_pet_img,
        step_one_ks, 1, rpt_sect, logger=logger
    )
    rpt_sect.add_line(str(k_means_results[1]['best_centroid']))
    save_table_of_centroid_stats(results, 1)
    step_1_atlas_path, step_1_mask_path = make_atlas_and_mask(
        k_means_results[1]['best_centroid'],
        curr_3d_pet_img, out_path=results.args.output_path / "masks",
        file_desc=f"step-{1}_best",
        resample_to=crop_3d_pet_img if data_are_resampled else None,
        logger=logger,
    )
    results.best_vascular_mask_path[1] = step_1_mask_path
    if results.args.axial_slices_to_clip > 0:
        step_1_atlas_path, step_1_mask_path = make_atlas_and_mask(
            k_means_results[1]['best_centroid'],
            curr_3d_pet_img, out_path=results.args.output_path / "masks",
            file_desc=f"step-{1}_best",
            resample_to=crop_3d_pet_img if data_are_resampled else None,
            pad_to=orig_3d_pet_img,
            logger=logger,
        )
        results.best_vascular_mask_path[1] = step_1_mask_path

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

    # Run the second k-means, but only on the best cluster from the first.
    # If prior models were saved to disk, load them rather than running.
    k_means_results[2] = load_or_calculate_clusters(
        results, cluster_function, k_means_results[1]['best_cluster_as_image'],
        step_two_ks, 2, rpt_sect, logger=logger
    )
    rpt_sect.add_line(str(best_of(results.cluster_centroids[2])))
    save_table_of_centroid_stats(results, 2)
    step_2_atlas_path, step_2_mask_path = make_atlas_and_mask(
        k_means_results[2]['best_centroid'],
        curr_3d_pet_img, out_path=results.args.output_path / "masks",
        file_desc=f"step-2_best",
        resample_to=crop_3d_pet_img if data_are_resampled else None,
        logger=logger,
    )
    results.best_vascular_mask_path[2] = step_2_mask_path
    if results.args.axial_slices_to_clip > 0:
        step_2_atlas_path, step_2_mask_path = make_atlas_and_mask(
            k_means_results[2]['best_centroid'],
            curr_3d_pet_img, out_path=results.args.output_path / "masks",
            file_desc=f"step-2_best",
            resample_to=crop_3d_pet_img if data_are_resampled else None,
            pad_to=orig_3d_pet_img,
            logger=logger,
        )
        results.best_vascular_mask_path[2] = step_2_mask_path

    # Plot the top centroids over PET data, and add it to the report.
    best_centroid_1 = k_means_results[1]['best_centroid']
    best_centroid_2 = k_means_results[2]['best_centroid']
    top_centroid_fig = plot_top_centroids_atlas(
        k_means_results[1]['best_cluster_as_image'],
        k_means_results[2]['best_cluster_as_image'],
        curr_3d_pet_img,
        title="\n".join([
            f"{results.args.subject}:",
            f"step 1. orange. {best_centroid_1.label} of {best_centroid_1.k}",
            f"step 2. red. {best_centroid_2.label} of {best_centroid_2.k}",
        ]),
    )
    filename = f"sub-{results.args.subject}_step-1-vascular_cluster_masks.png"
    top_centroid_fig.savefig(results.args.fig_path / filename)
    caption = "Step one (orange) and step two (red) vascular clusters"
    rpt_sect.add_figure(results.args.fig_path / filename, caption)
    plt.close(top_centroid_fig)

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

        # These data can be used to build custom plots or otherwise explore.
        filename = f"sub-{results.args.subject}_step-1-{step}_kmeans_tac.csv"
        tacs_to_plottable_dataframe(results.cluster_centroids[step]).to_csv(
            results.args.output_path / filename, index=False,
        )
        logger.info(f"WROTE {filename} to {str(results.args.output_path)}")

        # Before pickling, remove redundant copies of the huge label arrays.
        for c in results.cluster_centroids[step]:
            c.labels = None
        filename = "sub-{}_step-{}_kmeans_centroid.pkl".format(
            results.args.subject, step
        )
        # noinspection PyTypeChecker
        pickle.dump(
            best_of(results.cluster_centroids[step]),
            open(results.args.output_path / "debug" / filename, "wb")
        )

        # Save out nifti masks (which ones conditional on verbosity)
        resample_template = curr_3d_pet_img
        if data_are_resampled:
            resample_template = crop_3d_pet_img
        if results.args.save_all_cluster_masks or results.args.verbose > 2:
            save_centroid_masks(
                results.cluster_centroids[step],
                results.cluster_model_fits[step],
                results.args.debug_path,
                curr_3d_pet_img,
                step=step,
                resample_to_template=resample_template,
                logger=logger,
            )
            filename = f"sub-{results.args.subject}_step-{step}_vasc_tacs.png"
            unique_ks = sorted(np.unique([
                c.k for c in results.cluster_centroids[step]
            ]))
            for k in unique_ks:
                cs_in_k = [c for c in results.cluster_centroids[step]
                           if c.k == k]
                # noinspection PyTypeChecker
                pickle.dump(
                    cs_in_k,
                    open(results.args.debug_path / "masks" /
                         f"centroids_step-{step}_k-{k}.pkl",
                         "wb")
                )
                plot_vascular_tacs(
                    cs_in_k, draw_non_vascular=True, tall=True
                ).savefig(
                    results.args.debug_path / "masks" /
                    filename.replace("_vas", f"_k-{k}_vas")
                )

    rpt_sect.end()
    results.write_report()

    # Copy over a script to view the clusters
    try:
        # If 'stare_pet' was installed, this is where we'll find it.
        src_file = files('starelib.scripts').joinpath('view_in_fsleyes.sh')
    except ModuleNotFoundError as e:
        # If 'stare_pet' is running from source, we'll find it here.
        src_file = Path(__file__).parent.parent / "scripts" / "view_in_fsleyes.sh"
    if src_file.exists():
        tgt_file = results.args.output_path / "view_in_fsleyes.sh"
        logger.info("Copying 'view_in_fsleyes.sh' from '{}' to '{}'".format(
            src_file.parent, tgt_file.parent
        ))
        tgt_file.write_text(src_file.read_text())
        tgt_file.chmod(
            tgt_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
    else:
        logger.warning(f"I could not find the 'view_in_fsleyes.sh' script.")

    return results
