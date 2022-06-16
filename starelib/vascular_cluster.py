from pathlib import Path
import numpy as np
import logging
from sklearn.cluster import KMeans
import nibabel as nib
from nibabel.funcs import concat_images
from datetime import datetime
import pickle

from starelib.plotting import centroids_to_plottable_tacs, plot_tacs


def flatten_4d_to_2d(a4d, zxy=True):
    """ Flatten the first 3 dimensions of a 4d image into a 2D matrix

        :param ndarray a4d: The 4d ndarray to flatten
        :param bool zxy: Flatten in z, x, y order
        :returns ndarray: The 2d flattened ndarray
    """

    # For continuity, this pattern matches the original matlab code.
    # It is very likely that a one-line numpy flatten argument will do
    # something comparable with less effort, but that remains to
    # be tested.
    new_shape = (a4d.shape[0] * a4d.shape[1] * a4d.shape[2], a4d.shape[3])
    if zxy:
        a2d = np.zeros(new_shape)
        i = 0
        for d in range(a4d.shape[2]):
            for r in range(a4d.shape[0]):
                for c in range(a4d.shape[1]):
                    a2d[i, :] = a4d[r, c, d, :]
                    i += 1
    else:
        a2d = a4d.reshape(new_shape)
    return a2d


def unflatten_2d_to_4d(a2d, new_shape, zxy=True):
    """ Unflatten a 2d array of voxel-based timeseries to 4d image

        :param ndarray a2d: The 2d ndarray of timeseries
        :param tuple new_shape: The new shape for the timeseries
        :param bool zxy: Unflatten in z, x, y order
        :returns ndarray: The 4d image
    """

    # This unflattening matches the flattening above so it
    # can reverse what was done before. If one function is
    # changed, the other should be changed to match.
    if zxy:
        img4d = np.zeros(new_shape)
        i = 0
        for d in range(new_shape[2]):
            for r in range(new_shape[0]):
                for c in range(new_shape[1]):
                    img4d[r, c, d, :] = a2d[i, :]
                    i += 1
    else:
        img4d = a2d.reshape(new_shape)
    return img4d


def reshape_labels_to_3d(labels, new_shape, zxy=True):
    """ Unflatten an array of labels to 3d image

        :param numpy.array labels: The array of integer labels
        :param tuple new_shape: The new shape for the volume
        :param bool zxy: Unflatten in z, x, y order
        :returns ndarray: The 4d image
    """

    # This unflattening matches the flattening above so it
    # can reverse what was done before. If one function is
    # changed, the other should be changed to match.
    if zxy:
        img4d = np.zeros(new_shape)
        i = 0
        for d in range(new_shape[2]):
            for r in range(new_shape[0]):
                for c in range(new_shape[1]):
                    img4d[r, c, d] = labels[i]
                    i += 1
    else:
        img4d = labels.reshape(new_shape)
    return img4d


def make_atlas_and_mask(centroid, template_img, out_path=None):
    """ Save a centroid's cluster as a mask.

        This function intentionally saves the mask as-is, meaning if
        axial slices were cropped, they are still cropped here. In
        this version of stare_pet, each mask may or may not overlay
        the original PET image correctly.

    :param dict centroid: A dict containing centroid data and metadata
    :param Nifti1Image template_img: Image to use as a template for mask data
    :param Path out_path: If provided, directory for writing out atlas and mask
                          By default, these are not written to disk
    :return: paths to atlas image and mask image
    """

    # Shape the voxel labels into a 3d matrix to match the template image.
    cluster_atlas_data = reshape_labels_to_3d(
        centroid['labels'],
        (template_img.shape[0], template_img.shape[1], template_img.shape[2])
    )
    # Build an atlas and a mask, based on these labels.
    cluster_atlas_img = nib.Nifti1Image(
        cluster_atlas_data.astype(int),
        affine=template_img.get_affine(), header=template_img.get_header()
    )
    cluster_atlas_img.update_header()
    cluster_mask_img = nib.Nifti1Image(
        (cluster_atlas_data == centroid['label']).astype(int),
        affine=template_img.get_affine(), header=template_img.get_header()
    )
    cluster_mask_img.update_header()

    # Write out images if they don't already exist.
    k = centroid['k']
    label = centroid['label']
    atlas_filename = f"cluster_k-{k:02d}_atlas.nii.gz"
    mask_filename = f"cluster_k-{k:02d}_label-{label:02d}_mask.nii.gz"
    if out_path is not None:
        if not (out_path / atlas_filename).exists():
            nib.save(cluster_atlas_img, out_path / atlas_filename)
        if not (out_path / mask_filename).exists():
            nib.save(cluster_mask_img, out_path / mask_filename)

    return cluster_atlas_img, cluster_mask_img


def vascular_clustering(output_path, images, pet_units, axial_slices_to_clip, mid_times, verbose=0):
    """ Perform vascular clustering, step 1 of 6 in the STARE process

        This step is the first of six in the STARE process.
        It concatenates and clips all pre-motion-corrected 3D volumes
        into a single 4D Nifti file. It then flattens those four dimensions
        into a 2D matrix with a timeseries vector for each voxel represented
        in each 3D volume. Those vectors are clustered to find likely
        vascular regions. Centroids from each cluster are then plotted
        before and after centroids most likely to be vascular are
        recognized.

        :param Path output_path: the main output path for one subject
        :param list images: A list of images
        :param str pet_units: 'kBq' or 'Bq', anything else treated as 'mCi'
        :param int axial_slices_to_clip: how many axial slices to remove
        :param list mid_times: A list of images
        :param int verbose: Set to non-zero to trigger logging, higher is more

        :return: None
    """

    # Collect all the 3d image data into a single 4d structure.
    logging.info(f"Merging {len(images)} PET images into a 4D file")
    combined_image = concat_images(
        [img['data'] for img in sorted(images, key=lambda x: int(x.get('frame')))]
    )
    nib.save(combined_image, Path(output_path) / "combined.nii.gz")

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = combined_image.slicer[:, :, axial_slices_to_clip:, :]
    nib.save(cropped_image, output_path / "combined_cropped.nii.gz")
    template_image = cropped_image.slicer[:, :, :, 0]

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

    fig_path = output_path / "anchoring" / "figs-masks"
    fig_path.mkdir(parents=True, exist_ok=True)

    # Step 1. Find the best candidate for a vascular cluster of voxels.

    # If prior models were saved to disk, load them rather than running.
    cache_file_1 = Path(output_path) / "cache" / "centroids_and_fits_step-1.pkl"
    if cache_file_1.exists():
        all_centroids, model_fits = pickle.load(cache_file_1.open("rb"))
    else:
        all_centroids, model_fits = find_vascular_centroids(
            to_cluster, range(6, 40, 4), verbose=verbose
        )
        cache_file_1.parent.mkdir(exist_ok=True)
        pickle.dump((all_centroids, model_fits), cache_file_1.open("wb"))

    # Lengthen data from wide to long for plotting, and plot TACs
    centroid_data = centroids_to_plottable_tacs(
        all_centroids, mid_times
    )
    centroid_data.to_csv(output_path / "debug_centroids_1.csv")
    fig = plot_tacs(centroid_data)
    fig.savefig(fig_path / "tacs_from_kmeans_step_1.png")

    # Step 2. Mask out only the voxels belonging to that cluster.
    (output_path / "cluster_masks").mkdir(exist_ok=True)
    for centroid in all_centroids:
        if (verbose > 1) or centroid['best_overall']:
            make_atlas_and_mask(
                centroid, template_image,
                out_path=output_path / "cluster_masks"
            )

    # Run a second k-means, but only on the best cluster from the first.
    top_centroid = [c for c in all_centroids if c['best_overall']][0]
    top_cluster_mask = top_centroid['labels'] == top_centroid['label']
    # Formerly, we would filter out non-top-data altogether
    # top_centroid_masked_data = to_cluster[top_cluster_mask, :, ]
    # Now we fill it with zeroes instead.
    top_centroid_masked_data = np.zeros(to_cluster.shape, )
    top_centroid_masked_data[top_cluster_mask] = to_cluster[top_cluster_mask, :]

    # If prior models were saved to disk, load them rather than running.
    cache_file_2 = Path(output_path) / "cache" / "centroids_and_fits_step-2.pkl"
    if cache_file_2.exists():
        second_centroids, second_model_fits = pickle.load(cache_file_2.open("rb"))
    else:
        second_centroids, second_model_fits = find_vascular_centroids(
            top_centroid_masked_data, [4, ], verbose=verbose
        )
        cache_file_2.parent.mkdir(exist_ok=True)
        pickle.dump((second_centroids, second_model_fits), cache_file_2.open("wb"))

    # Lengthen data from wide to long for plotting, and plot TACs
    centroid_data_2 = centroids_to_plottable_tacs(
        second_centroids, mid_times
    )
    centroid_data_2.to_csv(output_path / "debug_centroids_2.csv")
    fig = plot_tacs(centroid_data_2)
    fig.savefig(fig_path / "tacs_from_kmeans_step_2.png")

    for centroid in second_centroids:
        if (verbose > 1) or centroid['best_overall']:
            # Before making masks and atlases, we must put all the zeroes back
            # inflated_centroid_labels = np.zeros((len(top_cluster_mask), ), )
            # inflated_centroid_labels[top_cluster_mask] = centroid['labels']
            # centroid['labels'] = inflated_centroid_labels
            make_atlas_and_mask(
                centroid, template_image,
                out_path=output_path / "cluster_masks"
            )

    return None


def find_vascular_centroids(data, ks, verbose=0):
    """ Step 1. From all PET data, find a vascular cluster.

        Loop over all values for k in ks, looking for clusters that
        exhibit vascular-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means
        :param int verbose: Set non-zero to increase logging, higher is more

        :returns tuple: The best TAC, and all the TACs
    """

    # Do k-means clustering of timeseries for many values of k
    # from Matlab vascClust.m:112:158
    pre_k_timestamp = datetime.now()
    k_means_fits = {}
    all_centroids = []
    for k in ks:
        vascular_centroids = []
        other_centroids = []
        logging.info(f"K-means (k={k})")
        pre_1k_timestamp = datetime.now()
        k_means = KMeans(init="k-means++", n_clusters=k,
                         n_init=3, max_iter=1024**2, random_state=42,
                         verbose=verbose, )
        k_means.fit(data)
        k_means_fits[k] = k_means
        post_1k_timestamp = datetime.now()
        logging.info(f"  lowest inertia == {k_means.inertia_:0.0f}"
                     f" after {k_means.n_iter_} iterations"
                     f" in {post_1k_timestamp - pre_1k_timestamp}.")

        # Find reasonable timeseries in the cluster means.
        count_irreversible, count_noise = 0, 0
        for i in range(k_means.n_clusters):
            cc = k_means.cluster_centers_[i]
            this_centroid = {
                "k": k,
                "label": i + 1,  # should be non-zero as zero indicates background
                "peak_value": np.max(cc),
                "peak_index": np.argmax(cc),
                "best_in_k": False,
                "best_overall": False,
                "vascular": False,
                "centroid": cc,
                "labels": k_means.labels_ + 1,
            }
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
                this_centroid["vascular"] = True
                vascular_centroids.append(this_centroid)
            else:
                other_centroids.append(this_centroid)

        logging.debug(f"  Eliminate {count_irreversible} of {k}"
                      " cluster centers as irreversible.")
        logging.debug(f"  Eliminate {count_noise} of {k}"
                      " cluster centers as noise.")
        logging.debug(f"When k == {k},"
                      f"we find {len(vascular_centroids)} vascular centroids.")
        for i, vc in enumerate(vascular_centroids):
            logging.debug(f"  {vc['peak_value']:0.3f} at {vc['peak_index']}")

        # Label the top candidate for a vascular cluster from this clustering.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous or sinus
        if len(vascular_centroids) > 0:
            peak_idxs = np.array([c['peak_index'] for c in vascular_centroids])
            earliest_peak_idxs = np.where(peak_idxs == np.min(peak_idxs))[0]
            highest_early_peak_idx = earliest_peak_idxs[
                np.argmax([vascular_centroids[i]['peak_value']
                           for i in earliest_peak_idxs])
            ]
            vascular_centroids[highest_early_peak_idx]['best_in_k'] = True
            logging.debug(
                "  Best centroid [{}] has peak of {:0.3f} at time idx {}".format(
                    vascular_centroids[highest_early_peak_idx]['label'],
                    vascular_centroids[highest_early_peak_idx]['peak_value'],
                    vascular_centroids[highest_early_peak_idx]['peak_index'],
                )
            )

        plural_string = "" if len(vascular_centroids) == 1 else "s"
        logging.info(f"  found {len(vascular_centroids)} potential vascular"
                     f" cluster{plural_string} with k={k}.")

        # Add centroids from this k to the larger collection.
        # We originally only returned vascular centroids, but now include
        # all centroids, with vascular ones labeled as such within-dict.
        all_centroids = all_centroids + vascular_centroids + other_centroids

    post_k_timestamp = datetime.now()
    logging.info(f"All {len(ks)} k-means finished in "
                 f"{post_k_timestamp - pre_k_timestamp}")

    # Which cluster-centroid timeseries has the highest peak?
    # And where is that peak?
    # from Matlab vascClust.m:160:174
    best_in_k_centroids = [c for c in all_centroids if c['best_in_k']]
    top_indices, top_frequencies = np.unique(
        [c['peak_index'] for c in best_in_k_centroids], return_counts=True
    )
    # Which time point is most likely to have the highest value?
    best_centroid_idx = top_indices[np.argmax(top_frequencies)]

    # Make a list of centroids that peak at the same, most common, time point
    centroids_with_best_idx = [c for c in all_centroids
                               if ((c['peak_index'] == best_centroid_idx)
                                   and c['best_in_k'])]
    # Of those centroids peaking together, which one peaks highest?
    best_centroid = centroids_with_best_idx[
        np.argmax([c['peak_value'] for c in centroids_with_best_idx])
    ]
    # Label the centroid with the highest peak value
    best_centroid['best_overall'] = True
    logging.info(f"The very best cluster is label {best_centroid['label']} "
                 f"from k {best_centroid['k']}.")
    logging.info(f"It peaked at frame {best_centroid['peak_index'] + 1} "
                 f"to a value of {best_centroid['peak_value']}.")

    # Return a list of all centroids, with the best labelled as such.
    return all_centroids, k_means_fits
