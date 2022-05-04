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
    """ Flatten a 4d image into all voxels by timeseries

        :param ndarray a4d: The 4d ndarray to flatten
        :param bool zxy: Flatten in z, x, y order
        :returns ndarray: The 2d flattened ndarray
    """

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


def vascular_clustering(output_path, images, pet_units, axial_slices_to_clip, mid_times):
    """ A function stub for vascular clustering.

        This is just a stub function to examine sphinx, autodocumentation,
        and import paths.

        :param Path output_path: the main output path for one subject
        :param list images: A list of images
        :param str pet_units: 'kBq' or 'Bq', anything else treated as 'mCi'
        :param int axial_slices_to_clip: how many axial slices to remove
        :param list mid_times: A list of images

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
    vascular_centroids = find_vascular_centroids(
        to_cluster, range(6, 40, 4)
    )
    # for debugging the formatter and plotter:
    pickle.dump((vascular_centroids, mid_times),
                open(output_path / "debug_centroids_1.pkl", "wb"))
    centroid_data = centroids_to_plottable_tacs(
        vascular_centroids, mid_times
    )
    centroid_data.to_csv(output_path / "debug_centroids_1.csv")
    fig = plot_tacs(centroid_data)
    fig.savefig(fig_path / "tacs_from_kmeans_step_1.png")

    # Step 2. Mask out only the voxels belonging to that cluster.

    top_centroid = [c for c in vascular_centroids if c['best_overall']][0]
    cluster_atlas = unflatten_2d_to_4d(
        top_centroid['labels'],
        (pet_4d_data.shape[0], pet_4d_data.shape[1], pet_4d_data.shape[2])
    )
    vascular_cluster = to_cluster[top_centroid['labels'] == top_centroid['label']]
    something_else = find_vascular_tac(vascular_cluster, range(6, 40, 4))

    return something_else


def find_vascular_centroids(data, ks):
    """ Step 1. From all PET data, find a vascular cluster.

        Loop over all values for k in ks, looking for clusters that
        exhibit vascular-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means

        :returns tuple: The best TAC, and all the TACs
    """

    # Do k-means clustering of timeseries for many values of k
    pre_k_timestamp = datetime.now()
    all_vascular_centroids = []
    for k in ks:
        vascular_centroids = []
        vascular_centroids_found = 0
        logging.info(f"K-means (k={k})")
        pre_1k_timestamp = datetime.now()
        k_means = KMeans(init="k-means++", n_clusters=k,
                         n_init=3, max_iter=1024**2, random_state=42, )
        k_means.fit(data)
        post_1k_timestamp = datetime.now()
        logging.info(f"  lowest inertia == {k_means.inertia_:0.0f}"
                     f" after {k_means.n_iter_} iterations"
                     f" in {post_1k_timestamp - pre_1k_timestamp}.")

        # Find reasonable timeseries in the cluster means.
        for i in range(k_means.n_clusters):
            cc = k_means.cluster_centers_[i]
            # Rule out timeseries that climb through the end.
            probably_irreversible = cc[-1] == max(cc)
            # Rule out timeseries with negative values beyond time 0.
            probably_noise = np.any(cc[1:] < 0)
            logging.debug(f"  Eliminate {np.sum(probably_irreversible)} of {k}"
                          " cluster centers as irreversible.")
            logging.debug(f"  Eliminate {np.sum(probably_noise)} of {k}"
                          " cluster centers as noise.")
            if not probably_irreversible and not probably_noise:
                # Store the data (cc) and its metadata along with it.
                # The best* fields are unknowable now, will be updated later.
                vascular_centroids.append({
                    "k": k, "label": i, "centroid": cc,
                    "peak_value": np.max(cc),
                    "peak_index": np.argmax(cc),
                    "best_in_k": False, "best_overall": False,
                    "labels": k_means.labels_,
                })
                vascular_centroids_found += 1

        # Label the top candidate for a vascular cluster from this clustering.
        # Higher initial values are more indicative of arterial signal,
        # which is preferable to venous or sinus
        if vascular_centroids_found > 0:
            top_starting_activity = np.max(
                [c['centroid'][0] for c in vascular_centroids]
            )
            for c in vascular_centroids:
                if c['centroid'][0] == top_starting_activity:
                    c['best_in_k'] = True

        plural_string = "" if vascular_centroids_found == 1 else "s"
        logging.info(f"  found {vascular_centroids_found} potential vascular"
                     f" cluster{plural_string} with k={k}.")

        # Add centroids from this k to the larger collection.
        all_vascular_centroids = all_vascular_centroids + vascular_centroids

    post_k_timestamp = datetime.now()
    logging.info(f"All {len(ks)} k-means finished in "
                 f"{post_k_timestamp - pre_k_timestamp}")

    # Which cluster-centroid timeseries has the highest peak?
    # And where is that peak?
    top_indices, top_frequencies = np.unique(
        [c['peak_index'] for c in all_vascular_centroids], return_counts=True
    )
    best_centroid_idx = top_indices[np.argmax(top_frequencies)]
    centroids_with_best_idx = [c for c in all_vascular_centroids
                               if c['peak_index'] == best_centroid_idx]
    best_centroid = centroids_with_best_idx[
        np.argmax([c['peak_value'] for c in centroids_with_best_idx])
    ]
    # Label the centroid with the highest peak value
    for c in all_vascular_centroids:
        if c['k'] == best_centroid['k']:
            if c['label'] == best_centroid['label']:
                c['best_overall'] = True

    # Return a dict containing the best centroid.
    return all_vascular_centroids


def find_vascular_tac(data, ks):
    """ Step 2. From only the most likely vascular cluster, find a TAC.

        Loop over all values for k in ks, looking for clusters that
        exhibit vascular-like properties. Return the best possible
        cluster.

        :param ndarray data: Array of timeseries
        :param iterable ks: Iterable of integers, each used as a k in k-means
    """

    # The best centroid available
    return data
