import logging
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from nibabel.funcs import concat_images
import nilearn.image
from scipy.stats import gaussian_kde
import warnings
import pickle
import re
from nibabel import Nifti1Image
from sklearn.metrics.pairwise import cosine_similarity
from nilearn.image import coord_transform

class StareVolume:
    """ Wrap a Nifti1Image with additional metadata needed for STARE """

    def __init__(self, nifti, path, filename, prefix, frame, usable):
        self.nifti = nifti
        self.path = path
        self.filename = filename
        self.prefix = prefix
        self.frame = frame
        self.usable = usable

    def save_nifti(self):
        nib.save(self.nifti, self.path / self.filename)

    def set_affine(self, new_affine):
        new_img = Nifti1Image(self.nifti.dataobj, new_affine)
        self.nifti = new_img


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

    # This un-flattening matches the flattening above to
    # reverse what was done before. If one function is
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
        :returns ArrayLike: The 4d image
    """

    # This un-flattening matches the flattening above to
    # reverse what was done before. If one function is
    # changed, the other should be changed to match.
    if len(new_shape) > 3:
        new_shape = (new_shape[0], new_shape[1], new_shape[2])
    if len(new_shape) < 3:
        raise ValueError("New shape must be at least 3 dimensional")
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


def combine_volumes_into_4d(volumes, output_file, alerts=None, logger=None):
    """ From a list of 3D volumes, build a 4D image.

    :param list volumes: A list of Image tuples describing volumes
    :param output_file: The path to save the 4D file
    :param alerts: If not None, a list for passing back alerts
    :param logging.Logger logger: A logger for output

    :return: The 4D Nifti1Image object
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    output_file = Path(output_file)

    # Collect all the 3d image data into a single 4d structure.
    # First, make sure the volumes are all aligned.
    # Numpy's np.unique doesn't work for arrays, so we have to roll our own.
    affine_values, affine_counts = [], []
    for vi, vol in enumerate(volumes):
        affine_found = False
        for ai, a in enumerate(affine_values):
            if np.array_equal(a, vol.nifti.affine):
                affine_counts[ai] += 1
                affine_found = True
        if not affine_found:
            affine_values.append(vol.nifti.affine)
            affine_counts.append(1)
    if len(affine_counts) > 1:
        warn_fxn = logger.warning if alerts is None else alerts.append
        warn_fxn(f"Mismatched affines, potentially misaligned data!")
        for i in range(len(affine_counts)):
            warn_fxn(f"{affine_counts[i]} volumes with affine:")
            warn_fxn(str(affine_values[i]))
        for vol in volumes:
            vol.set_affine(affine_values[np.argmax(affine_counts)])
    combined_image = concat_images(
        [v.nifti for v in sorted(volumes, key=lambda x: int(x.frame))]
    )

    nib.save(combined_image, str(output_file))
    logger.debug(f"WROTE {output_file.name} ({combined_image.shape}) "
                 f"to {str(output_file.parent)}")

    return combined_image


def explode_4d_into_volumes(
        image, out_path, name_template,
        ignored_volumes=None, logger=None
):
    """ Save individual 3d volumes from 4d image.

    :param image: 4d nifti image
    :param out_path: path to save separate volumes
    :param name_template: format string for naming volume files
    :param list ignored_volumes: a list of volumes to pass over and not use
    :param logger: logger object for writing information
    :return: list of individual volumes
    """

    ignored_volumes = [] if ignored_volumes is None else ignored_volumes
    volumes = []
    write_volumes = True
    existing_images = list(out_path.glob("*.nii.gz"))
    if len(existing_images) >= image.shape[3]:
        write_volumes = False
        if logger:
            logger.info(f"found {len(existing_images)} volumes in {out_path}, "
                        "not overwriting.")
    # Whether we write or not, still split and keep in memory.
    for i, nifti_vol in enumerate([
        image.slicer[:, :, :, t] for t in range(image.shape[3])
    ]):
        vol = StareVolume(
            nifti=nifti_vol,
            path=out_path,
            filename=name_template.format(i + 1),
            prefix="orig",
            frame=i + 1,
            usable=((i + 1) not in ignored_volumes),
        )
        if write_volumes:
            vol.path.mkdir(parents=True, exist_ok=True)
            vol.save_nifti()
        volumes.append(vol)

    return volumes


def characterize_mid_times(mid_times, missing_mid_times=None, beginning=0.0):
    """ From an array of timing mid-points, return durations and end points.

        We are typically provided a list of mid-times representing the time
        of each PET acquisition. But PET volumes are acquired with
        different amounts of time between each one, so the TAC is not
        sampled linearly. We can use those mid-times to also
        calculate how much time span each volume represents when the
        scans are spaced differently. This function calculates those data
        and returns a pandas dataframe containing them for each point
        in time.

        :param Iterable mid_times: A list or array of timing midpoints.
        :param Iterable missing_mid_times: left out timing midpoints.
        :param float beginning: Assumed 0.0, when the first mid-time started

        :return pandas.DataFrame: data about each time point in mid_times
    """

    # Finalize the complete mid_times vector
    if missing_mid_times is None:
        all_times = sorted(set(mid_times))
    else:
        all_times = sorted(set(np.concatenate([
            mid_times, missing_mid_times,
        ])))

    # Loop over mid_times, filling in start, end, and duration for each.
    start_times = [beginning, ]
    rows = []
    for i, t in enumerate(all_times):
        if i < 1:
            last_t = 0.0
        else:
            last_t = all_times[i - 1]
        if i > len(all_times) - 2:
            next_t = 999999999.9
        else:
            next_t = all_times[i + 1]
        duration = np.min([t - last_t, next_t - t, ])
        # end_time = start_times[-1] + duration
        start_time = t - (duration / 2.0)
        end_time = t + (duration / 2.0)
        row = {
            "t_start": start_time,
            "t_mid": t,
            "t_end": end_time,
            "duration": duration,
            "used": t in mid_times,
        }
        rows.append(row)
        start_times.append(t)

    # An alternative way to calculate 't_end' values uses scipy.signals:
    # end_time_frame = lfilter([2, ], [1, 1, ], mid_times)
    # but even though I watch it return identical values,
    # I don't understand how it works, so I wrote it out here
    # explicitly instead. I guess I need to learn more signal processing.
    # I've also noticed that for cases with a missing frame, the linear
    # filter can produce oscillating results (that still work OK), and a
    # manual method that doesn't min the differences can create negative
    # durations (that don't work at all).

    return pd.DataFrame(rows)


def image_in_millicuries(image, units):
    """ Return an image in millicuries from existing units.

        :param Nifti1Image image: The current image
        :param str units: The current image's units
    """

    # If they already are in millicuries, good,
    # but other units get converted here.
    if units.lower() == "kbq":
        return nilearn.image.math_img('a / 37000', a=image)
    elif units.lower() == "bq":
        return nilearn.image.math_img('a / 37000000', a=image)
    else:
        return image


def get_kde_fwhm_points(values, stat='count', num_bootstraps=1000):
    """ From a collection of values, return fwhm and peak x,y """

    # Estimate the density curve
    kde = gaussian_kde(values)

    # Create x and y values based on that curve at specified resolution
    kde_x = np.linspace(np.min(values), np.max(values), num=num_bootstraps)
    kde_y = kde(kde_x)
    if stat == 'probability':
        kde_y = (num_bootstraps * kde_y) / (np.sum(kde_y) * 100)
    elif stat != 'count':
        warnings.warn(f"In get_kde_fwhm_points, stat '{stat}' "
                      "is not supported. Defaulting to 'count'.")

    # Figure out where three critical points are
    max_y = np.max(kde_y)
    max_y_idx = np.argmax(kde_y)
    xs_over_half_max = np.array([
        val for idx, val in enumerate(kde_x) if kde_y[idx] > max_y / 2.0
    ])

    # Store the full-width-half-max values, and the peak with its index
    xy_trio = np.array([
        [np.min(xs_over_half_max), max_y / 2.0, ],
        [kde_x[max_y_idx], max_y, ],
        [np.max(xs_over_half_max), max_y / 2.0, ]
    ])

    return xy_trio, kde_x, kde_y


def from_cache(cache_path, filename, force=False):
    """ Look for cached data, return it if available.
    """

    thing, cache_file = None, None

    if cache_path is not None and cache_path.exists():
        cache_file = cache_path / filename
    if cache_file is not None and cache_file.exists() and not force:
        with open(cache_file, "rb") as f:
            thing = pickle.load(f)

    return thing


def to_cache(thing, cache_path, filename):
    cache_file = cache_path / filename
    with cache_file.open("wb") as f:
        pickle.dump(thing, f)


def get_cluster_blobs(array_3d, label=1, max_gap=1, verbose=0, messages=None):
    """Find connected blobs in array_3d"""

    _voxels_in_mask = []
    _blobs = {}
    voxels_added_by_scan = 0
    voxels_added_recursively = 0

    def add_voxel(loc):
        """for any voxel, find which blob it's in, then add it to the list"""
        nonlocal voxels_added_recursively

        if loc in _blobs:
            print("false alarm")
            return

        # First pass through the searchlight, are we near an existing blob?
        # If a nearby mask member is labeled, adopt this label
        still_looking, current_blob_id = True, None
        for _x in range(loc[0] - max_gap, loc[0] + max_gap + 1):
            for _y in range(loc[1] - max_gap, loc[1] + max_gap + 1):
                for _z in range(loc[2] - max_gap, loc[2] + max_gap + 1):
                    if (
                            still_looking
                            and ((_x, _y, _z) in _blobs)
                            and (_x >= 0)
                            and (_x < array_3d.shape[0])
                            and (_y >= 0)
                            and (_y < array_3d.shape[1])
                            and (_z >= 0)
                            and (_z < array_3d.shape[2])
                    ):
                        current_blob_id = _blobs[(_x, _y, _z)]
                        # We know our blob; we can stop cycling through
                        still_looking = False

        if still_looking:
            # No neighbors are yet recorded; this is a new blob
            if len(_blobs) == 0:
                max_blob = 0
            else:
                max_blob = np.max([v for k, v in _blobs.items()])
            current_blob_id = max_blob + 1
            # if verbose:
            #     print(f" new blob, #{current_blob_id}")

        # label the voxel we've been asked to add
        _blobs[loc] = current_blob_id

        # Second pass, label all in-mask voxels
        for _x in range(loc[0] - max_gap, loc[0] + max_gap + 1):
            for _y in range(loc[1] - max_gap, loc[1] + max_gap + 1):
                for _z in range(loc[2] - max_gap, loc[2] + max_gap + 1):
                    try:
                        if (
                                ((_x, _y, _z) not in _blobs)
                                and (array_3d[_x, _y, _z] == label)
                                and (_x >= 0)
                                and (_x < array_3d.shape[0])
                                and (_y >= 0)
                                and (_y < array_3d.shape[1])
                                and (_z >= 0)
                                and (_z < array_3d.shape[2])
                        ):
                            # _blobs[(_x, _y, _z)] = current_blob_id
                            # This voxel is in the mask, but not yet labeled
                            # expand outward, seeking more voxels within-blob
                            voxels_added_recursively += 1
                            add_voxel((_x, _y, _z))
                    except IndexError:
                        # No problem, we're searching beyond the array
                        # boundaries and don't need to look here anyway
                        voxels_added_recursively -= 1
                        pass
                    except RecursionError:
                        # We got pretty deep following this voxel's trail.
                        # Pick it up on the next one.
                        voxels_added_recursively -= 1
                        pass
        return  # from add_voxel, not get_cluster_blobs

    # Run through every voxel, adding it to the list if it's in the mask
    for x in range(array_3d.shape[0]):
        for y in range(array_3d.shape[1]):
            for z in range(array_3d.shape[2]):
                if array_3d[x, y, z] == label:
                    _voxels_in_mask.append((x, y, z))

    # Run through only in-mask voxels, adding them to a numbered blob.
    # print(f"Label {label} has {len(_voxels_in_mask):,} voxels.")
    for x, y, z in _voxels_in_mask:
        if (x, y, z) not in _blobs:
            voxels_added_by_scan += 1
            # if verbose:
            #     print(f"Adding {voxels_added_by_scan}. ({x}, {y}, {z})")
            add_voxel((x, y, z))

    # All in-mask voxels have been added,
    # now organize them into a DataFrame for easy analyses
    _labels, _counts = np.unique(array_3d, return_counts=True)
    count_message = "; ".join([f"#{int(_labels[i])} has {_counts[i]:,}" for i in range(len(_labels))])
    messages.append(f"Found {len(_blobs.keys())} blobs for label {label}: [{count_message}]")
    if len(_blobs.keys()) < 1:
        messages.append(  f"nothing to store (#{label}); bailing out")
        return None, list(), list()

    blob_data = pd.DataFrame(
        [
            {
                "blob": blob,
                "gap": max_gap,
                "x": locus[0],
                "y": locus[1],
                "z": locus[2],
            }
            for locus, blob in _blobs.items()
        ]
    )
    blob_ids, voxel_counts = np.unique(blob_data["blob"], return_counts=True)
    if (verbose > 0) and (messages is not None):
        messages.append(
            f"Label {label}: {len(blob_ids):,} blobs "
            f"with {np.mean(voxel_counts):0,.1f} voxels each"
        )
    if (verbose > 1) and (messages is not None):
        messages.append(
            f"  found {len(blob_data):,} voxels, grouped them into "
            f"{len(blob_data['blob'].unique()):,} blobs "
            f"with max gap of {max_gap}."
        )
        messages.append(
            f"  {voxels_added_by_scan:,} voxels were added while scanning, "
            f"{voxels_added_recursively:,} were added recursively."
        )

    return blob_data, blob_ids, voxel_counts


def dice_coef(y_true, y_pred):
    """ Calculate one scalar Dice's coefficient for two binary vectors. """

    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0.0:
        # Comparing two all-zero vectors would be an odd choice, but handle it.
        return 1.0
    else:
        return 2.0 * intersection / denominator


def dice_similarity(mat_a, mat_b):
    """ Calculate Dice's coefficients across each row of each matrix. """

    dice_mat = np.zeros((mat_a.shape[0], mat_b.shape[0]))
    for row_a in range(mat_a.shape[0]):
        for row_b in range(mat_b.shape[0]):
            dice_mat[row_a, row_b] = dice_coef(
                mat_b[row_b, :], mat_a[row_a, :]
            )
            # dice_mat[row_b, row_a] = dice_mat[row_a, row_b]
    return dice_mat


def cos_sim_coef(y_true, y_pred):
    return cosine_similarity(
        y_true.reshape(1, -1), y_pred.reshape(1, -1)
    )[0, -1]


def mask_matrix(labels):
    """ Build a binary vector mask for each label in labels,
        then stack them into a matrix.
    """
    mask_vectors = list()
    for label in sorted(np.unique(labels)):
        mask_vectors.append(
            np.array(labels == label).astype(int).reshape(1, -1)
        )
    return np.vstack(mask_vectors)


def get_mask_file(result_path, step=2, platform="unknown", verbose=False):
    """ From a STARE run path, find the best vascular cluster. """

    candidates = []
    print(f"  searching '{result_path}'...")
    if platform.lower().startswith("matlab"):
        candidates = [_p for _p in result_path.glob(
            f"anchoring/figs-masks/Step{step}_*clusters_Vasc_only_mask-ind*.nii"
        ) if "ORIGINAL" not in str(_p)]
    elif platform.lower().startswith("python"):
        candidates = list(result_path.glob(
            f"masks/cluster_step-{step}_best_mask.nii.gz"
        ))
    else:
        print(f"ERROR: directory neither matlab nor python.")
    if len(candidates) == 1:
        if verbose:
            print(f"  found mask at '{candidates[0]}'")
        return candidates[0]
    elif len(candidates) > 1:
        print(f"WARNING: Multiple masks available!")
        for candidate in candidates:
            print(f"  {str(candidate)}")
        if verbose:
            print(f"  found mask at '{candidates[0]}'")
        return candidates[0]
    else:
        print(f"ERROR: No masks found in '{str(result_path)}'!")
        return None


def get_mask(mask_path, verbose=False):
    """ Read the 'mask_path' file and return its contents """

    img = nib.Nifti1Image.from_filename(mask_path)
    if verbose:
        print(f"  loaded {img.shape}-shaped array from '{mask_path.name}'.")
    return img, np.asarray(img.get_fdata()).astype(bool)


""" Look through the python outputs, load each averaged PET file, and each
    'best' mask, and Gjertrud's best mask for the same subject. Stack them
    all up on the same plot and save it for review. """


def alternate_selected(stare_output_path):
    """ Find the log file, extract override lines, return override info. """
    pat = (
        r"centroid ([0-9]+)/([0-9]+): peak=([0-9.]+) @ t=([0-9]+)/([0-9]+), "
        r"([0-9]+) blobs w/~([0-9.]+) voxels"
    )
    pattern = re.compile('alternate.*' + pat + r".*" + pat)
    alternates = dict()
    for log_path in stare_output_path.glob("stare*.log"):
        with open(log_path, "r") as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    if match.group(2) == '4':
                        clust_step = 2
                    else:
                        clust_step = 1
                    alternates[clust_step] = {
                        'str': match.group(0)[12:],
                        'orig': {
                            'str': match.string[
                                       match.regs[0][0]:match.regs[7][1] + 11
                                   ],
                            'label': int(match.group(1)),
                            'k': int(match.group(2)),
                            'peak': float(match.group(3)),
                            't': int(match.group(4)),
                            't_len': int(match.group(5)),
                            'blobs': int(match.group(6)),
                            'vox_per_blob': float(match.group(7)),
                        },
                        'final': {
                            'str': match.string[
                                       match.regs[8][0] - 9:match.regs[-1][-1]
                                   ],
                            'label': int(match.group(8)),
                            'k': int(match.group(9)),
                            'peak': float(match.group(10)),
                            't': int(match.group(11)),
                            't_len': int(match.group(12)),
                            'blobs': int(match.group(13)),
                            'vox_per_blob': float(match.group(14)),
                        },
                    }
    return alternates


def collapse_slices_3d(data, by=2, along=2):
    """ With array data, average 'by' slices at a time over the 'along' axis.
    """

    assert data.ndim >= along
    assert along < data.ndim, \
        f"'along' is {along}, but data only have {data.ndim} axes."

    # New array for storing collapsed data
    new_dim = int(np.ceil(data.shape[along] / 2.0))
    _shape = (*data.shape[:along], new_dim, *data.shape[along + 1:])
    _data = np.zeros(_shape)
    # print(f"reshaping data from {data.shape} to {_shape}")

    # Collapse data, one slice at-a-time
    for i in range(new_dim):
        # set range of z values to average, without going outside the array.
        i_beg = i * by
        i_end = i_beg + by
        if i_end > data.shape[along]:
            i_end = data.shape[along]

        # Collapse 'by' slices of data
        _slice = None
        if along == 0:
            _slice = np.mean(data[i_beg:i_end, :, :], axis=along)
            _data[i, :, :] = _slice
        elif along == 1:
            _slice = np.mean(data[:, i_beg:i_end, :], axis=along)
            _data[:, i, :] = _slice
        elif along == 2:
            _slice = np.mean(data[:, :, i_beg:i_end], axis=along)
            _data[:, :, i] = _slice
        # print(f"  {i}: {'None' if _slice is None else _slice.shape}")

    return _data


def collapse_array_3d(data, by=2):
    """ With array data, average 'by' slices at a time over all dimensions.
    """

    _data = collapse_slices_3d(data, by=by, along=2)
    _data = collapse_slices_3d(_data, by=by, along=1)
    _data = collapse_slices_3d(_data, by=by, along=0)
    return _data


def stat_str(a):
    """ Build a descriptive string about the values in the array provided.

        :param np.array a: Any numeric array to be summarized
    """

    _mu, _sd = np.mean(a), np.std(a)
    _lo, _hi = np.min(a), np.max(a)
    lo_95 = _mu - _sd - _sd
    hi_95 = _mu + _sd + _sd
    return (
        f"mean {_mu:0.1f} +/- sd {_sd:0.1f}, "
        f"95% [{lo_95:0.1f} to {hi_95:0.1f}] "
        f"range [{_lo:0.1f} to {_hi:0.1f}]"
    )


def get_s_i_axis(img_shape, img_affine):
    """ For an image with the given shape and affine, which axis is inf-sup?

        If the third of three axes is the inferior-superior axis, or S+,
        get_s_i_axis will return (2, 1). If it's superior-inferior, or S-,
        it will return (2, -1). Other orderings are also available.
    """

    # Calculate the world extents
    x0, y0, z0 = coord_transform(0, 0, 0, img_affine)
    x1, y1, z1 = coord_transform(
        img_shape[0], img_shape[1], img_shape[2], img_affine
    )
    min_z, max_z = min(z0, z1), max(z0, z1)

    img_center_ijk = np.array([
        int(img_shape[0] / 2),
        int(img_shape[1] / 2),
        int(img_shape[2] / 2)
    ])
    for ax in (2, 1, 0):
        ijk = img_center_ijk.copy()
        ijk[ax] = 0
        world_coords = coord_transform(ijk[0], ijk[1], ijk[2], img_affine)
        if world_coords[ax] == min_z:
            return ax, 1
        elif world_coords[ax] == max_z:
            return ax, -1
