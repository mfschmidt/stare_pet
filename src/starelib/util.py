import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
import nibabel as nib
from nibabel.funcs import concat_images
import nilearn.image
from scipy.stats import gaussian_kde
import warnings
import pickle


# Store each image as a namedtuple with more data
Image = namedtuple('Image', 'path filename prefix frame nifti')


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
                    img4d[r, c, d] = labels[i]
                    i += 1
    else:
        img4d = labels.reshape(new_shape)
    return img4d


def combine_volumes_into_4d(volumes, output_file, logger=None):
    """ From a list of 3D volumes, build a 4D image.

    :param list volumes: A list of Image tuples describing volumes
    :param output_file: The path to save the 4D file
    :param logging.Logger logger: A logger for output

    :return: The 4D Nifti1Image object
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    output_file = Path(output_file)

    # Collect all the 3d image data into a single 4d structure.
    combined_image = concat_images(
        [v.nifti for v in sorted(volumes, key=lambda x: int(x.frame))]
    )
    nib.save(combined_image, str(output_file))
    logger.debug(f"WROTE {output_file.name} ({combined_image.shape}) "
                 f"to {str(output_file.parent)}")

    return combined_image


def explode_4d_into_volumes(
        image, out_path, name_template, ignored_volumes=None,
        cached=False, logger=None
):
    """ Save individual 3d volumes from 4d image.

    :param image: 4d nifti image
    :param out_path: path to save separate volumes
    :param name_template: format string for naming volume files
    :param list ignored_volumes: a list of volumes to pass over and not save
    :param bool cached: True if loading a cached 4D image, missing volumes
    :param logger: logger object for writing information
    :return: list of individual volumes
    """

    if ignored_volumes is None:
        ignored_volumes = []
    volumes = []
    write_volumes = True
    nifti_vols = [image.slicer[:, :, :, t] for t in range(image.shape[3])]
    existing_images = list(out_path.glob("*.nii.gz"))
    if len(existing_images) >= len(nifti_vols):
        write_volumes = False
        if logger:
            logger.info(f"found {len(existing_images)} volumes in {out_path}, "
                        "not overwriting.")
    # Whether we write or not, still split and keep in memory.
    ignored_volume_spacer = 0
    for i, nifti_vol in enumerate(nifti_vols):
        if i + 1 in ignored_volumes:
            if cached:
                # skip a volume number, it wasn't in the 4D image anyway
                ignored_volume_spacer += 1
            else:
                # pass over the volume, it is in the 4D image, but not usable
                continue
        image = Image(
            path=out_path,
            filename=name_template.format(i + 1 + ignored_volume_spacer),
            prefix="orig",
            frame=i + 1 + ignored_volume_spacer,
            nifti=nifti_vol,
        )
        if write_volumes:
            image.path.mkdir(parents=True, exist_ok=True)
            nib.save(nifti_vol, image.path / image.filename)
        volumes.append(image)

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
        duration = 2 * (t - start_times[-1])
        end_time = start_times[-1] + duration
        row = {
            "t_start": start_times[-1],
            "t_mid": t,
            "t_end": end_time,
            "duration": duration,
            "used": t in mid_times,
        }
        rows.append(row)
        start_times.append(end_time)

    # An alternative way to calculate 't_end' values uses scipy.signals:
    # end_time_frame = lfilter([2, ], [1, 1, ], mid_times)
    # but even though I watch it return identical values,
    # I don't understand how it works, so I wrote it out here
    # explicitly instead. I guess I need to learn more signal processing.

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
