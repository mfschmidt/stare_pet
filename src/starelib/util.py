import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
import nibabel as nib
from nibabel.funcs import concat_images
from humanize import ordinal
import nilearn.image

from .timeactivitycurve import TimeActivityCurve


# Store lists of images as lists of namedtuples
Image = namedtuple('Image', 'path filename prefix frame nifti')


def get_tsv_data(input_path, subject_id, contents):
    """ Find a tsv/txt file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str contents: what kind of data is in the file
    :return pandas.DataFrame: data
    """

    logger = logging.getLogger("STARE")

    # Set context for different types of data
    header = 0
    index_col = None
    sep = '\t'
    names = None
    if contents.lower() == "tacs":
        possible_names = [f"{subject_id}.TACs", "tacs.txt", ]
    elif contents.lower() == "plasma":
        possible_names = [f"{subject_id}plasma.txt", "plasma.txt", ]
    elif contents.lower() in ["midtimes", "mid-times", ]:
        possible_names = [f"{subject_id}.raw.midtime.txt", "midtimes.txt", ]
        header = None
        names = ['t', ]
    else:
        logger.error("I do not understand '{contents}' content.")
        logger.error("I can only load 'tacs', 'midtimes', or 'plasma'.")
        return None

    data = None
    subject_dir = Path(input_path) / subject_id
    for f in possible_names:
        actual_f = subject_dir / f
        if actual_f.exists():
            if data is None:
                logger.info(f"Reading tacs file '{actual_f}'")
                data = pd.read_csv(
                    actual_f,
                    header=header, index_col=index_col, sep=sep, names=names,
                )
                logger.debug(f"  {contents} data shaped {data.shape}")
            else:
                logger.warning(f"Ignoring extra {contents} file '{actual_f}'")
    return data


def get_tacs(input_path, subject_id):
    """ Find a tacs file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :return pandas.DataFrame: TACs
    """

    return get_tsv_data(input_path, subject_id, "tacs")


def get_plasma(input_path, subject_id):
    """ Find a plasma file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :return pandas.DataFrame: A Centroid containing plasma activity
    """

    plasma_data = get_tsv_data(input_path, subject_id, "plasma")
    if 'PlasRawY' in plasma_data.columns and 'PlasRawT' in plasma_data.columns:
        return TimeActivityCurve(
            activity=plasma_data['PlasRawY'].values.astype(float),
            timepoints=plasma_data['PlasRawT'].values.astype(float),
            source="plasma",
            name="plasma",
        )
    else:
        # This should not happen
        return None


def get_mid_times(input_path, subject_id, frames_to_ignore):
    """ Find a mid-times file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: ignored frames to cut from the list
    :return tuple: array of raw mid-times, mid-times w/o ignored frames
    """

    logger = logging.getLogger("STARE")
    mid_times = get_tsv_data(input_path, subject_id, "midtimes")
    if (len(frames_to_ignore) > 0) and mid_times is not None:
        logger.warning(
            f"  {len(frames_to_ignore)} frames being removed."
        )
        ignored_mid_times = mid_times[
            mid_times.index.isin([f - 1 for f in frames_to_ignore])
        ]
        # Replace mid_times AFTER the ignored time point has been stored.
        mid_times = mid_times[
            ~mid_times.index.isin([f - 1 for f in frames_to_ignore])
        ]
    else:
        ignored_mid_times = pd.DataFrame(data=[], columns=["t", ], dtype=float)

    if mid_times is not None and 't' in mid_times:
        return mid_times['t'].values, ignored_mid_times['t'].values
    else:
        return None, None


def get_images(input_path, output_path, subject_id, frames_to_ignore):
    """ Find images, a volume for each mid-time

    :param input_path: path to find subjects
    :param output_path: path to rewrite subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: list of frame numbers to avoid

    :return dict: key-value dict with image data
    """

    logger = logging.getLogger("STARE")
    images = []
    image_dir = Path(input_path) / subject_id / "moco"
    orig_dir = Path(output_path) / "orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["{subject}.*.MCFI.hdr", "*.nii", "*.nii.gz", ]:
        actual_pattern = pattern.format(subject=subject_id)
        for i, img_file in enumerate(sorted(image_dir.glob(actual_pattern))):
            # Check for named frame numbers, just to warn about misunderstanding
            match = re.search(r"[._-](\d+)[._-]", img_file.name)
            if match and int(match.group(1)) != i + 1:
                logger.warning("Image numbering does not match sort order.")
                logger.warning(f"  '{img_file.name}' "
                               f"is the {ordinal(i + 1)} file, "
                               f"but #{match.group(1)}.")
                logger.warning(f"stare_pet uses sort ordering, #{i + 1}")

            # Store the image if it is not to be ignored.
            if i + 1 in frames_to_ignore:
                logger.warning(f"Frame {i + 1} exists, but is being ignored.")
            else:
                logger.info(f"Reading volume '{img_file}' as frame {i + 1:02d}")
                img = nib.load(img_file)
                logger.debug(f"  frame {i + 1} is shaped "
                             f"{'n/a' if img is None else img.shape}")
                # No matter the original image format, we will save our own
                # copy of each image as a Nifti1 nii.gz for consistency
                # throughout the pipeline.
                nifti_img = nib.Nifti1Image(img.get_fdata(), img.affine)
                nifti_file = orig_dir / f"orig_{i + 1:02d}.nii.gz"
                nib.save(nifti_img, str(nifti_file))
                images.append(Image(
                    path=nifti_file.parent,
                    filename=nifti_file.name,
                    prefix="orig",
                    frame=i + 1,
                    nifti=nifti_img,
                ))
    return images


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


def combine_volumes_into_4d(volumes, output_file, logger=None):
    """ From a list of 3D volumes, build a 4D image.

    :param list volumes: A list of dicts describing volumes
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


def characterize_mid_times(mid_times, missing_mid_times=None, beginning=0.0):
    """ From an array of timing mid-points, return durations and end points.

        This is just a stub function to examine sphinx, autodocumentation,
        and import paths.

        :param Iterable mid_times: A list or array of timing midpoints.
        :param Iterable missing_mid_times: left out timing midpoints.
        :param float beginning: Assumed 0.0, when the first mid-time started

        :return: endpoints, durations, weights
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


def tac_vascular_correction(thing):
    """ A function stub for vascular correction of TACs.

        This is just a stub function to examine sphinx, autodocumentation,
        and import paths.

        :param str thing: the thing to print

        :return: None
    """

    logging.info("Running a stub for TAC vascular correction.")
    return thing


def boot_anchor(thing):
    """ A function stub for bootstrap anchoring.

        This is just a stub function to examine sphinx, autodocumentation,
        and import paths.

        :param str thing: the thing to print

        :return: None
    """

    logging.info("Running a stub for bootstrap anchoring.")
    return thing


def minimize_cost_function(thing):
    """ A function stub for minimizing the cost function.

        This is just a stub function to examine sphinx, autodocumentation,
        and import paths.

        :param str thing: the thing to print

        :return: None
    """

    logging.info("Running a stub for cost function minimization.")
    return thing


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
