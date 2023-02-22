import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
import nibabel as nib
from humanize import ordinal

from starelib.timeactivitycurve import TimeActivityCurve


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


def get_tacs(input_path, subject_id, regions, frames_to_ignore=None):
    """ Find a tacs file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param iterable regions: list of regions to include from loaded TACs
    :param iterable frames_to_ignore: rows to drop from the loaded TACs
    :return pandas.DataFrame: TACs
    """

    tac_df = get_tsv_data(input_path, subject_id, "tacs")
    if tac_df is None:
        return None
    good_regions = [r for r in regions if r in tac_df.columns]
    if frames_to_ignore is not None and len(frames_to_ignore) > 0:
        return tac_df.drop(
            np.asarray(frames_to_ignore) - 1, axis=0
        )[good_regions]
    else:
        return tac_df[good_regions]


def get_plasma(input_path, subject_id):
    """ Find a plasma file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :return pandas.DataFrame: A Centroid containing plasma activity
    """

    plasma_data = get_tsv_data(input_path, subject_id, "plasma")
    if plasma_data is None:
        return None
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
