import sys
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from humanize import ordinal
import pickle

from .timeactivitycurve import TimeActivityCurve
from .util import Image, image_in_millicuries,\
    combine_volumes_into_4d, explode_4d_into_volumes


def get_tsv_data(input_path, subject_id, contents, logger=None):
    """ Find a tsv/txt file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str contents: what kind of data is in the file
    :param logger: where to send messages
    :return pandas.DataFrame: data
    """

    if logger is None:
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


def get_tacs(
        input_path, subject_id, regions, frames_to_ignore=None
):
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


def get_mid_times(input_path, subject_id, frames_to_ignore, logger=None):
    """ Find a mid-times file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: ignored frames to cut from the list
    :param logger: where to send messages
    :return tuple: array of raw mid-times, mid-times w/o ignored frames
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    mid_times = get_tsv_data(
        input_path, subject_id, "midtimes"
    )
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


def get_individual_volumes(
        input_path, output_path, subject_id, frames_to_ignore, logger=None
):
    """ Find images, a volume for each mid-time

    :param input_path: path to find subjects
    :param output_path: path to rewrite subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: list of frame numbers to avoid
    :param logger: where to send messages

    :return dict: key-value dict with image data
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    volumes = []
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
                volumes.append(Image(
                    path=nifti_file.parent,
                    filename=nifti_file.name,
                    prefix="orig",
                    frame=i + 1,
                    nifti=nifti_img,
                ))
    return volumes


def gather_data(results):
    """ Manage the gathering of all input data on disk """

    logger = results.logger
    rpt_sect = results.report.begin_section("Gather Data")

    results.logger.debug(f"{results.name} is running with these arguments.")
    for k, v in vars(results.args).items():
        spaces = " " * (23 - len(k))
        results.logger.debug(f"  '{k}'{spaces}: {v}")
    results.logger.info(f"The command issued: '{' '.join(sys.argv)}'")

    # Assume everything's good until we encounter a problem.
    ok_to_run = True
    args = results.args

    # Read PET TAC data
    tacs = get_tacs(
        args.input_path, args.subject, args.regions, args.ignore_frames
    )
    if tacs is None:
        logger.error("Failed to load TACs")
        ok_to_run = False
    else:
        if len(tacs.columns) < len(args.regions):
            dropped_regions = [r for r in args.regions if r not in tacs.columns]
            logger.warning("Specified regions were NOT found in the TACs:")
            for region in dropped_regions:
                logger.warning(f"   {region}")
        logger.info(f"Running with {len(tacs.columns)} regions:"
                    f"    [{', '.join(tacs.columns)}]")

    # Find and load mid_times
    mid_times, ignored_mid_times = get_mid_times(
        args.input_path, args.subject, args.ignore_frames
    )
    if mid_times is None:
        logger.error("Failed to load midtimes")
        ok_to_run = False

    # Get plasma data if it's available, but this is not required.
    plasma_tac = get_plasma(
        args.input_path, args.subject
    )
    if plasma_tac is None:
        logger.warning("Failed to load plasma TAC. "
                       "STARE will run fine, but cannot compare to plasma.")
    if args.debug:
        pickle.dump(
            plasma_tac,
            open(args.debug_path / "tac_plasma.pkl", "wb")
        )

    # Load PET images
    # The first preference is if there is already a cached 4D image.
    cached_img_file = args.output_path / "orig.nii.gz"
    if cached_img_file.exists():
        # There's one 4D image to load and break up.
        logger.info(f"Reading 4d image '{cached_img_file}'")
        combined_image = nib.load(cached_img_file)
        logger.debug(f"  image contains {combined_image.shape[3]} volumes.")
        # Split the 4d data out into separate volumes.
        volumes = explode_4d_into_volumes(
            combined_image, args.output_path,
            name_template=args.subject + "_{:02d}.nii.gz"
        )
    else:
        # Older FDG data are saved as one Analyze volume per time point.
        # Newer data are saved as a single 4D nifti image.
        # Future data will be BIDS-compliant.
        # We need to support all of these, and also allow for PVC correction
        # of individual volumes later.
        moco_path = Path(args.input_path) / args.subject / "moco"
        moco_images = list(moco_path.glob("*.hdr"))
        moco_images.extend(list(moco_path.glob("*.nii")))
        moco_images.extend(list(moco_path.glob("*.nii.gz")))
        if len(moco_images) > 1:
            # We can load a bunch of volumes.
            volumes = get_individual_volumes(
                args.input_path, args.output_path, args.subject,
                args.ignore_frames, logger=logger
            )
            # Collect all the 3d image data into a single 4d structure.
            combined_image = combine_volumes_into_4d(
                volumes, args.output_path / "orig.nii.gz", logger=logger
            )
        elif len(moco_images) == 1:
            # There's one 4D image to load and break up.
            logger.info(f"Reading 4d image '{moco_images[0]}'")
            combined_image = nib.load(moco_images[0])
            logger.debug(f"  image contains {combined_image.shape[3]} volumes.")
            # Split the 4d data out into separate volumes.
            volumes = explode_4d_into_volumes(
                combined_image, args.output_path,
                name_template=results.subject + "{:03d}.nii.gz"
            )
        else:
            # We don't know how to handle anything else.
            volumes, combined_image = None, None

    if volumes is None:
        logger.error("Failed to load PET image data")
        ok_to_run = False
    if not ok_to_run:
        logger.error("Unable to find sufficient data to run STARE.\n"
                     "See previous errors above to determine what's missing.")
        return 1

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = combined_image.slicer[:, :, args.axial_slices_to_clip:, :]
    nib.save(cropped_image, args.output_path / "orig_cropped.nii.gz")
    logger.debug(f"WROTE orig_cropped.nii.gz ({cropped_image.shape}) "
                 f"to {str(args.output_path)}")
    # cropped_volumes = [cropped_image.slicer[:, :, :, i]
    #                    for i in range(cropped_image.shape[3])]

    # PET data should be in units of 'mCi'
    mci_image = image_in_millicuries(cropped_image, args.pet_units)

    # Store the relevant data to results object.
    results.tacs = tacs
    results.mid_times = mid_times
    results.ignored_mid_times = ignored_mid_times
    results.plasma_tac = plasma_tac
    results.input_4D = combined_image
    results.cropped_4D = mci_image
    results.volume_images = volumes

    rpt_sect.end()
    return results
