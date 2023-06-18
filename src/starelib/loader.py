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


def get_tsv_data(input_path, subject_id, contents, tracer, logger=None):
    """ Find a tsv/txt file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str contents: what kind of data is in the file
    :param str tracer: what tracer was injected
    :param logger: where to send messages
    :return pandas.DataFrame: data
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    # Set context for different types of data. These are the defaults.
    header = 0
    index_col = None
    sep = '\t'
    names = None
    subject_dir = Path(input_path) / subject_id
    if contents.lower() == "tacs":
        picnic_tacs = list(subject_dir.glob(
            f"ses-{tracer.lower()}*_tacs/out_file/wmparc_reoriented_tacs.tsv"
        ))
        old_school_tacs = [subject_dir / f"{subject_id}.TACs", ]
        alternate_tacs = [subject_dir / "tacs.txt", ]
        possible_files = picnic_tacs + old_school_tacs + alternate_tacs
    elif contents.lower() == "plasma":
        old_school_plasma = [subject_dir / f"{subject_id}plasma.txt", ]
        alternate_plasma = [subject_dir / "plasma.txt", ]
        possible_files = old_school_plasma + alternate_plasma
    elif contents.lower() in ["midtimes", "mid-times", "mid_times", ]:
        picnic_midtimes = list(subject_dir.glob(
            f"ses-{tracer.lower()}*_tacs/out_file/wmparc_reoriented_tacs.tsv"
        ))
        old_school_times = [subject_dir / f"{subject_id}.raw.midtime.txt", ]
        alternate_times = [subject_dir / "midtimes.txt", ]
        possible_files = old_school_times + alternate_times + picnic_midtimes
    else:
        logger.error("I do not understand '{contents}' content.")
        logger.error("I can only load 'tacs', 'midtimes', or 'plasma'.")
        return None, None

    f, data = None, None
    for f in possible_files:
        if f.exists():
            if data is None:
                # Load the first file found
                logger.info(f"Reading {contents.lower()} file '{f}'")
                if contents.lower() in ["midtimes", "mid-times", "mid_times"]:
                    if f.name.endswith(".txt"):
                        # For one-column naked text files, no header and 1 col
                        header = None
                        names = ['t', ]
                data = pd.read_csv(
                    f, header=header, index_col=index_col,
                    sep=sep, names=names,
                )
                if contents.lower() in ["midtimes", "mid-times", "mid_times"]:
                    if f.name.endswith(".tsv"):
                        # For PICNIC runs, we get mid-times from a TACs file
                        data = pd.DataFrame(data.iloc[:, 0].rename('t'))
                logger.debug(f"  {contents} data shaped {data.shape}")
            else:
                # Ignore files found after we loaded the first
                logger.warning(f"Ignoring extra {contents} file '{f}'")

    if data is None:
        return data, possible_files
    else:
        return data, f


def get_tacs(args, logger=None):
    """ Find a tacs file and read its data

    :param args: command line arguments
    :param logger: logger to report out findings
    :return pandas.DataFrame: TACs
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    tac_df, tac_file = get_tsv_data(
        args.input_path, args.subject, "tacs", args.tracer, logger
    )
    if tac_df is None:
        return None, None, tac_file
    good_regions = [r for r in args.regions if r in tac_df.columns]
    if args.ignore_frames is not None and len(args.ignore_frames) > 0:
        final_tac_df = tac_df.drop(
            np.asarray(args.ignore_frames) - 1, axis=0
        )
        return tac_df, final_tac_df[good_regions], tac_file
    else:
        return tac_df, tac_df[good_regions], tac_file


def get_plasma(input_path, subject_id, tracer, logger=None):
    """ Find a plasma file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str tracer: what tracer was injected
    :param logger: where to send messages
    :return pandas.DataFrame: A Centroid containing plasma activity
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    plasma_data, plasma_file = get_tsv_data(
        input_path, subject_id, "plasma", tracer, logger
    )
    if plasma_data is None:
        return None, plasma_file
    if 'PlasRawY' in plasma_data.columns and 'PlasRawT' in plasma_data.columns:
        return TimeActivityCurve(
            activity=plasma_data['PlasRawY'].values.astype(float),
            timepoints=plasma_data['PlasRawT'].values.astype(float),
            source="plasma",
            name="plasma",
        ), plasma_file
    else:
        # This should not happen
        return None, plasma_file


def get_mid_times(
        input_path, subject_id, frames_to_ignore, tracer, logger=None
):
    """ Find a mid-times file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: ignored frames to cut from the list
    :param str tracer: what tracer was injected
    :param logger: where to send messages
    :return tuple: array of raw mid-times, mid-times w/o ignored frames
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    mid_times, mt_file = get_tsv_data(
        input_path, subject_id, "midtimes", tracer, logger
    )
    if (len(frames_to_ignore) > 0) and mid_times is not None:
        logger.warning(
            f"  {len(frames_to_ignore)} frames being removed."
        )
        ignored_mid_times = mid_times[
            mid_times.index.isin([f - 1 for f in frames_to_ignore])
        ]
        # Replace mid_times AFTER the ignored time point has been stored.
        final_mid_times = mid_times[
            ~mid_times.index.isin([f - 1 for f in frames_to_ignore])
        ]
    else:
        ignored_mid_times = pd.DataFrame(data=[], columns=["t", ], dtype=float)
        final_mid_times = mid_times

    if final_mid_times is not None and 't' in final_mid_times:
        return (
            mid_times['t'].values,
            final_mid_times['t'].values,
            ignored_mid_times['t'].values,
            mt_file
        )
    else:
        return None, None, None, mt_file


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
                nifti_file = orig_dir / f"{subject_id}_orig_{i + 1:02d}.nii.gz"
                nib.save(nifti_img, str(nifti_file))
                volumes.append(Image(
                    path=nifti_file.parent,
                    filename=nifti_file.name,
                    prefix="orig",
                    frame=i + 1,
                    nifti=nifti_img,
                ))
    return volumes


def get_4D_data(
        img_file, output_path, subject_id, frames_to_ignore,
        section, cached=False, logger=None
):
    """ Find images, a volume for each mid-time

    :param img_file: path to a 4D image file
    :param output_path: path to rewrite subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: list of frame numbers to avoid
    :param section: report section for adding lines to the report
    :param bool cached: True if from a cached image missing ignored volumes
    :param logger: where to send messages

    :return dict: key-value dict with image data
    """

    if logger is None:
        logger = logging.getLogger("STARE")

    # There's one 4D image to load and break up.
    logger.info(f"Reading 4d image '{img_file}'")
    combined_image = nib.load(img_file)
    original_shape = combined_image.shape
    logger.debug(f"  image contains {original_shape[3]} volumes.")
    if cached:
        # We must assume the volume has skipped any 'frames-to-ignore' already
        extra_sentence = "Assumed volumes [{}] not present in cached 4D".format(
            ", ".join([str(_) for _ in frames_to_ignore])
        )
    else:
        extra_sentence = "No volumes were ignored."
        if len(frames_to_ignore) > 0:
            chunks = []
            begin = 0
            for skipped in frames_to_ignore:
                end = skipped - 1
                chunks.append(combined_image.slicer[:, :, :, begin:end])
                begin = skipped
            chunks.append(combined_image.slicer[:, :, :, begin:])
            combined_image = nib.concat_images(chunks, axis=3)
            logger.debug(f"  image now contains {combined_image.shape[3]} volumes.")
            extra_sentence = (f"After removing {len(frames_to_ignore)} volumes, "
                              f"it contains {combined_image.shape[3]}.")

    # Split the 4d data out into separate volumes.
    volumes = explode_4d_into_volumes(
        combined_image, output_path / "orig",
        name_template=subject_id + "_orig_{:02d}.nii.gz",
        ignored_volumes=frames_to_ignore,
        cached=cached,
        logger=logger
    )

    section.add_line(f"Loaded PET data from <code>'{img_file}'</code>. "
                     f"It contained {original_shape[3]} volumes, "
                     f"each shaped {combined_image.shape[0:3]}. "
                     + extra_sentence)

    return combined_image, volumes


def gather_data(results):
    """ Manage the gathering of all input data on disk """

    logger = results.logger
    rpt_sect = results.report.begin_section("Gather Data")
    issued_command = " ".join(sys.argv)

    logger.debug(f"{results.name} is running with these arguments.")
    for k, v in vars(results.args).items():
        spaces = " " * (23 - len(k))
        logger.debug(f"  '{k}'{spaces}: {v}")
    logger.info(f"The command issued: '{issued_command}'")

    rpt_sect.add_line("\n".join([
        "The stare_pet command executed:<br />",
        "<pre>",
        issued_command.replace("--", "\\\n--"),
        "</pre>",
    ]))

    # Assume everything's good until we encounter a problem.
    ok_to_run = True
    args = results.args

    # Read PET TAC data
    full_tacs, tacs, tacs_file = get_tacs(results.args, logger)
    if tacs is None:
        logger.error("Failed to load TACs")
        for failed_file in tacs_file:  # list of files if none are found
            logger.error(f"  tried '{str(failed_file)}'")
        ok_to_run = False
    else:
        if len(tacs.columns) < len(args.regions):
            dropped_regions = [r for r in args.regions if r not in tacs.columns]
            logger.warning("Specified regions were NOT found in the TACs:")
            for region in dropped_regions:
                logger.warning(f"   {region}")
        logger.info(f"Running with {len(tacs.columns)} regions:"
                    f"    [{', '.join(tacs.columns)}]")
        rpt_sect.add_line(f"Loaded TACs from <code>'{tacs_file}'</code>. "
                          f"Using {len(tacs.columns)} regions.")

    results.original_tacs = full_tacs
    results.tacs = tacs
    results.source_tacs_path = tacs_file

    # Find and load mid_times
    all_mid_times, mid_times, ignored_mid_times, mid_times_file = get_mid_times(
        args.input_path, args.subject, args.ignore_frames,
        args.tracer, logger=logger
    )
    if mid_times is None:
        logger.error("Failed to load midtimes")
        for failed_file in mid_times_file:  # list of files if none are found
            logger.error(f"  tried '{str(failed_file)}'")
        ok_to_run = False
    else:
        rpt_sect.add_line(
            f"Loaded mid_times from <code>'{mid_times_file}'</code>. "
            f"Running with {len(mid_times)} time points."
        )

    results.mid_times = mid_times
    results.ignored_mid_times = ignored_mid_times
    results.original_mid_times = all_mid_times
    results.source_mid_times_path = mid_times_file

    # Get plasma data if it's available, but this is not required.
    plasma_tac, plasma_file = get_plasma(
        args.input_path, args.subject, args.tracer, logger=logger
    )
    if plasma_tac is None:
        logger.warning("Could not find any plasma TACs.")
        for failed_file in plasma_file:  # list of files if none are found
            logger.error(f"  tried '{str(failed_file)}'")
        logger.warning("STARE doesn't need plasma, but plots it if available.")
    else:
        rpt_sect.add_line(f"Found plasma data in <code>'{plasma_file}'</code>.")

    if args.debug:
        pickle.dump(
            plasma_tac,
            open(args.debug_path / "tac_plasma.pkl", "wb")
        )

    results.plasma_tac = plasma_tac
    results.source_plasma_tac_file = plasma_file

    # Load PET images
    combined_image, volumes = None, None

    # The first preference is if there is already a cached 4D image.
    cached_img_file = args.output_path / f"{args.subject}_orig.nii.gz"
    if combined_image is None and cached_img_file.exists():
        # If the image was cached, it was cached without the ignored frames
        combined_image, volumes = get_4D_data(
            cached_img_file, args.output_path, args.subject,
            args.ignore_frames, rpt_sect, cached=True, logger=logger
        )

    # The next preference is a 4D image from the PICNIC pipeline.
    picnic_img_file = Path("/NOT_A_FILE.ext")
    for img in (args.input_path / args.subject).glob(
        "ses-{t}*_moco/out_file/ses-{t}*.nii.gz".format(t=args.tracer.lower())
    ):
        # There should only be one file (or none)
        picnic_img_file = img
    if combined_image is None and picnic_img_file.exists():
        combined_image, volumes = get_4D_data(
            picnic_img_file, args.output_path, args.subject,
            args.ignore_frames, rpt_sect, logger=logger
        )

    # Older FDG data are saved as one Analyze volume per time point.
    # Newer data are saved as a single 4D nifti image.
    # Future data will be BIDS-compliant.
    # We need to support all of these, and also allow for PVC correction
    # of individual volumes later.
    moco_path = Path(args.input_path) / args.subject / "moco"
    moco_images = list(moco_path.glob("*.hdr"))
    moco_images.extend(list(moco_path.glob("*.nii")))
    moco_images.extend(list(moco_path.glob("*.nii.gz")))
    if combined_image is None and len(moco_images) > 1:
        # We can load a bunch of volumes.
        volumes = get_individual_volumes(
            args.input_path, args.output_path, args.subject,
            args.ignore_frames, logger=logger
        )
        # Collect all the 3d image data into a single 4d structure.
        combined_image = combine_volumes_into_4d(
            volumes, args.output_path / f"{args.subject}_orig.nii.gz",
            logger=logger
        )
        rpt_sect.add_line(f"Loaded PET data from {len(moco_images)} moco files."
                          f" They contained {combined_image.shape[3]} volumes, "
                          f"each shaped {combined_image.shape[0:3]}.")
    elif combined_image is None and len(moco_images) == 1:
        combined_image, volumes = get_4D_data(
            picnic_img_file, args.output_path, args.subject,
            args.ignore_frames, rpt_sect, logger=logger
        )

    if volumes is None:
        logger.error("Failed to load PET image data")
        ok_to_run = False
    if not ok_to_run:
        logger.error("Unable to find sufficient data to run STARE.\n"
                     "See previous errors above to determine what's missing.")
        sys.exit(1)  # No point continuing on

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = combined_image.slicer[:, :, args.axial_slices_to_clip:, :]
    nib.save(cropped_image,
             args.output_path / f"{args.subject}_orig_cropped.nii.gz")
    logger.debug(f"WROTE {args.subject}_orig_cropped.nii.gz "
                 f"({cropped_image.shape}) to {str(args.output_path)}")
    rpt_sect.add_line(f"Cropped {args.axial_slices_to_clip} slices from "
                      "the inferior of each PET volume taking them to "
                      f"{cropped_image.shape}.")
    # cropped_volumes = [cropped_image.slicer[:, :, :, i]
    #                    for i in range(cropped_image.shape[3])]

    # PET data should be in units of 'mCi'
    mci_image = image_in_millicuries(cropped_image, args.pet_units)

    # Store the relevant data to results object.
    results.input_4D = combined_image
    results.cropped_4D = mci_image
    results.volume_images = volumes

    rpt_sect.end()
    return results
