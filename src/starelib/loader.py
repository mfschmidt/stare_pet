import sys
import logging
import re
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from humanize import ordinal
import pickle
from csv import Sniffer

from .timeactivitycurve import TimeActivityCurve
from .util import StareVolume, image_in_millicuries,\
    combine_volumes_into_4d, explode_4d_into_volumes


def get_tsv_data(
        input_path, subject_id, contents, tracer,
        logger=None
):
    """ Find a tsv/txt file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str contents: what kind of data is in the file
    :param str tracer: what tracer was injected
    :param logger: where to send messages
    :return pandas.DataFrame: data
    """

    logger = logging.getLogger("STARE") if logger is None else logger

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
        old_school_tacs = [subject_dir / "raw" / f"{subject_id}.tacs.tsv",
                           subject_dir / f"{subject_id}.TACs", ]
        alternate_tacs = [subject_dir / "tacs.txt",
                          subject_dir / "BS_Stats" / "coreg" / subject_id, ]
        possible_files = picnic_tacs + old_school_tacs + alternate_tacs
    elif contents.lower() == "plasma":
        old_school_plasma = [subject_dir / "raw" / f"{subject_id}.plasma.tsv",
                             subject_dir / f"{subject_id}plasma.txt", ]
        alternate_plasma = [subject_dir / "plasma.txt", ]
        possible_files = old_school_plasma + alternate_plasma
    elif contents.lower() in ["midtimes", "mid-times", "mid_times", ]:
        picnic_midtimes = list(subject_dir.glob(
            f"ses-{tracer.lower()}*_tacs/out_file/wmparc_reoriented_tacs.tsv"
        ))
        old_school_times = [
            subject_dir / "raw" / f"{subject_id}.raw.midtime.txt",
            subject_dir / f"{subject_id}.raw.midtime.txt",
        ]
        alternate_times = [
            subject_dir / "midtimes.txt",
            subject_dir / "raw" / f"midtimes.txt",
        ]
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


def get_tacs(results):
    """ Find a tacs file and read its data

    :param results: an object containing all the global data for stare_pet
    :return results: an object containing all the global data for stare_pet
    """

    # If an explicit TAC file was specified, use it.
    if results.args.tac_file is not None:
        # There is no safe way to assume comma or tab or space delimited.
        # so before reading, we'll sniff the second line of the file and
        # use that as a delimiter. Still not bullet-proof, but a good bet.
        sniffer = Sniffer()
        delimiter = "\t" if results.args.tac_file.suffix == ".tsv" else ","
        with open(results.args.tac_file, "r") as f:
            for row in range(2):
                delimiter = sniffer.sniff(next(f).strip()).delimiter
        results.original_tacs = pd.read_csv(
            results.args.tac_file, header=0, index_col=None, sep=delimiter
        )
        results.source_tacs_path = results.args.tac_file

        if 'MidTime' in results.original_tacs.columns:
            results.original_mid_times = results.original_tacs[['MidTime', ]]
            results.original_mid_times.columns = ['t', ]

    # Else, find one in the input_path
    else:
        results.original_tacs, results.source_tacs_path = get_tsv_data(
            results.args.input_path, results.args.subject, "tacs",
            results.args.tracer, results.logger
        )

    good_regions = [r for r in results.args.regions
                    if r in results.original_tacs.columns]
    if (
            (results.args.ignore_frames is not None) and
            (len(results.args.ignore_frames) > 0)
    ):
        final_tac_df = results.original_tacs.drop(
            np.asarray(results.args.ignore_frames) - 1, axis=0
        )
        results.tacs = final_tac_df[good_regions]
    else:
        results.tacs = results.original_tacs[good_regions]

    return results


def get_plasma(input_path, subject_id, tracer, logger=None):
    """ Find a plasma file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param str tracer: what tracer was injected
    :param logger: where to send messages
    :return pandas.DataFrame: A Centroid containing plasma activity
    """

    logger = logging.getLogger("STARE") if logger is None else logger

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


def get_mid_times(results):
    """ Find a mid-times file and read its data

    :param results: an object containing all global data for stare_pet
    :return results: an object containing all global data for stare_pet
    """

    if results.original_mid_times is None:
        # We need to find the mid_times file
        local_mid_times, results.source_mid_times_path = get_tsv_data(
            results.args.input_path, results.args.subject, "midtimes",
            results.args.tracer, results.logger
        )
    else:
        # mid_times were already loaded while reading the TACs file.
        local_mid_times = results.original_mid_times
        results.source_mid_times_path = results.source_tacs_path

    if (len(results.args.ignore_frames) > 0) and local_mid_times is not None:
        plural = "s" if len(results.args.ignore_frames) > 1 else ""
        results.logger.warning(
            f"  {len(results.args.ignore_frames)} frame{plural} being removed."
        )
        ignored_mid_times = local_mid_times[
            local_mid_times.index.isin(
                [f - 1 for f in results.args.ignore_frames]
            )
        ]
        # Replace mid_times AFTER the ignored time point has been stored.
        final_mid_times = local_mid_times[
            ~local_mid_times.index.isin(
                [f - 1 for f in results.args.ignore_frames]
            )
        ]
    else:
        ignored_mid_times = pd.DataFrame(data=[], columns=["t", ], dtype=float)
        final_mid_times = local_mid_times

    if final_mid_times is not None and 't' in final_mid_times:
        results.original_mid_times = local_mid_times['t'].values
        results.mid_times = final_mid_times['t'].values
        results.ignored_mid_times = ignored_mid_times['t'].values

    return results


def get_individual_volumes(
        input_path, output_path, subject_id, frames_to_ignore,
        logger=None
):
    """ Find images, a volume for each mid-time

    :param input_path: path to find subjects
    :param output_path: path to rewrite subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: list of frame numbers to avoid
    :param logger: where to send messages

    :return dict: key-value dict with image data
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    volumes = []
    image_dir = Path(input_path) / subject_id / "moco"
    orig_dir = Path(output_path) / "orig"
    orig_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["{subject}.*.hdr", "*.nii", "*.nii.gz", ]:
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
                logger.warning(f"Frame {i + 1} exists, and will be ignored.")
            else:
                logger.info(f"Reading volume '{img_file}' as frame {i + 1:02d}")
            img = nib.load(img_file)
            logger.debug(f"  frame {i + 1} is shaped "
                         f"{'n/a' if img is None else img.shape}")
            # No matter the original image format, we will save our own
            # copy of each image as a Nifti1 nii.gz for consistency
            # throughout the pipeline.
            if len(img.get_fdata().shape) > 3:
                nifti_img = nib.Nifti1Image(
                    img.get_fdata()[:, :, :, -1], img.affine
                )
            else:
                nifti_img = nib.Nifti1Image(
                    img.get_fdata(), img.affine
                )
            nifti_img.header.set_xyzt_units("mm", "sec")
            nifti_file = orig_dir / f"{subject_id}_orig_{i + 1:02d}.nii.gz"
            nib.save(nifti_img, str(nifti_file))
            volumes.append(StareVolume(
                nifti=nifti_img,
                path=nifti_file.parent,
                filename=nifti_file.name,
                prefix="orig",
                frame=i + 1,
                usable=((i + 1) not in frames_to_ignore),
            ))

    # Return all the volumes, including any skipped one(s)
    return volumes


def get_4D_data(
        img_file, output_path, subject_id, section,
        ignored_volumes=None, logger=None
):
    """ Find images, a volume for each mid-time

    :param img_file: path to a 4D image file
    :param output_path: path to rewrite subjects
    :param subject_id: name of subject folder
    :param section: report section for adding lines to the report
    :param list ignored_volumes: a list of volumes to pass over and not save
    :param logger: where to send messages

    :return dict: key-value dict with image data
    """

    logger = logging.getLogger("STARE") if logger is None else logger

    # There's one 4D image to load and break up.
    logger.info(f"Reading 4d image '{img_file}'")
    combined_image = nib.Nifti1Image.from_filename(img_file)
    original_shape = combined_image.shape
    logger.debug(f"  image contains {original_shape[3]} volumes.")

    # Split the 4d data out into separate volumes.
    volumes = explode_4d_into_volumes(
        combined_image, output_path / "orig",
        name_template=subject_id + "_orig_{:02d}.nii.gz",
        ignored_volumes=[] if ignored_volumes is None else ignored_volumes,
        logger=logger
    )

    section.add_line(f"Loaded PET data from <code>'{img_file}'</code>. "
                     f"It contained {original_shape[3]} volumes, "
                     f"each shaped {combined_image.shape[0:3]}. ")

    return combined_image, volumes


def gather_data(results):
    """ Manage the gathering of all input data on disk """

    logger = results.logger
    rpt_sect = results.report.begin_section("Gather Data")
    issued_command = " ".join(sys.argv)
    # Keep track of arguments we set to defaults because they weren't specified
    implemented_defaults = []
    calculated_paths = []
    for arg in dir(results.args):
        if (
                not arg.startswith("_") and
                arg != "subject" and
                ("--" + arg.replace("_", "-")) not in sys.argv
        ):
            if "path" in arg:
                calculated_paths.append(
                    f"{arg} writing to '{getattr(results.args, arg)}'"
                )
            else:
                implemented_defaults.append(
                    f"{arg} set to '{getattr(results.args, arg)}'"
                )
    logger.debug(f"{results.name} {results.report.app_version} "
                 f"is running with these arguments.")
    for k, v in vars(results.args).items():
        spaces = " " * (23 - len(k))
        logger.debug(f"  '{k}'{spaces}: {v}")
    logger.info(f"The command issued: '{issued_command}'")
    for default_line in implemented_defaults:
        logger.info(default_line)
    for default_line in calculated_paths:
        logger.info(default_line)
    rpt_sect.add_line("\n".join([
        "The stare_pet command executed:<br />",
        "<pre>",
        issued_command.replace("--", "\\\n--"),
        "</pre>",
    ]))
    rpt_sect.add_line("\n".join([
        "Unspecified variables were set to defaults:<br />",
        "<pre>",
        "\n".join(implemented_defaults),
        "</pre>",
    ]))
    rpt_sect.add_line("\n".join([
        "Default locations for paths were used:<br />",
        "<pre>",
        "\n".join(calculated_paths),
        "</pre>",
    ]))

    # Assume everything's good until we encounter a problem.
    ok_to_run = True
    args = results.args

    # Read PET TAC data
    results = get_tacs(results)
    # full_tacs, tacs, tacs_file = get_tacs(results.args, logger)
    if results.tacs is None:
        logger.error("Failed to load TACs")
        for failed_file in results.source_tacs_path:
            # list of attempted files if none were found
            logger.error(f"  tried '{str(failed_file)}'")
        ok_to_run = False
    else:
        if len(results.tacs.columns) < len(args.regions):
            dropped_regions = [r for r in args.regions
                               if r not in results.tacs.columns]
            logger.warning("Specified regions were NOT found in the TACs:")
            for region in dropped_regions:
                logger.warning(f"   {region}")
        logger.info(f"Running with {len(results.tacs.columns)} regions:"
                    f"    [{', '.join(results.tacs.columns)}]")
        rpt_sect.add_line("Loaded TACs from " 
                          f"<code>'{str(results.source_tacs_path)}'</code>. "
                          f"Using {len(results.tacs.columns)} regions.")

    # Find and load mid_times
    results = get_mid_times(results)
    if results.mid_times is None:
        logger.error("Failed to load midtimes")
        for failed_file in results.source_mid_times_path:
            # list of attempted files if none are found
            logger.error(f"  tried '{str(failed_file)}'")
        ok_to_run = False
    else:
        rpt_sect.add_line(
            "Loaded mid_times from "
            f"<code>'{results.source_mid_times_path}'</code>. "
            f"Running with {len(results.mid_times)} time points."
        )

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
        with open(args.debug_path / "tac_plasma.pkl", "wb") as f:
            pickle.dump(plasma_tac, f)

    results.plasma_tac = plasma_tac
    results.source_plasma_tac_file = plasma_file

    # Load PET images
    combined_image, volumes = None, None

    # The first preference is if there is already a cached 4D image.
    cached_img_file = args.output_path / f"{args.subject}_orig_4d.nii.gz"
    if combined_image is None and cached_img_file.exists():
        # If the image was cached, it was cached without the ignored frames
        combined_image, volumes = get_4D_data(
            cached_img_file, args.output_path, args.subject, rpt_sect,
            ignored_volumes=args.ignore_frames, logger=logger
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
            picnic_img_file, args.output_path, args.subject, rpt_sect,
            ignored_volumes=args.ignore_frames, logger=logger
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
        alerts = []
        combined_image = combine_volumes_into_4d(
            volumes, args.output_path / f"{args.subject}_orig_4d.nii.gz",
            alerts=alerts, logger=logger
        )
        rpt_sect.add_line(f"Loaded PET data from {len(moco_images)} moco files."
                          f" They contained {combined_image.shape[3]} volumes, "
                          f"each shaped {combined_image.shape[0:3]}.")
        for alert in alerts:
            rpt_sect.add_line(alert, css_class='warning', log=True)
    elif combined_image is None and len(moco_images) == 1:
        combined_image, volumes = get_4D_data(
            picnic_img_file, args.output_path, args.subject, rpt_sect,
            ignored_volumes=args.ignore_frames, logger=logger
        )

    if volumes is None:
        logger.error("Failed to load PET image data")
        ok_to_run = False
    if not ok_to_run:
        logger.error("Unable to find sufficient data to run STARE.\n"
                     "See previous errors above to determine what's missing.")
        sys.exit(1)  # No point continuing on

    # Preserve original image before removing slices and cropping.
    mean_image = nib.Nifti1Image(
        np.mean(combined_image.get_fdata(), axis=3),
        affine=combined_image.affine,
    )
    mean_image.header.set_xyzt_units("mm", "sec")
    nib.save(mean_image,
             args.output_path / f"{args.subject}_orig_mean.nii.gz")

    # Handle ignored frames, keeping both an original and a modified
    if len(args.ignore_frames) > 0:
        chunks = []
        begin = 0
        for skipped in args.ignore_frames:
            end = skipped - 1
            chunks.append(combined_image.slicer[:, :, :, begin:end])
            begin = skipped
        chunks.append(combined_image.slicer[:, :, :, begin:])
        cropped_image = nib.concat_images(chunks, axis=3)
        logger.debug(f"  image now contains {cropped_image.shape[3]} volumes.")
    else:
        cropped_image = combined_image

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = cropped_image.slicer[:, :, args.axial_slices_to_clip:, :]
    nib.save(cropped_image,
             args.output_path / f"{args.subject}_cropped_4d.nii.gz")
    logger.debug(f"WROTE {args.subject}_cropped_4d.nii.gz "
                 f"({cropped_image.shape}) to {str(args.output_path)}")
    rpt_sect.add_line(f"Cropped {args.axial_slices_to_clip} slices from "
                      "the inferior of each PET volume taking them to "
                      f"{cropped_image.shape}.")
    # cropped_volumes = [cropped_image.slicer[:, :, :, i]
    #                    for i in range(cropped_image.shape[3])]

    # PET data should be in units of 'mCi'
    mci_image = image_in_millicuries(cropped_image, args.pet_units)
    mean_mci_image = nib.nifti1.Nifti1Image(
        np.mean(mci_image.get_fdata(), axis=3),
        affine=mci_image.affine,
    )
    mean_mci_image.header.set_xyzt_units("mm", "sec")
    nib.save(mean_mci_image,
             args.output_path / f"{args.subject}_cropped_mean.nii.gz")

    # Store the relevant data to results object.
    results.input_4D = combined_image
    results.cropped_4D = mci_image
    results.volume_images = volumes

    rpt_sect.end()

    # Check some assertions before wasting processing time
    num_good_vols = len([v for v in volumes if v.usable])
    if len(results.mid_times) != num_good_vols:
        error_string = (
            "Volumes and time points must match one-to-one, but we have kept "
            f"{num_good_vols}/{len(volumes)} volumes to match up with "
            f"{len(results.mid_times)}/{len(results.original_mid_times)} "
            "time points. This error is fatal."
        )
        rpt_sect = results.report.begin_section("Fatal Error")
        rpt_sect.add_line(error_string)
        rpt_sect.end()
        results.write_report()
        logger.error(error_string)
        sys.exit(1)
    else:
        results.write_report()
    return results
