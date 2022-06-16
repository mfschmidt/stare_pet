import os
import sys
import argparse

import re
from datetime import datetime
from pathlib import Path
from humanize import ordinal
import nibabel as nib
from starelib.util import *
from starelib.vascular_cluster import vascular_clustering


def get_argument_parser():
    """ Collect command line arguments """

    parser = argparse.ArgumentParser(
        description="Execute the STARE pipeline.",
    )
    parser.add_argument(
        "subject",
        help="The subject id",
    )
    parser.add_argument(
        "-i", "--input-path", type=Path, default=".",
        help="The path for input files",
    )
    parser.add_argument(
        "-o", "--output-path", type=Path, default=".",
        help="The path for output files",
    )
    parser.add_argument(
        "-a", "--axial-slices-to-clip", type=int, default=0,
        help="Axial slices to clip.",
    )
    parser.add_argument(
        "-u", "--pet-units", type=str, default='kBq',
        help="PET Units, default to 'kBq'.",
    )
    parser.add_argument(
        "--pvc-method", type=str, default='STC',
        help="PVC method"
             ", only 'single target correction' ('STC') is supported",
    )
    parser.add_argument(
        "--fwhm", type=float, default=5.9,
        help="Full width half maximum for ?",
    )
    parser.add_argument(
        "--tracer", type=str, default="FDG",
        help="The irreversible PET tracer used, only 'FDG' is supported",
    )
    parser.add_argument(
        "-c", "--vasc-corr-pct", type=int, default=5,
        help="The vascular correction percentage, as an integer from 0 to 100",
    )
    parser.add_argument(
        "--ignore-frames", type=int, nargs="+",
        help="Any frames listed with this argument will be ignored ",
    )
    parser.add_argument(
        "--regions", type=str, nargs="+",
        help="Brain region names to be quantified in STARE."
    )
    parser.add_argument(
        "-f", "--options-file", type=str,
        help="A file containing command-line arguments."
             "The arguments in the file will override defaults,"
             "but be overridden by explicit command-line arguments."
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="set from 0 to 2 times to trigger more verbose output",
    )

    return parser


def arguments_valid(args):
    """ Check arguments and establish context consistency before starting.

    :param argparse.parser.arguments args: Parsed arguments

    :return bool: True if everything is workable, False if there's a problem.
    """

    # Cache error messages, so we can report them all at once.
    errors = []

    # Ensure the input location exists, and contains the subject.
    if Path(args.input_path).exists():
        if not (Path(args.input_path) / args.subject).exists():
            errors.append(f"There is no subject '{args.subject}' at '{args.input_path}'.")
    else:
        errors.append(f"The input path, '{args.input_path}' does not exist.")

    # Ensure the output location exists, and is writable.
    if not Path(args.output_path).name == args.subject:
        args.output_path = Path(args.output_path) / args.subject
    if not Path(args.output_path).exists():
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating '{args.output_path}', which did not exist.")
    if not Path(args.output_path).exists():
        msg = f"The output_path '{args.output_path}' does not exist and I cannot create it."
        errors.append(msg)
    else:
        tmp_file = Path(args.output_path) / "test.tmp"
        tmp_file.touch()
        if not tmp_file.exists():
            msg = f"The output_path '{args.output_path}' is not writable."
            errors.append(msg)
        os.remove(tmp_file)

    # Ensure we have regions to work with.
    print("regions:", args.regions)
    if args.regions is None:
        # If not specified, use a default bucket of regions.
        setattr(args, "regions", ['cerfullcs_c', 'cin', 'hip', 'par', 'pfc', 'pip', ])
        # msg = f"No regions are specified; there's nothing to be done."
        # errors.append(msg)
    else:
        print("regions are good, no need to overwrite.")

    # Set up a logger to handle output, and attach two handlers
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # TODO: Add an html logger to write a report.
    # TODO: See if a logger can intercept sklearn.KMeans verbose output

    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s : %(message)s", datefmt="%I:%M:%S",
    ))
    logger.addHandler(terminal_handler)

    file_handler = logging.FileHandler(Path(args.output_path) / "stare_pet.log")
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p",
    ))
    logger.addHandler(file_handler)
    if args.verbose > 1:
        file_handler.setLevel(logging.DEBUG)
        terminal_handler.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        file_handler.setLevel(logging.DEBUG)
        terminal_handler.setLevel(logging.INFO)
    else:
        file_handler.setLevel(logging.INFO)
        terminal_handler.setLevel(logging.ERROR)

    # Ignored frames should be indexed by integer
    if args.ignore_frames is None:
        args.ignore_frames = []
    else:
        args.ignore_frames = [int(f) for f in args.ignore_frames]

    # If verbose is set, output all arguments
    logging.debug(f"Stare is running with these arguments.")
    for k, v in vars(args).items():
        spaces = " " * (23 - len(k))
        logging.debug(f"  '{k}'{spaces}: {v}")
    logging.info(f"The command issued: '{' '.join(sys.argv)}'")

    # Report the problems and quit if we have fatal errors.
    if len(errors) > 0:
        for error in errors:
            logging.error(error)
        return False

    # Good to continue on
    return True


def get_tacs(input_path, subject_id):
    """ Find a tacs file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :return pandas.DataFrame: TACs
    """

    tacs = None
    subject_dir = Path(input_path) / subject_id
    for f in ["{subject}.TACs", "tacs.txt", ]:
        actual_f = subject_dir / f.format(subject=subject_id)
        if actual_f.exists():
            if tacs is None:
                logging.info(f"Reading tacs file '{actual_f}'")
                tacs = pd.read_csv(
                    actual_f, header=0, index_col=None, sep='\t'
                )
                logging.debug(f"  tacs data shaped {tacs.shape}")
            else:
                logging.warning(f"Ignoring extra mid_times file '{actual_f}'")
    return tacs


def get_mid_times(input_path, subject_id, frames_to_ignore):
    """ Find a mid-times file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: ignored frames to cut from the list
    :return pandas.DataFrame: mid-times
    """

    mid_times = None
    subject_dir = Path(input_path) / subject_id
    for f in ["{subject}.raw.midtime.txt", "midtimes.txt", ]:
        actual_f = subject_dir / f.format(subject=subject_id)
        if actual_f.exists():
            if mid_times is None:
                logging.info(f"Reading mid_times file '{actual_f}'")
                mid_times = pd.read_csv(
                    actual_f, header=None, index_col=None, sep='\t',
                    names=['t', ],
                )
                logging.debug(f"  mid-times data shaped {mid_times.shape}")
                if len(frames_to_ignore) > 0:
                    logging.warning(
                        f"  {len(frames_to_ignore)} frames being removed."
                    )
                    mid_times = mid_times[~mid_times.index.isin([f - 1 for f in frames_to_ignore])]
            else:
                logging.warning(f"Ignoring extra mid_times file '{actual_f}'")
    return mid_times


def get_images(input_path, subject_id, frames_to_ignore):
    """ Find images, a volume for each mid-time

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :param frames_to_ignore: list of frame numbers to avoid

    :return dict: key-value dict with image data
    """

    images = []
    image_dir = Path(input_path) / subject_id / "moco"
    for pattern in ["{subject}.*.MCFI.hdr", "*.nii", "*.nii.gz", ]:
        actual_pattern = pattern.format(subject=subject_id)
        for i, img_file in enumerate(sorted(image_dir.glob(actual_pattern))):
            # Check for named frame numbers, just to warn about misunderstanding
            match = re.search(r"[._-](\d+)[._-]", img_file.name)
            if match and int(match.group(1)) != i + 1:
                logging.warning("Image numbering does not match sort order.")
                logging.warning(f"  '{img_file.name}' "
                                f"is the {ordinal(i + 1)} file, "
                                f"but #{match.group(1)}.")
                logging.warning(f"stare_pet uses sort ordering, #{i + 1}")

            # Store the image if it is not to be ignored.
            if i + 1 in frames_to_ignore:
                logging.warning(f"Frame {i + 1} exists, but is being ignored.")
            else:
                logging.info(f"Reading volume '{img_file}' as frame {i + 1:02d}")
                img = nib.load(img_file)
                logging.debug(f"  frame {i + 1} is shaped "
                              f"{'n/a' if img is None else img.shape}")
                images.append({
                    "path": img_file.parent,
                    "name": img_file.name,
                    "frame": i + 1,
                    "data": img,
                })
    return images


def stare(args):
    """ The stare function validates the execution context,
        then orchestrates the entire STARE pipeline.

    :param args: The parsed argparse object

    :return: 0 if successful, error code if not
    :rtype: int
    """

    # Validate out_path argument
    begin_timestamp = datetime.now()
    logging.info(f"Begin STARE.")

    # Read data
    tacs = get_tacs(args.input_path, args.subject)
    mid_times = get_mid_times(args.input_path, args.subject, args.ignore_frames)
    images = get_images(args.input_path, args.subject, args.ignore_frames)

    # Run vascular k-means clustering
    rslt1 = vascular_clustering(args.output_path, images,
                                pet_units=args.pet_units,
                                axial_slices_to_clip=args.axial_slices_to_clip,
                                mid_times=mid_times,
                                verbose=args.verbose)

    # Correct partial volumes from vascular clustering
    rslt2 = correct_partial_volumes(mid_times)

    # Correct TACs by extracting the mean signal from each cluster
    rslt3 = fit_vascular_mean_tac(tacs)
    # Then apply vascular correction
    rslt1 = tac_vascular_correction(rslt1)

    # Bootstrap signal in PVCed vasculature to generate input functions
    rslt2 = boot_anchor(rslt2)

    # Minimize the cost function
    rslt3 = minimize_cost_function(rslt3)

    # Since all functions are stubs, just keep python's
    # linters happy by using the rslt
    print(type(rslt1), type(rslt2), type(rslt3))

    # Validate out_path argument
    finish_timestamp = datetime.now()
    logging.info("STARE is finished.")
    logging.info(f"{finish_timestamp - begin_timestamp} elapsed.")

    return 0
