import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import nibabel as nib

# Deleted get_plasma and other mentions of plasma throughout
from .util import get_tacs, get_images, get_mid_times, \
    combine_volumes_into_4d, image_in_millicuries
from .clustering import two_step_clustering, best_of, \
    save_centroid_masks
from .centroid_heuristics import find_vascular_centroids
from .plotting import plot_detailed_tacs
from .plotting import tacs_to_plottable_dataframe, plot_vascular_tacs


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
        "--cache-path", type=Path,
        help="Fast local storage for caching interim data",
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
    parser.add_argument(
        "--force", action="store_true",
        help="even if data are cached, recalculate and overwrite all output",
    )

    return parser


def setup_logger(app_name, args):
    """ Create a logger and configure it. """

    # Set up a logger to handle output, and attach two handlers
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    # TODO: Add an html logger to write a report.
    # TODO: See if a logger can intercept sklearn.KMeans verbose output

    # Create a handler to write out to the terminal
    # This handler adapts to the verbosity in the command line.
    terminal_handler = logging.StreamHandler(sys.stdout)
    terminal_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s : %(message)s", datefmt="%I:%M:%S",
    ))
    if args.verbose > 1:
        terminal_handler.setLevel(logging.DEBUG)
    elif args.verbose > 0:
        terminal_handler.setLevel(logging.INFO)
    else:
        terminal_handler.setLevel(logging.WARNING)
    logger.addHandler(terminal_handler)

    # Create a handler to write detailed information to a log file.
    # This handler always captures all info, debug and higher
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M")
    file_handler = logging.FileHandler(
        Path(args.output_path) / f"stare_pet_{timestamp}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s : %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S %p",
    ))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def validate_arguments(args):
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
    setattr(args, "output_path", Path(args.output_path))
    if not args.output_path.name == args.subject:
        args.output_path = Path(args.output_path) / args.subject
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating '{str(args.output_path)}', which did not exist.")
    if not args.output_path.exists():
        msg = f"The output_path '{str(args.output_path)}' does not exist and I cannot create it."
        errors.append(msg)
    else:
        tmp_file = args.output_path / "test.tmp"
        tmp_file.touch()
        if not tmp_file.exists():
            msg = f"The output_path '{str(args.output_path)}' is not writable."
            errors.append(msg)
        os.remove(tmp_file)
    setattr(args, "fig_path", Path(args.output_path) / "figures")
    args.fig_path.mkdir(parents=True, exist_ok=True)
    setattr(args, "debug_path", Path(args.output_path) / "debug")
    args.debug_path.mkdir(parents=True, exist_ok=True)
    setattr(args, "cluster_path", Path(args.output_path) / "clusters")
    args.cluster_path.mkdir(parents=True, exist_ok=True)
    if not hasattr(args, "cache_path"):
        setattr(args, "cache_path", Path(args.output_path) / "cache")
    args.cache_path.mkdir(parents=True, exist_ok=True)

    # Ensure we have regions to work with.
    print("regions:", args.regions)
    if args.regions is None:
        # If not specified, use a default bucket of regions.
        setattr(args, "regions", ['cerfullcs_c', 'cin', 'hip', 'par', 'pfc', 'pip', ])
        # msg = f"No regions are specified; there's nothing to be done."
        # errors.append(msg)
    else:
        print("regions are good, no need to overwrite.")

    # Ignored frames should be indexed by integer
    if args.ignore_frames is None:
        args.ignore_frames = []
    else:
        args.ignore_frames = [int(f) for f in args.ignore_frames]

    # Set up a logger with handlers appropriate to the arguments provided.
    logger = setup_logger("STARE", args)

    # Log all arguments
    logger.debug(f"Stare is running with these arguments.")
    for k, v in vars(args).items():
        spaces = " " * (23 - len(k))
        logger.debug(f"  '{k}'{spaces}: {v}")
    logger.info(f"The command issued: '{' '.join(sys.argv)}'")

    # Report the problems and quit if we have fatal errors.
    if len(errors) > 0:
        for error in errors:
            logger.error(error)
        return False

    # Good to continue on
    return True


def clust_er176(args):
    """ The clust_er176 function validates the execution context,
        then orchestrates execution of the different steps.

    :param args: The parsed argparse object

    :return: 0 if successful, error code if not
    :rtype: int
    """

    logger = logging.getLogger("STARE")

    # Validate out_path argument
    begin_timestamp = datetime.now()
    logger.info(f"Begin clust_er176 at {begin_timestamp}.")

    # Read PET data
    tacs = get_tacs(
        args.input_path, args.subject
    )
    if tacs is None:
        logger.error("Failed to load TACs")


    mid_times, ignored_mid_times = get_mid_times(
        args.input_path, args.subject, args.ignore_frames
    )
    if mid_times is None:
        logger.error("Failed to load midtimes")

    orig_images = get_images(
        args.input_path, args.output_path, args.subject, args.ignore_frames
    )
    if orig_images is None:
        logger.error("Failed to load PET image data")

    # Step 0. Format PET data

    # Collect all the 3d image data into a single 4d structure.
    combined_image = combine_volumes_into_4d(
        orig_images, args.output_path / "orig.nii.gz", logger=logger
    )
    combined_template = combined_image.slicer[:, :, :, 0]

    # Handle any requested axial clipping.
    # if axial_slices_to_clip == zero, this will not affect the image.
    # The header is updated along with the data.
    cropped_image = combined_image.slicer[:, :, args.axial_slices_to_clip:, :]
    nib.save(cropped_image, args.output_path / "orig_cropped.nii.gz")
    logger.debug(f"WROTE orig_cropped.nii.gz ({cropped_image.shape}) "
                 f"to {str(args.output_path)}")
    cropped_template = cropped_image.slicer[:, :, :, 0]

    # PET data should be in units of 'mCi'
    mci_image = image_in_millicuries(cropped_image, args.pet_units, logger=logger)

    # -------------------------------------------------------------------------
    # Step 1. Run two-step k-means clustering

    # TODO: Shiv: For other than vascular clusters, change the cluster_function
    # TODO: Shiv: and rename the figures below.

    centroids_step_1, centroids_step_2 = two_step_clustering(
        mci_image,
        step_one_ks=[30, ], # change to 30
        step_two_ks=[4, ],
        mid_times=mid_times,
        cache_path=args.cache_path,
        cluster_function=find_vascular_centroids,
        force=args.force,
        verbose=args.verbose
    )
    # Plot the TACs from the first k-means step
    fig = plot_vascular_tacs(centroids_step_1)
    fig.savefig(args.fig_path / "step_1_vascular_tacs.png")
    fig = plot_vascular_tacs(centroids_step_2)
    fig.savefig(args.fig_path / "step_2_vascular_tacs.png")

    if args.verbose > 1:
        # These data can be used to build custom plots or otherwise explore.
        tacs_to_plottable_dataframe(centroids_step_1).to_csv(
            args.debug_path / "step_1_centroids.csv"
        )
        logger.debug(f"WROTE step_1_centroids.csv to {str(args.debug_path)}")
        tacs_to_plottable_dataframe(centroids_step_2).to_csv(
            args.debug_path / "step_2_centroids.csv"
        )
        logger.debug(f"WROTE step_2_centroids.csv to {str(args.debug_path)}")

    best_centroid_step_1 = best_of(centroids_step_1)
    best_centroid_step_2 = best_of(centroids_step_2)
    for centroid_list in [centroids_step_1, centroids_step_2, ]:
        best_atlas = save_centroid_masks(
            centroid_list, FITS, args.output_path / "masks",
            cropped_template, combined_template,
            axial_slices_to_clip=args.axial_slices_to_clip,
            verbose=args.verbose
        )
    if best_atlas is None:
        logger.error("Could not determine best centroid, cannot continue PVC.")
        sys.exit(1)

    # Paint a picture of progress so far
    fig_top_tacs = plot_detailed_tacs(
        data=[
            best_centroid_step_1, best_centroid_step_2,
        ],
        title=f"Subject {args.subject} Best Cluster TACs",
        palette={
            best_centroid_step_1.name: "blue",
            best_centroid_step_2.name: "red"
        },
    )
    fig_top_tacs.savefig(args.fig_path / "best_cluster_tacs.png")

    # Output time and duration for those who care to benchmark
    finish_timestamp = datetime.now()
    logger.info(f"STARE is finished at {finish_timestamp}.")
    logger.info(f"{finish_timestamp - begin_timestamp} elapsed.")

    return 0
