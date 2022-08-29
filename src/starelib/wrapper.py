import os
import sys
import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from .util import get_tacs, get_images, get_mid_times, get_plasma
from .vascular_cluster import vascular_clustering
from .partial_volume import correct_partial_volumes
from .plotting import tacs_to_plottable_dataframe, plot_detailed_tacs

# temporary stubs
from .util import fit_vascular_mean_tac, tac_vascular_correction,\
                  boot_anchor, minimize_cost_function


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


def stare(args):
    """ The stare function validates the execution context,
        then orchestrates the entire STARE pipeline.

    :param args: The parsed argparse object

    :return: 0 if successful, error code if not
    :rtype: int
    """

    logger = logging.getLogger("STARE")

    # Validate out_path argument
    begin_timestamp = datetime.now()
    logger.info(f"Begin STARE at {begin_timestamp}.")

    # Read data
    tacs = get_tacs(
        args.input_path, args.subject
    )
    plasma_tac = get_plasma(
        args.input_path, args.subject
    )
    mid_times = get_mid_times(
        args.input_path, args.subject, args.ignore_frames
    )
    orig_images = get_images(
        args.input_path, args.output_path, args.subject, args.ignore_frames
    )

    # Run two-step vascular k-means clustering
    best_mask, best_centroid_1, best_centroid_2 = vascular_clustering(
        args.output_path, orig_images,
        mid_times=mid_times,
        pet_units=args.pet_units,
        axial_slices_to_clip=args.axial_slices_to_clip,
        force=args.force,
        verbose=args.verbose
    )

    # Correct partial volumes from vascular clustering
    pvc_mean_centroid = correct_partial_volumes(
        orig_images,
        args.fwhm,
        args.output_path,
        best_mask,
        mid_times=mid_times,
    )

    # Paint a picture of progress so far
    fig_path = args.output_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    top_tacs_data = tacs_to_plottable_dataframe(
        [best_centroid_1, best_centroid_2,
         pvc_mean_centroid, plasma_tac, ],
    )
    # Customize the 'run' value for plotting with a legend
    # three_tacs_data['run'] = three_tacs_data.apply(
    #     lambda row: {
    #         (best_centroid_1.source, best_centroid_1.k, best_centroid_1.label): "step 1",
    #         (best_centroid_2.source, best_centroid_2.k, best_centroid_2.label): "step 2",
    #         (pvc_mean_centroid.source, pvc_mean_centroid.k, pvc_mean_centroid.label): "pvc",
    #         (plasma_tac.source, plasma_tac.k, plasma_tac.label): "plasma",
    #     }.get((row['source'], row['k'], row['label']), "n/a"),
    #     axis=1
    # )
    pickle.dump(
        top_tacs_data,
        (args.output_path / "figures" / "three_centroid_dataframe.pkl").open("wb")
    )
    # Create the plot
    fig_top_tacs = plot_detailed_tacs(
        top_tacs_data,
        title=f"Subject {args.subject} Vascular TACs",
        palette={"step 1": "blue", "step 2": "green",
                 "pvc": "orange", "plasma": "red", },
    )
    fig_top_tacs.savefig(fig_path / "three_tacs.png")

    # Correct TACs by extracting the mean signal from each cluster
    rslt3 = fit_vascular_mean_tac(tacs)
    # Then apply vascular correction
    rslt1 = tac_vascular_correction(pvc_mean_centroid)

    # Bootstrap signal in PVCed vasculature to generate input functions
    rslt2 = boot_anchor(pvc_mean_centroid)

    # Minimize the cost function
    rslt3 = minimize_cost_function(rslt3)

    # Since all functions are stubs, just keep python's
    # linters happy by using the rslts
    logger.debug(f"Ignore: {type(rslt1)}, {type(rslt2)}, {type(rslt3)}")

    # Validate out_path argument
    finish_timestamp = datetime.now()
    logger.info(f"STARE is finished at {finish_timestamp}.")
    logger.info(f"{finish_timestamp - begin_timestamp} elapsed.")

    return 0
