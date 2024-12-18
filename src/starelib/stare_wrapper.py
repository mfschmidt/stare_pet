import os
import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import warnings
from importlib.metadata import version


def get_argument_parser():
    """ Collect command line arguments """

    # If the user calls us with --version, all other assumptions fail.
    # They don't need a subject; their paths don't matter, etc.
    # So parse just that first, then move along to the "real" parser.
    version_arg_parser = argparse.ArgumentParser(
        prog="stare_pet",
        description="Execute the STARE pipeline.",
        add_help=False,
    )
    version_arg_parser.add_argument(
        "--version", action="store_true",
    )
    version_args, other_args = version_arg_parser.parse_known_args()
    if version_args.version:
        print(f"stare_pet v{version('stare_pet')}")
        sys.exit(0)
    else:
        # No worries; let the next parser take it.
        print("moving on, no --version arg.")
        pass

    # They didn't ask for --version, so we continue as usual.
    # We include --version in these args to generate accurate usage.
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
        "--tac-file", type=Path, default=None,
        help="Override searching through input-path with a specific file",
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
        help="Full width half maximum for partial volume correction",
    )
    parser.add_argument(
        "--tracer", type=str, default="FDG",
        help="The irreversible PET tracer used, only 'FDG' is supported",
    )
    parser.add_argument(
        "--override-step-1-cluster", type=Path, default=None,
        help="A binary mask may be used in place of k-means clustering.",
    )
    parser.add_argument(
        "--override-step-2-cluster", type=Path, default=None,
        help="A binary mask may be used in place of k-means clustering.",
    )
    parser.add_argument(
        "-c", "--vasc-corr-pct", type=int, default=0,
        help="The vascular correction percentage, as an integer from 0 to 100",
    )
    parser.add_argument(
        "--ignore-frames", type=int, nargs="+",
        help="Any frames listed with this argument will be ignored",
    )
    parser.add_argument(
        "--regions", type=str, nargs="+",
        help="Brain region names to be quantified in STARE."
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=500,
        help="How many bootstrapped curves shall be fit to feed the annealer?"
    )
    parser.add_argument(
        "--annealer-iterations", type=int, default=5000,
        help="How many iterations should the annealer be capped at?"
    )
    parser.add_argument(
        "--resample-for-clustering", type=str, default="",
        help="Down-sample the PET images for k-means clustering. "
             "'x2' down-samples by halving in each dimension. "
             "'2mm' resamples to a resolution of 2mm isotropic. "
             "'3mm' resamples to a resolution of 3mm isotropic. "
             "'4mm' resamples to a resolution of 4mm isotropic."
    )
    parser.add_argument(
        "--reduce-step-one-sparsity", type=int, default=0,
        help="The threshold for removing small blobs within the best "
             "cluster. A threshold of 10 will remove up to 10%% of the voxels, "
             "from the smallest blobs, leaving the largest, most contiguous "
             "blobs, constituting at least 90%% of the voxels."
    )
    parser.add_argument(
        "--latest-usable-volume", type=int, default=-1,
        help="Run STARE only on the earliest N volumes specified. "
             "This can be useful for time stability analyses."
    )
    parser.add_argument(
        "--decompose-components", action="store_true",
        help="Turn on to generate PCA and ICA component maps. "
             "Stare_pet doesn't use these yet, but they can be compared with "
             "k-means clusters."
    )
    parser.add_argument(
        "--save-all-cluster-masks", action="store_true",
        help="Turn on to save nifti masks of all clusters, not just best. "
             "These masks will be saved in the 'debug/masks/' directory."
    )
    parser.add_argument(
        "--save-all-failures", action="store_true",
        help="This feature is not yet available. In a future version, "
             "Turn on to save parameters of failed curve fits. "
             "This can be useful for debugging or investigating the range of "
             "parameters useful for fitting curves, but can consume tens of "
             "gigabytes of memory for hard-to-fit TACs. These failures will "
             "be written to a csv file in the 'debug/' directory."
    )
    parser.add_argument(
        "--ignore-spatial-info", action="store_true",
        help="Turn on to ensure the cluster selected by k-means, "
             "based only on temporal information, is used "
             "without considering spatial information to override it."
    )
    parser.add_argument(
        "--stop-after-clustering", action="store_true",
        help="Exit after clustering is complete"
    )
    parser.add_argument(
        "--stop-after-pvc", action="store_true",
        help="Exit after partial volume correction is complete"
    )
    parser.add_argument(
        "--output-all-fit-failures", action="store_true",
        help="Turn on to emit a note on each curve fit failure to the log"
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
        "--debug", action="store_true",
        help="Log extra data and pickle extra data to the debug directory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="even if data are cached, recalculate and overwrite all output",
    )
    parser.add_argument(
        "--num-cpus", default="",
        help="where parallel processing is supported, use this many processes",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="If specified, print the version and exit.",
    )

    return parser


def validate_arguments(args):
    """ Check arguments and establish context consistency before starting.

    :param argparse.parser.arguments args: Parsed arguments

    :return bool: True if everything is workable, False if there's a problem.
    """

    # Cache error messages, so we can report them all at once.
    errors = []

    # Turn off warnings from other libraries about deprecated functions and
    # the like. We only care if we are debugging.
    if not args.debug:
        warnings.filterwarnings("ignore")

    # Ensure the input location exists, and contains the subject.
    if Path(args.input_path).exists():
        if not (Path(args.input_path) / args.subject).exists():
            errors.append(f"There is no subject '{args.subject}' "
                          "at '{args.input_path}'.")
    else:
        errors.append(f"The input path, '{args.input_path}' does not exist.")
    if args.tac_file is not None:
        if not args.tac_file.exists():
            errors.append(f"An explicit TAC file, '{str(args.tac_file)}' was "
                          "specified, but it does not exist. If this file is "
                          "in the input-dir, stare will look for it by default."
                          " It does not need to be specified.")

    # Ensure the output location exists, and is writable.
    setattr(args, "output_path", Path(args.output_path))
    if not args.output_path.name == args.subject:
        args.output_path = Path(args.output_path) / args.subject
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Creating '{str(args.output_path)}', which did not exist.")
    if not args.output_path.exists():
        msg = (f"The output_path '{str(args.output_path)}' "
               "does not exist and I cannot create it.")
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
    setattr(args, "mask_path", Path(args.output_path) / "masks")
    args.mask_path.mkdir(parents=True, exist_ok=True)
    if (not hasattr(args, "cache_path")) or (args.cache_path is None):
        setattr(args, "cache_path", Path(args.output_path) / "cache")
    args.cache_path.mkdir(parents=True, exist_ok=True)

    # Ensure we have regions to work with.
    if args.regions is None:
        # If not specified, use a default bucket of regions.
        setattr(args, "regions",
                ['cerfullcs_c', 'cin', 'hip', 'par', 'pph', 'pip', ])
        # msg = f"No regions are specified; there's nothing to be done."
        # errors.append(msg)

    # Ignored frames should be indexed by integer
    if args.ignore_frames is None:
        args.ignore_frames = []
    else:
        args.ignore_frames = [int(f) for f in args.ignore_frames]

    # Interpret how many CPUs to use for multiprocessing.
    if args.num_cpus == "":
        setattr(args, "num_cpus", 1)
    elif args.num_cpus == "max":
        setattr(args, "num_cpus", int(cpu_count()))
    else:
        if int(args.num_cpus) > cpu_count():
            setattr(args, "num_cpus", int(cpu_count()))
        else:
            setattr(args, "num_cpus", int(args.num_cpus))

    # Anything indicating not to resample should make this an empty string.
    if args.resample_for_clustering in ["none", "orig", "", ]:
        setattr(args, "resample_for_clustering", "")

    # Report the problems and quit if we have fatal errors.
    if len(errors) > 0:
        for error in errors:
            print(error)
        return False

    # Good to continue on
    return True
