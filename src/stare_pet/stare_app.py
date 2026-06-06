import os
import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count
import warnings
from importlib.metadata import version

from stare_pet.starelib.stare_results import StareResults
from stare_pet.starelib.loader import gather_data
from stare_pet.starelib.clustering import two_step_cluster
from stare_pet.starelib.partial_volume import correct_partial_volumes
from stare_pet.starelib.fit_mean_tac import fit_vascular_mean_tac
from stare_pet.starelib.vascular_correction import tac_vascular_correction
from stare_pet.starelib.boot_anchor import boot_anchor
from stare_pet.starelib.minimize_cost import minimize_parameter_cost


class StareApp:
    def __init__(self):
        self.args = None
        self.results = None
        self.logger = None

    def run(self):
        # To traverse blobs in k-means space, we need to recurse deeper
        # than the system default of 3000.
        sys.setrecursionlimit(10000)

        # Collect and validate arguments.
        self.args = self.get_argument_parser().parse_args()

        # Asking validate_arguments(parsed_args) to handle arguments allows it
        # to fill in defaults, check for consistency and validity, and
        # notify the caller of any problems. If anything is missing or
        # unexpected, it will return False, and we will quit rather than
        # attempting to run with bad input.
        if self.validate_arguments():
            # The self.results object is a big container for intermediate
            # and output data structures we want to keep around.
            # We can read to it and write from it, and it keeps a log of
            # everything, so building a report at the end is easier.
            self.results = StareResults(
                "STARE", f"STARE Results for {self.args.subject}", self.args
            )
            self.results = gather_data(self.results)
            self.results = two_step_cluster(self.results)
            if self.results.args.stop_after_clustering:
                return 0
            self.results = correct_partial_volumes(self.results)
            if self.results.args.stop_after_pvc:
                return 0
            self.results = fit_vascular_mean_tac(self.results)
            self.results = tac_vascular_correction(self.results)
            self.results = boot_anchor(self.results)
            self.results = minimize_parameter_cost(self.results)
            self.results.end()
            self.results.write_report()
            if self.args.debug and self.args.debug_path.exists():
                # If debug is on, save the complete results object.
                self.results.save()
            else:
                # If debug is off, we're finished; delete unneeded things
                self.results.logger.debug("Deleting cache & debug directories")
                for directory in [
                    self.results.args.cache_path, self.results.args.debug_path
                ]:
                    for cache_file in directory.iterdir():
                        cache_file.unlink(missing_ok=True)
                    directory.rmdir()
            return 0
        else:
            return 1

    @staticmethod
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
            pass

        # They didn't ask for --version, so we continue as usual.
        # We include --version in these args to generate accurate usage.
        parser = argparse.ArgumentParser(
            description="Execute the STARE pipeline.",
        )
        parser.add_argument(
            "subject",
            help="The subject id. Arguments 'ID' or 'sub-ID' are equivalent.",
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
            "--pet-file", type=Path, default=None,
            help="Override searching through input-path with a specific file",
        )
        parser.add_argument(
            "--tac-file", type=Path, default=None,
            help="Override searching through input-path with a specific file",
        )
        parser.add_argument(
            "--plasma-file", type=Path, default=None,
            help="Override searching through input-path with a specific file",
        )
        parser.add_argument(
            "-a", "--axial-slices-to-clip", type=int, default=0,
            help="Axial slices to clip.",
        )
        parser.add_argument(
            "-u", "--pet-units", type=str, default='',
            help="PET Units, defaults to 'kBq'.",
        )
        parser.add_argument(
            "-t", "--time-units", type=str, default='min',
            help="Time Units, from TACS and/or midtimes files, default to 'min'.",
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
            "--override-step-1-cluster", type=Path, default=None,
            help="A binary mask may be used in place of k-means clustering.",
        )
        parser.add_argument(
            "--override-step-2-cluster", type=Path, default=None,
            help="A binary mask may be used in place of k-means clustering.",
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
            "--consider-alternate-step-one-cluster",
            action="store_true", default=False,
            help="Set this to True to cause STARE to assess the step one k-means "
                 "cluster, compare it to alternative clusters, and recommend "
                 "an alternate selection if it finds a better option. "
                 "Note that 'better' is subject to many factors and may change "
                 "version-to-version. And this doesn't actually select the new "
                 "cluster, but only recommends it. To implement the new cluster, "
                 "use the `--override-step-one-cluster` option with the suggested "
                 "cluster from this run."
        )
        parser.add_argument(
            "--keep-confetti-patterns-step-1",
            action="store_true", default=False,
            help="By default, STARE will not consider k-means clusters that look "
                 "like 'confetti on the floor'. Noisy scans may produce these "
                 "and they can have high early peaks that prevent selection "
                 "of good vascular clusters. This option causes STARE to include "
                 "these clusters as 'likely_vascular' while scoring them."
        )
        parser.add_argument(
            "--drop-confetti-patterns-step-2",
            action="store_true", default=False,
            help="By default, STARE will assume the step 1 filter excluded "
                 "problematic clusters, so this is no longer needed at step 2. "
                 "This option causes STARE to apply the filter again, "
                 "excluding any of the four sub-clusters marked as noise."
        )
        parser.add_argument(
            "--latest-usable-volume", type=int, default=-1,
            help="Run STARE only on the earliest N volumes specified. "
                 "This can be useful for time stability analyses."
        )
        parser.add_argument(
            "--decompose-components",
            action="store_true", default=False,
            help="Turn on to generate PCA and ICA component maps. "
                 "Stare_pet doesn't use these yet, but they can be compared with "
                 "k-means clusters."
        )
        parser.add_argument(
            "--save-all-cluster-masks",
            action="store_true", default=False,
            help="Turn on to save nifti masks of all clusters, not just best. "
                 "These masks will be saved in the 'debug/masks/' directory."
        )
        parser.add_argument(
            "--save-all-failures",
            action="store_true", default=False,
            help="This feature is not yet available. In a future version, "
                 "Turn on to save parameters of failed curve fits. "
                 "This can be useful for debugging or investigating the range of "
                 "parameters useful for fitting curves, but can consume tens of "
                 "gigabytes of memory for hard-to-fit TACs. These failures will "
                 "be written to a csv file in the 'debug/' directory."
        )
        parser.add_argument(
            "--ignore-spatial-info",
            action="store_true", default=False,
            help="By default, spatial info is used to classify and remove "
                 "step-one k-means clusters made up of only noise in inferior "
                 "slices. Optionally, it can also be used for sparsity reduction "
                 "or to consider alternate clusters. "
                 "This flag can be used to turn off the calculation of spatial "
                 "information and speed up processing slightly. This will prevent "
                 "exclusion of vascular clusters with all noise in inferior slices. "
                 "This flag will be overridden by --reduce-step-one-sparsity > 0, "
                 "--consider-alternate-step-one-cluster, "
                 "or --drop-confetti-patterns-step-2, because they all require "
                 "spatial information."
        )
        parser.add_argument(
            "--stop-after-clustering",
            action="store_true", default=False,
            help="Exit after clustering is complete"
        )
        parser.add_argument(
            "--stop-after-pvc",
            action="store_true", default=False,
            help="Exit after partial volume correction is complete"
        )
        parser.add_argument(
            "--output-all-fit-failures",
            action="store_true", default=False,
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
            "--debug", action="store_true", default=False,
            help="Log extra data and pickle extra data to the debug directory.",
        )
        parser.add_argument(
            "--force", action="store_true", default=False,
            help="even if data are cached, recalculate and overwrite all output",
        )
        parser.add_argument(
            "--num-cpus", default="",
            help="where parallel processing is supported, use this many processes",
        )
        parser.add_argument(
            "--version", action="store_true", default=False,
            help="If specified, print the version and exit.",
        )

        return parser

    def validate_arguments(self):
        """ Check arguments and establish context consistency before starting.

            :return bool: True if everything is workable, False if there's a problem.
        """

        # Cache error messages, so we can report them all at once.
        errors = []

        # Turn off warnings from other libraries about deprecated functions and
        # the like. We only care if we are debugging.
        if not self.args.debug:
            warnings.filterwarnings("ignore")

        # Use just the ID value, not its BIDS key
        if self.args.subject.startswith("sub-"):
            self.args.subject = self.args.subject[4:]

        # Ensure the input location exists and contains the subject.
        if Path(self.args.input_path).exists():
            if (Path(self.args.input_path) / self.args.subject).is_dir():
                setattr(self.args, "subject_path",
                        Path(self.args.input_path) / self.args.subject)
            elif (Path(self.args.input_path) / f"sub-{self.args.subject}").is_dir():
                setattr(self.args, "subject_path",
                        Path(self.args.input_path) / f"sub-{self.args.subject}")
            else:
                errors.append(f"There is neither subject '{self.args.subject}' nor "
                              f"'sub-{self.args.subject}' at '{self.args.input_path}'.")
        else:
            errors.append(f"The input path, '{self.args.input_path}' does not exist.")

        if self.args.tac_file is not None:
            if not self.args.tac_file.exists():
                errors.append(f"An explicit TAC file, '{str(self.args.tac_file)}' was "
                              "specified, but it does not exist. If this file is "
                              "in the input-dir, stare will look for it by default."
                              " It does not need to be specified.")

        # Ensure the output location exists, and is writable.
        setattr(self.args, "output_path", Path(self.args.output_path) / self.args.subject)
        # if not args.output_path.name == f"sub-{args.subject}":
        #     args.output_path = Path(args.output_path) / f"sub-{args.subject}"
        if not self.args.output_path.exists():
            self.args.output_path.mkdir(parents=True, exist_ok=True)
            print(f"Creating '{str(self.args.output_path)}', which did not exist.")
        if not self.args.output_path.exists():
            msg = (f"The output_path '{str(self.args.output_path)}' "
                   "does not exist and I cannot create it.")
            errors.append(msg)
        else:
            tmp_file = self.args.output_path / "test.tmp"
            tmp_file.touch()
            if not tmp_file.exists():
                msg = f"The output_path '{str(self.args.output_path)}' is not writable."
                errors.append(msg)
            os.remove(tmp_file)
        setattr(self.args, "fig_path", Path(self.args.output_path) / "figures")
        self.args.fig_path.mkdir(parents=True, exist_ok=True)
        setattr(self.args, "debug_path", Path(self.args.output_path) / "debug")
        self.args.debug_path.mkdir(parents=True, exist_ok=True)
        setattr(self.args, "cluster_path", Path(self.args.output_path) / "clusters")
        self.args.cluster_path.mkdir(parents=True, exist_ok=True)
        if (not hasattr(self.args, "cache_path")) or (self.args.cache_path is None):
            setattr(self.args, "cache_path", Path(self.args.output_path) / "cache")
        self.args.cache_path.mkdir(parents=True, exist_ok=True)

        # Ensure we have regions to work with.
        if self.args.regions is None:
            # If not specified, use a default bucket of regions.
            # These are old BAT regions, and should probably be updated to
            # FreeSurfer or PICNIC
            # setattr(args, "regions",
            #         ['cerfullcs_c', 'cin', 'hip', 'par', 'pph', 'pip', ])
            msg = f"No regions are specified; there's nothing to be done."
            errors.append(msg)

        # Ignored frames should be indexed by integer
        if self.args.ignore_frames is None:
            self.args.ignore_frames = []
        else:
            self.args.ignore_frames = [int(f) for f in self.args.ignore_frames]

        # Interpret how many CPUs to use for multiprocessing.
        if self.args.num_cpus == "":
            setattr(self.args, "num_cpus", 1)
        elif self.args.num_cpus == "max":
            setattr(self.args, "num_cpus", int(cpu_count()))
        else:
            if int(self.args.num_cpus) > cpu_count():
                setattr(self.args, "num_cpus", int(cpu_count()))
            else:
                setattr(self.args, "num_cpus", int(self.args.num_cpus))

        # Anything indicating not to resample should make this an empty string.
        if self.args.resample_for_clustering in ["none", "orig", "", ]:
            setattr(self.args, "resample_for_clustering", "")

        # Report the problems and quit if we have fatal errors.
        if len(errors) > 0:
            for error in errors:
                print(error)
            return False

        # Good to continue on
        return True


def main():
    """ Script entry point """
    sys.exit(StareApp().run())


if __name__ == "__main__":
    main()
