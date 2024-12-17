import subprocess
from datetime import datetime
import nibabel as nib
import numpy as np
import pickle

from .util import StareVolume, combine_volumes_into_4d, flatten_4d_to_2d
from .timeactivitycurve import TimeActivityCurve
from .plotting import tacs_to_plottable_dataframe, plot_detailed_tacs


def correct_partial_volumes(results):
    """ Correct partial volumes

        :param Results results: A results object for reading and writing data
        :return: results, with more data
    """

    logger = results.logger
    rpt_sect = results.report.begin_section("Partial volume correction")

    pre_pvc_timestamp = datetime.now()
    logger.info(f"Started PVC at {pre_pvc_timestamp}")

    # Create a path for our partial-volume data
    fig_path = results.args.output_path / "pvc"
    fig_path.mkdir(parents=True, exist_ok=True)

    # Perform PVC on each of the original volumes provided
    pvc_volumes = []
    pvc_exe = "/usr/local/bin/petpvc"
    for img in [v for v in results.volume_images if v.usable]:
        pvc_filename = f"{results.args.subject}_pvc_{img.frame:02d}.nii.gz"
        pvc_path = results.args.output_path / "pvc" / pvc_filename
        full_command = [
            pvc_exe,
            "-i", str(img.path / img.filename),  # orig/orig_01.nii.gz
            "-o", str(pvc_path),  # anchoring/pvc/pvc_01.nii.gz
            "-m", str(results.best_vascular_mask_path[2]),
            "-p", "STC",
            "-x", f"{results.args.fwhm:0.1f}",
            "-y", f"{results.args.fwhm:0.1f}",
            "-z", f"{results.args.fwhm:0.1f}",
        ]
        if results.args.debug:
            full_command = full_command + ["--debug", ]
        logger.debug("Running '" + " ".join(full_command) + "'")
        if pvc_path.exists() and not results.args.force:
            logger.warning(f"Skipping {str(pvc_path)}, it already exists.")
        else:
            p = subprocess.run(full_command, capture_output=True)
            logger.info(f"Ran petpvc on {img.filename} -> {pvc_path.name}")
            logger.info(p.stdout.decode("utf-8"))
            if len(p.stderr) > 0:
                logger.error("ERROR: " + p.stderr.decode("utf-8"))

        # Maintain a list of pvc_images, analogous to list of orig_images
        pvc_image = nib.Nifti1Image.from_filename(str(pvc_path))
        pvc_volumes.append(StareVolume(
            nifti=pvc_image,
            path=pvc_path.parent,
            filename=pvc_path.name,
            prefix="pvc",
            frame=img.frame,
            usable=img.usable,
        ))

    # Collect all the 3d image data into a single 4d structure.
    combined_image = combine_volumes_into_4d(
        [vol for vol in pvc_volumes if vol.usable],
        results.args.output_path / f"sub-{results.args.subject}_pvc.nii.gz",
        logger=logger
    )

    # PET data should be in units of 'mCi'
    # If they already are, good, but other units get converted here.
    pet_4d_data = combined_image.get_fdata()
    if results.args.pet_units.lower() == "kbq":
        pet_4d_data = pet_4d_data / 37000
    elif results.args.pet_units.lower() == "bq":
        pet_4d_data = pet_4d_data / 37000000

    reshaped_pvc_data = flatten_4d_to_2d(pet_4d_data)

    vascular_mask_img = nib.Nifti1Image.from_filename(
        results.best_vascular_mask_path[2]
    )
    vascular_mask_data = vascular_mask_img.get_fdata().astype(np.double)
    reshaped_vascular_mask_data = flatten_4d_to_2d(
        np.reshape(
            vascular_mask_data,
            (
                vascular_mask_data.shape[0],
                vascular_mask_data.shape[1],
                vascular_mask_data.shape[2],
                1,
            )
        )
    )

    masked_data = reshaped_pvc_data[reshaped_vascular_mask_data.ravel() == 1]

    pvc_tac = TimeActivityCurve(
        activity=np.mean(masked_data, axis=0),
        timepoints=np.array(results.mid_times),
        missing_timepoints=results.ignored_mid_times,
        sd=np.std(masked_data, axis=0),
        source="pvc",
        name="pvc",
    )
    # In the case (CerePET scans, in particular) that the TAC starts at its
    # peak and drops from there, fake it so that it seems to have risen from
    # 0.0 to its peak, so it behaves like a real TAC.
    if np.argmax(pvc_tac.activity) == 0:
        # We need to pad our TAC with a zero time point.
        pvc_tac.activity = np.insert(pvc_tac.activity, 0, 0.0)
        pvc_tac.timepoints = np.insert(pvc_tac.timepoints, 0, 0.0)
        pvc_tac.peak_index = np.argmax(pvc_tac.activity)
        pvc_tac.sd = np.insert(pvc_tac.sd, 0, 0.0)

        # And we need to pad our regional TACs to match
        results.tacs.loc[-1] = np.zeros((len(results.tacs.columns),))
        results.tacs.index = results.tacs.index + 1
        results.tacs = results.tacs.sort_index()

        # And we need to pad our mid-times to match, too
        results.mid_times = np.insert(results.mid_times, 0, 0.0)

        # And notify the user of our decision.
        logger.warning(f"The vascular peak appears to be at the first time "
                       f"point. So a zero time, zero activity point was "
                       f"inserted before the first time point in the PVC "
                       f"TAC, in the table of regional TACs, and in the "
                       f"list of mid-times. The number of time points was "
                       f"initially {len(results.tacs) - 1}, "
                       f"and is now {len(results.tacs)}.")

    results.pvc_mean_vascular_tac = pvc_tac

    if results.args.debug:
        # noinspection PyTypeChecker
        pickle.dump(
            results.pvc_mean_vascular_tac,
            open(results.args.debug_path /
                 f"sub-{results.args.subject}_tac_pvc.pkl",
                 "wb")
        )

    tacs_to_plottable_dataframe([results.pvc_mean_vascular_tac, ]).to_csv(
        results.args.output_path /
        f"sub-{results.args.subject}_step-2_pvc_mean_tac.csv",
        index=False,
    )
    logger.info(f"WROTE sub-{results.args.subject}_step-2_pvc_mean_tac.csv to "
                f"{str(results.args.output_path)}")

    # Paint a picture of progress so far
    tac_plot_data = [
        results.best_centroid(step=1),
        results.best_centroid(step=2),
        results.pvc_mean_vascular_tac,
    ]
    tac_plot_palette = {
        results.best_centroid(step=1).name: "blue",
        results.best_centroid(step=2).name: "red",
        results.pvc_mean_vascular_tac.name: "orange",
    }
    if results.plasma_tac is not None:
        tac_plot_data.append(results.plasma_tac)
        tac_plot_palette[results.plasma_tac.name] = "green"

    fig_top_tacs = plot_detailed_tacs(
        data=tac_plot_data,
        title=f"Subject {results.args.subject} Vascular TACs",
        palette=tac_plot_palette,
    )
    fig_top_tacs.savefig(results.args.fig_path /
                         f"sub-{results.args.subject}_step-2_four_tacs.png")

    caption = "All TACs through PVC"
    rpt_sect.add_figure(
        results.args.fig_path /
        f"sub-{results.args.subject}_step-2_four_tacs.png",
        caption
    )

    rpt_sect.end()
    results.write_report()
    return results
