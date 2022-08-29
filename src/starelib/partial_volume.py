import logging
import subprocess
from datetime import datetime
from pathlib import Path
import nibabel as nib
import numpy as np

from .util import Image, combine_volumes_into_4d, flatten_4d_to_2d
from .timeactivitycurve import TimeActivityCurve


def correct_partial_volumes(
        images, fwhm, output_path, vasc_mask, mid_times=None, pet_units='kBq',
        force=False, verbose=False
):
    """ Correct partial volumes

        :param list images: a list of paths to original 3D PET images
        :param float fwhm: The full-width-half-maximum value for PVC correction
        :param Path output_path: the main output path for one subject
        :param Path vasc_mask: a mask delineating a vascular ROI
        :param iterable mid_times: an iterable of time values for each sample
        :param str pet_units: a string indicating units used for PET data
        :param int force: If true, run everything regardless of cache
        :param int verbose: Set to non-zero to trigger logging, higher is more
        :return:
    """

    logger = logging.getLogger("STARE")

    pre_pvc_timestamp = datetime.now()
    logger.info(f"Started PVC at {pre_pvc_timestamp}")

    # Create a path for our partial-volume data
    fig_path = output_path / "anchoring" / "pvc"
    fig_path.mkdir(parents=True, exist_ok=True)

    # Perform PVC on each of the original volumes provided
    pvc_images = []
    pvc_exe = "/usr/local/bin/petpvc"
    for img in images:
        pvc_path = output_path / "anchoring" / "pvc" / f"pvc_{img.frame:02d}.nii.gz"
        full_command = [
            pvc_exe,
            "-i", str(img.path / img.filename),  # {output_path}/orig/orig_01.nii.gz
            "-o", str(pvc_path),  # {output_path}/anchoring/pvc/pvc_01.nii.gz
            "-m", str(vasc_mask),  # {output_path}/anchoring/figs-masks/{best_mask}.nii.gz
            "-p", "STC",
            "-x", f"{fwhm:0.1f}", "-y", f"{fwhm:0.1f}", "-z", f"{fwhm:0.1f}",
        ]
        if verbose:
            full_command = full_command + ["--debug", ]
        logger.debug("Running '" + " ".join(full_command) + "'")
        if pvc_path.exists() and not force:
            logger.warning(f"Skipping {str(pvc_path)}, it already exists.")
        else:
            p = subprocess.run(full_command, capture_output=True)
            logger.info(f"Ran petpvc on {img.filename} -> {pvc_path.name}")
            logger.info(p.stdout.decode("utf-8"))
            if len(p.stderr) > 0:
                logger.error("ERROR: " + p.stderr.decode("utf-8"))

        # Maintain a list of pvc_images, analogous to list of orig_images
        pvc_images.append(Image(
            path=pvc_path.parent,
            filename=pvc_path.name,
            prefix="pvc",
            frame=img.frame,
            nifti=nib.load(str(pvc_path)),
        ))

    # Collect all the 3d image data into a single 4d structure.
    combined_image = combine_volumes_into_4d(
        pvc_images, output_path / "pvc.nii.gz", logger=logger
    )

    # PET data should be in units of 'mCi'
    # If they already are, good, but other units get converted here.
    pet_4d_data = combined_image.get_fdata()
    if pet_units.lower() == "kbq":
        pet_4d_data = pet_4d_data / 37000
    elif pet_units.lower() == "bq":
        pet_4d_data = pet_4d_data / 37000000

    reshaped_pvc_data = flatten_4d_to_2d(pet_4d_data)

    vascular_mask_data = nib.load(vasc_mask).get_fdata().astype(np.double)
    reshaped_vascular_mask_data = flatten_4d_to_2d(
        np.reshape(
            vascular_mask_data,
            (vascular_mask_data.shape[0],
             vascular_mask_data.shape[1],
             vascular_mask_data.shape[2],
             1)
        )
    )

    masked_data = reshaped_pvc_data[reshaped_vascular_mask_data.ravel() == 1]
    vascular_tac_mean = np.mean(masked_data, axis=0)
    # vascular_tac_sd = np.std(masked_data, axis=0)

    return TimeActivityCurve(
        activity=vascular_tac_mean,
        timepoints=np.array(mid_times),
        source="pvc",
    )
