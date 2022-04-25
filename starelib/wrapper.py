from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
import nibabel as nib
import re
from starelib.util import *


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


def get_mid_times(input_path, subject_id):
    """ Find a mid-times file and read its data

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
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
            else:
                logging.warning(f"Ignoring extra mid_times file '{actual_f}'")
    return mid_times


def get_images(input_path, subject_id):
    """ Find images, a volume for each mid-time

    :param input_path: path to find subjects
    :param subject_id: name of subject folder
    :return dict: key-value dict with image data
    """

    images = []
    image_dir = Path(input_path) / subject_id / "moco"
    for pattern in ["{subject}.*.MCFI.hdr", "*.nii", "*.nii.gz", ]:
        actual_pattern = pattern.format(subject=subject_id)
        for img_file in sorted(image_dir.glob(actual_pattern)):
            match = re.search(r"[._-](\d+)[._-]", img_file.name)
            if match:
                frame = match.group(1)
            else:
                frame = None
            logging.info(f"Reading volume '{img_file}'")
            img = nib.load(img_file)
            logging.debug(f"  frame {frame} "
                          f"shaped {img.shape if img is not None else 'n/a'}")
            images.append({
                "path": img_file.parent,
                "name": img_file.name,
                "frame": frame,
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
    mid_times = get_mid_times(args.input_path, args.subject)
    images = get_images(args.input_path, args.subject)

    # Run vascular k-means clustering
    rslt1 = vascular_clustering(tacs)

    # Correct partial volumes from vascular clustering
    rslt2 = correct_partial_volumes(mid_times)

    # Correct TACs by extracting the mean signal from each cluster
    rslt3 = fit_vascular_mean_tac(images)
    # Then apply vascular correction
    rslt1 = tac_vascular_correction(rslt1)

    # Bootstrap signal in PVC'ed vasculature to generate input functions
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
