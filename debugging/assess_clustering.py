#!/usr/bin/env python3

# assess_clustering.py

import os
import sys
import argparse
import re
import shutil
import glob
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.image import coord_transform
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def printc(s, c=""):
    """ print in color """
    import sys

    color_dict = {
        'black': "\033[0;30m",
        'dark gray': "\033[1;30m",
        'light gray': "\033[0;37m",
        'white': "\033[1;37m",
        'blue': "\033[0;34m",
        'light blue': "\033[1;34m",
        'purple': "\033[0;35m",
        'light purple': "\033[1;35m",
        'cyan': "\033[0;36m",
        'light cyan': "\033[1;36m",
        'orange': "\033[0;33m",
        'yellow': "\033[1;33m",
        'green': "\033[0;32m",
        'light green': "\033[1;32m",
        'red': "\033[0;31m",
        'light red': "\033[1;31m",
    }
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        print(color_dict.get(c.lower(), "") + s + "\033[0m")
    else:
        print(s)


def find_fs_command(cmd_name):
    """ Locate the 'mri_whatever' freesurfer command.
    """

    for mtp in [
        shutil.which(cmd_name),
        f"/opt/freesurfer/bin/{cmd_name}",
        f"/usr/local/freesurfer/bin/{cmd_name}",
    ]:
        if mtp is not None and Path(mtp).is_file():
            return mtp
    return None


def get_env(args):
    """ Integrate environment variables into our args. """

    errors = list()

    # We need access to freesurfer and ffmpeg, so check for their existence.
    # First, ask the shell, then check some other places.
    mri_tessellate_path = find_fs_command('mri_tessellate')
    if mri_tessellate_path:
        setattr(args, 'mri_tessellate', mri_tessellate_path)
        if args.verbose:
            printc(f"using '{mri_tessellate_path}' to get 3D from masks.",
                   c='green')
    else:
        errors.append("I can't find 'mri_tessellate' in the PATH or other "
                      "common locations. It's necessary for building the "
                      "3D plots, so this error is fatal.")

    mris_convert_path = find_fs_command('mris_convert')
    if mris_convert_path:
        setattr(args, 'mris_convert', mris_convert_path)
        if args.verbose:
            printc(f"using '{mris_convert_path}' to get 3D from masks.",
                   c='green')
    else:
        errors.append("I can't find 'mris_convert' in the PATH or other "
                      "common locations. It's necessary for building the "
                      "3D plots, so this error is fatal.")

    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        setattr(args, 'ffmpeg_path', Path(ffmpeg_path))
        if args.verbose:
            printc(f"using '{str(ffmpeg_path)}' to combine plots into movies.",
                   c='green')
    else:
        errors.append("I can't find 'ffmpeg_path' in the PATH, and don't know "
                      "where else to look. It's necessary for building the "
                      "movies from plot frames, so this error is fatal.")

    if len(errors) > 0:
        for error in errors:
            printc(error, c='red')
        sys.exit(1)

    return args


def validate_args(args):
    """ Ensure the environment will support the requested workflows. """

    errors = list()
    explanation_needed = False
    explanation = ""

    # Ensure paths we need exist (input, output, work)
    # If we get a wildcard, we may be running in a debugger;
    # The shell didn't expand the paths, so glob it ourselves to check.
    if len(args.subject_paths) == 1:
        if "*" in args.subject_paths[0]:
            alt_paths = list(glob.glob(args.subject_paths[0]))
            if len(alt_paths) > 0:
                setattr(args, 'subject_paths', [str(p) for p in alt_paths])
    valid_subject_paths = list()
    for subject_path in args.subject_paths:
        if Path(subject_path).exists() and Path(subject_path).is_dir():
            if args.verbose:
                printc(f"found '{subject_path}'", c='green')
                valid_subject_paths.append(Path(subject_path))
        elif Path(subject_path).is_file():
            errors.append(f"The subject_path '{subject_path}' is a file. "
                          "I expect a directory containing STARE output..")
        else:
            errors.append(f"The subject_path '{subject_path}' doesn't exist.")
            if "*" in subject_path:
                explanation_needed = True
                explanation = (
                    "It appears you've used wildcards to select STARE output "
                    "directories. This should have worked because your shell "
                    "should have expanded them, and this script should have "
                    "seen the expanded list of paths, not the wild cards.\n"
                    "This error will happen if your expression with wildcards "
                    "doesn't actually find any paths. Try doing an 'ls' of "
                    "your expression to check. If you have quotation "
                    "marks around the path, try removing them. Or if you're "
                    "running from a debugger or IDE, it may not expand the "
                    "wildcards the way a proper shell would."
                )
    if len(valid_subject_paths) > 0:
        # Replace full list of strings with validated list of Paths
        setattr(args, 'subject_paths', valid_subject_paths)

    if (    Path(args.output_path).exists() and
            Path(args.output_path).is_dir()
    ):
        if args.verbose:
            printc(f"writing final movie to '{args.output_path}'.", 'green')
    elif Path(args.output_path).is_file():
        errors.append("The output-path needs to be a directory, but you've "
                      f"specified a file. I can't use '{args.output_path}'.")
    elif args.force:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        if args.verbose:
            printc(f"creating the output-path '{args.output_path}'.", 'green')
    else:
        errors.append(f"The output-path '{args.output_path}' doesn't exist, "
                      "which is fine, but I can't create it, "
                      "which causes a problem.")

    if (    Path(args.work_path).exists() and
            Path(args.work_path).is_dir()
    ):
        if args.verbose:
            printc(f"writing temp files to '{args.work_path}'.", 'green')
    elif Path(args.work_path).is_file():
        errors.append("The work-path needs to be a directory, but you've "
                      f"specified a file. I can't use '{args.work_path}'.")
    elif args.force:
        Path(args.work_path).mkdir(parents=True, exist_ok=True)
        if args.verbose:
            printc(f"creating the work-path '{args.work_path}'.", 'green')
    else:
        errors.append(f"The work-path '{args.work_path}' doesn't exist, "
                      "which is fine, but I can't create it, "
                      "which causes a problem.")

    # Anybody wanting a dry-run would want to see what's being planned.
    if args.dry_run:
        setattr(args, "verbose", True)

    if len(errors) > 0:
        for error in errors:
            printc(error, c='red')
        if explanation_needed:
            printc(explanation, c='cyan')
        sys.exit(1)

    return args


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description=(
            "Create 3D plots of step-1 and step-2 k-means clusters, "
            "and 2D plots of spatial statistics for each STARE run "
            "specified, and package them all into a single video."
        ),
    )
    parser.add_argument(
        "subject_paths", nargs="+",
        help=(
            "One or more paths to complete Python STARE runs. A movie will be "
            "created for each path, then they will all be concatenated in the "
            "order they were given as arguments. If the --order-by argument "
            "is used, these paths will be ordered by 'clip' and 'sr' instead."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=".",
        help="The output path for writing the final movie",
    )
    parser.add_argument(
        "--work-path",
        default=".",
        help=(
            "The output path for writing intermediate files, "
            "which will be deleted. This should be a fast, "
            "local SSD if possible."
        ),
    )
    parser.add_argument(
        "--order-by",
        default=None, type=str,
        help=(
            "Provide either 'clip' or 'sr' to re-order the paths provided by "
            "that property of the STARE run first, and the other property "
            "second. You can test the ordering by using --dry-run and "
            "--verbose to see the order before actually running anything."
        ),
    )
    parser.add_argument(
        "--camera-elevation",
        default=10, type=int,
        help=(
            "The angle above the horizon for the camera observing the 3D scene."
            " Default is looking slightly down onto the brain, angle 20. "
            "To look up from below, negative values may be used."
        ),
    )
    parser.add_argument(
        "--azimuth-center",
        default=90, type=int,
        help=(
            "The center point of rotation in each video. "
            "A value of 90 is looking the image in the face, in "
            "clinical orientation, with the subject looking back at you. "
            "0 looks at the subject's right ear as they look to their "
            "left (your right)."
        ),
    )
    parser.add_argument(
        "--azimuth-range",
        default=30, type=int,
        help=(
            "The range of rotation around 'azimuth-center', "
            "Half of this will be rotated left of center, "
            "and half to the right. Larger values provide both a larger "
            "viewing angle and a longer video clip."
        ),
    )
    parser.add_argument(
        "--paint-debug-grids-on-slides",
        action="store_true",
        help=(
            "To aid in debugging plot placement on movie frames, paint "
            "red frames around subplots and text boxes."
        ),
    )
    parser.add_argument(
        "--include-pca",
        action="store_true",
        help=(
            "If PCA data are found in the output directory, render their "
            "masks alongside clusters in 3D plot."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Run all tasks, even if it means overwriting existing data. "
            "This includes creating paths if they don't exist."
        ),
    )
    parser.add_argument(
        "--leave-intermediates", action="store_true",
        help=(
            "Set true to leave all the intermediate files around. "
            "By default, the .surf and .stl files are left in the "
            "STARE output masks/ directory, but the individual movie "
            "frames and clips are deleted."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Set to quit after finding STARE output directories."
            "This implies --verbose since there's no point otherwise."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Set to trigger verbose output.",
    )

    args = parser.parse_args()
    args = get_env(args)
    args = validate_args(args)

    return args


def make_clip_mesh(stare_out_path, slices_clipped, verbose=False):
    """ build a 3D plane and return it for plotting
    """

    # Find the original image shape
    orig_mean_imgs = list(Path(stare_out_path).glob("sub-*_orig_mean.nii.gz"))
    if len(orig_mean_imgs) == 0:
        return None
    img = nib.load(orig_mean_imgs[0])

    # Figure out the extents of the image
    x0, y0, z0 = coord_transform(0, 0, 0, img.affine)
    x1, y1, z1 = coord_transform(img.shape[0], img.shape[1], img.shape[2], img.affine)
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)
    min_z, max_z = min(z0, z1), max(z0, z1)
    if verbose:
        print(f"Image shape {img.shape}, and affine:")
        print(img.affine)
        print(f"x from {min_x:0.1f} to {max_x:0.1f}; "
              f"y from {min_y:0.1f} to {max_y:0.1f}; "
              f"z from {min_z:0.1f} to {max_z:0.1f}")

    # override the max_z with the top of the clipping range
    # rather than the top of the whole image.
    max_z = coord_transform(0, 0, slices_clipped, img.affine)[2]
    if verbose:
        print(f"max_z overridden to top of clipping range, {max_z:0.1f}")

    vertices = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ])
    # Define the faces and edges of the cube
    faces = np.array([
        [0, 3, 1], [1, 3, 2], [0, 4, 7], [0, 7, 3],
        [4, 5, 6], [4, 6, 7], [5, 1, 2], [5, 2, 6],
        [2, 3, 6], [3, 7, 6], [0, 1, 5], [0, 5, 4],
    ])
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])

    # Format the vertices and faces into proper STL
    clipping_cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            clipping_cube.vectors[i][j] = vertices[face[j], :]

    return clipping_cube


def remove_old_3d_files(stare_out_path):
    """ Clean up any prior version STL or surf stuff. """

    # All original files in this path should start with "clust*"
    # So the s* files were all added by us and can be removed.
    for old_file in stare_out_path.glob("masks/s*"):
        printc(f"    removing {old_file}", c='orange')
        old_file.unlink()


def extract_pca_masks(stare_out_path, threshold=95):
    """
    :param stare_out_path: path to STARE output
    :param threshold: percentile threshold for masking PCA weights
    :return:
    """

    pca_file = stare_out_path / "debug" / "components" / "pca_6.nii.gz"
    if pca_file.exists():
        pca_img = nib.Nifti1Image.from_filename(pca_file)
        mask_files = list()
        for i in range(6):
            pca_data = pca_img.get_fdata()[:, :, :, i].squeeze()
            pca_mask = np.array(
                pca_data > np.percentile(pca_data, threshold),
                dtype=np.uint8
            )
            pca_mask_img = nib.Nifti1Image(pca_mask, pca_img.affine)
            file_path = str(pca_file).replace("_6", f"_{i}_mask")
            pca_mask_img.to_filename(file_path)
            mask_files.append(Path(file_path))
        return mask_files
    return None


def best_mask(stare_out_path, step):
    """ Get the path to the best cluster mask from step 'step'
    """

    src_mask = Path(stare_out_path) / "masks" / f"cluster_step-{step}_best_mask_orig.nii.gz"
    if not src_mask.exists():
        src_mask = Path(stare_out_path) / "masks" / f"cluster_step-{step}_best_mask.nii.gz"
    if not src_mask.exists():
        printc(f"ERROR: I can't find a step {step} mask at "
               f"'{str(stare_out_path)}'.", c="red")
        return None
    return src_mask


def get_mesh_from_mask(mask_file, force_rebuild=False):
    """ Build a mesh from a 3D nifti mask.
    """

    mask_file = Path(mask_file)
    stl_file = Path(str(mask_file).replace(".nii.gz", ".stl"))
    surf_file = Path(str(mask_file).replace(".nii.gz", ".surf"))
    if not stl_file.exists() or force_rebuild:
        # Extract vertices and faces from the volumetric Nifti file.
        mri_tessellate_cmd = find_fs_command('mri_tessellate')
        p1 = subprocess.run(
            [mri_tessellate_cmd, "-n", str(mask_file), "1", str(surf_file), ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p1.returncode != 0:
            printc(f"{p1.stderr.decode('utf-8')}", c="red")
            printc(f"ERROR: I can't tessellate '{mask_file.name}' "
                   f"to '{surf_file.name}'.", c="red")
            return None

        # Convert the surface file to STL
        mris_convert_cmd = find_fs_command('mris_convert')
        p2 = subprocess.run(
            [mris_convert_cmd, str(surf_file), str(stl_file), ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p2.returncode != 0:
            printc(f"{p2.stderr.decode('utf-8')}", c="red")
            printc(f"ERROR: I can't convert '{surf_file.name}' "
                   f"to '{stl_file.name}'.", c="red")
            return None

    return mesh.Mesh.from_file(str(stl_file))


def build_initial_3d_figure(
        stare_out_path,
        s1_mesh=None, s2_mesh=None, clip_mesh=None,
        pca0_mesh=None, pca1_mesh=None,
):
    """ Build a 3d image of the vascular clusters
    """

    # Find the original image shape
    orig_mean_imgs = list(Path(stare_out_path).glob("sub-*_orig_mean.nii.gz"))
    if len(orig_mean_imgs) == 0:
        return None
    img = nib.load(orig_mean_imgs[0])

    # Figure out the extents of the image
    x0, y0, z0 = coord_transform(0, 0, 0, img.affine)
    x1, y1, z1 = coord_transform(img.shape[0], img.shape[1], img.shape[2], img.affine)
    min_x, max_x = min(x0, x1), max(x0, x1)
    min_y, max_y = min(y0, y1), max(y0, y1)
    min_z, max_z = min(z0, z1), max(z0, z1)

    # This plot will need to fit into white space 960 px by 540 px
    # But we oversize it so we can crop the excess white space later.
    _fig = plt.figure(figsize=(10, 7), layout='tight')
    _axes = _fig.add_subplot(projection='3d')

    ls = LightSource(azdeg=225, altdeg=45.0)

    # Create, color, and light the PCA 0 mesh (blue/gray)
    if pca0_mesh is not None:
        pca0_alpha = 0.10
        color_pca0 = np.array((0.8, 0.8, 1.0, pca0_alpha))
        mesh_pca0 = mplot3d.art3d.Poly3DCollection(
            pca0_mesh.vectors,
            shade=False,
            facecolors=color_pca0,
            lightsource=ls
        )
        _axes.add_collection3d(mesh_pca0)

    # Create, color, and light the step 1 mesh (green/gray)
    if pca1_mesh is not None:
        pca1_alpha = 0.10
        color_pca1 = np.array((0.8, 1.0, 0.8, pca1_alpha))
        mesh_pca1 = mplot3d.art3d.Poly3DCollection(
            pca1_mesh.vectors,
            shade=False,
            facecolors=color_pca1,
            lightsource=ls
        )
        _axes.add_collection3d(mesh_pca1)

    # Create, color, and light the step 1 mesh (yellow)
    if s1_mesh is not None:
        step_1_alpha = 0.40
        color_step_1 = np.array((1.0, 1.0, 57 / 255.0, step_1_alpha))
        mesh_step_1 = mplot3d.art3d.Poly3DCollection(
            s1_mesh.vectors,
            shade=False,
            facecolors=color_step_1,
            lightsource=ls
        )
        _axes.add_collection3d(mesh_step_1)

    # Create, color, and light the step 2 mesh
    if s2_mesh is not None:
        step_2_alpha = 1.0
        color_step_2 = np.array((1.0, 54.0 / 255.0, 57 / 255.0, step_2_alpha))
        mesh_step_2 = mplot3d.art3d.Poly3DCollection(
            s2_mesh.vectors,
            alpha=step_2_alpha, shade=True,
            facecolors=color_step_2, edgecolors=color_step_2,
            lightsource=ls
        )
        _axes.add_collection3d(mesh_step_2)

    # Create, color, and light the step 2 mesh
    if clip_mesh is not None:
        clip_alpha = 0.2
        color_clip = np.array((55.0 / 255.0, 54.0 / 255.0, 1.0, clip_alpha))
        mesh_clipping_cube = mplot3d.art3d.Poly3DCollection(
            clip_mesh.vectors,
            alpha=clip_alpha, shade=False,
            facecolors=color_clip, edgecolors=color_clip,
            lightsource=ls
        )
        _axes.add_collection3d(mesh_clipping_cube)

    """
    x_coords = np.concat([
        clip_mesh.points[:, col] for col in (0, 3, 6)
    ] + [
        s1_mesh.points[:, col] for col in (0, 3, 6)
    ] + [
        s2_mesh.points[:, col] for col in (0, 3, 6)
    ])
    y_coords = np.concat([
        clip_mesh.points[:, col] for col in (1, 4, 7)
    ] + [
        s1_mesh.points[:, col] for col in (1, 4, 7)
    ] + [
        s2_mesh.points[:, col] for col in (1, 4, 7)
    ])
    z_coords = np.concat([
        clip_mesh.points[:, col] for col in (2, 5, 8)
    ] + [
        s1_mesh.points[:, col] for col in (2, 5, 8)
    ] + [
        s2_mesh.points[:, col] for col in (2, 5, 8)
    ])
    all_points = np.concat([
        s1_mesh.points.flatten(),
        s2_mesh.points.flatten(),
        clip_mesh.points.flatten(),
    ])
    _axes.auto_scale_xyz(x_coords, y_coords, z_coords)
    """
    _axes.set_xlabel("x (-L to +R)")
    # _axes.invert_yaxis()  # We want clinical orientation, looking at the face
    _axes.set_ylabel("y (+A to -P)")
    _axes.set_zlabel("z (-I to +S)")

    _axes.xaxis.pane.fill = False
    _axes.yaxis.pane.fill = False
    _axes.zaxis.pane.fill = False

    _axes.view_init(azim=75, elev=10)  # This will be changed repeatedly later

    _axes.set_xlim((min_x, max_x))
    _axes.set_ylim((min_y, max_y))
    _axes.set_zlim((min_z, max_z))

    return _fig, _axes


def make_frame_with_stats(mask_3d_file, clip=0, sr=0, title="",
                          leave_intermediates=False, paint_debug_grids=False):
    """ Expand the size of an existing image, adding text
    """

    # Create new HD image.
    font_dir = Path("/usr/share/fonts/truetype/google")
    image = Image.new("RGB", (1920, 1080), "white")
    drawer = ImageDraw.Draw(image)

    # Set some parameters
    margin = 32
    spacing = 3
    title_size = 64
    subtitle_size = 48
    # text_size = 28  # Text wasn't used, but this is a good size to add some

    # Add stat text to the image; deprecated and replaced with an image
    """
    x = margin + inset_image.width + margin + 100  # manual 100 for aesthetics
    y = margin + title_height + 100  # manual 100 for aesthetics
    stats_font = ImageFont.truetype(font_dir / 'Roboto-Light.ttf', text_size)
    drawer.text((x, y), "Text 16 starts here", font=stats_font, fill="black")
    y = y + text_size + spacing
    drawer.text((x, y), "Text 20 starts here", font=stats_font, fill="black")
    y = y + text_size + spacing
    drawer.text((x, y), "Text 20 starts here", font=stats_font, fill="black")
    y = y + text_size + spacing
    drawer.text((x, y), "Text 24 starts here", font=stats_font, fill="black")
    y = y + text_size + spacing
    drawer.text((x, y), "Text 24 starts here", font=stats_font, fill="black")
    y = y + text_size + spacing
    """

    # Add the stats plots to the image
    stat_image = Image.open(
        Path(mask_3d_file).parent / f"stats_{clip:02d}_{sr:02d}.png"
    )
    if (stat_image.width != 1920) or (stat_image.height != 1080):
        print(f"ERROR: Stats image ({stat_image.width}x{stat_image.height})")
    stat_left = 0  # image.width - stat_image.width - margin
    stat_top = 0  # image.height - stat_image.height - margin
    image.paste(stat_image, (stat_left, stat_top))
    if paint_debug_grids:
        drawer.rectangle(
            (stat_left, stat_top,
             stat_left + stat_image.width, stat_top + stat_image.height),
            fill=None, outline='red'
        )

    # Open the 3D image and paste it into the new larger background.
    inset_image = Image.open(mask_3d_file)
    # White space left and right of 3D image can overlap axes labels,
    # so crop it just a bit. There's plenty of white space to give.
    # It was plotted 1000x700, to fill a space 920x540,
    # but the image only takes up about 600x500 in the center,
    # depending on the rotation
    horizontal_crop = int((1000 - 920) / 2)
    vertical_crop = int((700 - 540) / 2)
    inset_image = inset_image.crop(
        (horizontal_crop,
         vertical_crop,
         inset_image.width - horizontal_crop,
         inset_image.height - vertical_crop)
    )
    inset_left = 480 + horizontal_crop  # margin
    inset_top = 180  # image.height - inset_image.height - margin
    image.paste(inset_image, (inset_left, inset_top))
    if paint_debug_grids:
        drawer.rectangle(
            (inset_left, inset_top,
             inset_left + inset_image.width, inset_top + inset_image.height),
            fill=None, outline='red'
        )

    # Add titles to the image.
    title_font = ImageFont.truetype(
        font_dir / 'Roboto-Bold.ttf', title_size
    )
    tw = drawer.textlength(title, font=title_font)
    x = int((image.width / 2) - (tw / 2))
    y = margin
    drawer.text(
        (x, y), title, font=title_font, fill='black'
    )
    if paint_debug_grids:
        drawer.rectangle((x, y, x + tw, y + title_size),
                         fill=None, outline='red')

    subtitle_text = f"clip-{clip} sr-{sr}"
    subtitle_font = ImageFont.truetype(
        font_dir / 'Roboto-Medium.ttf', subtitle_size
    )
    sw = drawer.textlength(subtitle_text, font=subtitle_font)
    x = int((image.width / 2) - (sw / 2))
    y = margin + title_size + (2 * spacing)  # double spacing just for titles
    drawer.text(
        (x, y), subtitle_text, font=subtitle_font, fill='black'
    )
    if paint_debug_grids:
        drawer.rectangle((x, y, x + sw, y + subtitle_size), fill=None, outline='red')

    # Save it back out.
    image.save(str(mask_3d_file).replace("mask_3d", "frame"))

    # Clean up after ourselves.
    if not leave_intermediates:
        Path(mask_3d_file).unlink()


def build_rotated_plots(
        fig, axes, clip, sr, subject_id,
        middle_azimuth=90, azimuth_range=30, camera_elevation=10,
        paint_debug_grids_on_slides=False,
        work_dir="/var/tmp", leave_intermediates=False, verbose=False
):
    """ Build each frame of a movie by rotating fig around middle_azimuth.
    """

    # Spin it around, saving images of each perspective
    start_azimuth = int(middle_azimuth - (azimuth_range / 2))
    end_azimuth = int(middle_azimuth + (azimuth_range / 2))
    pairs_printed_this_line = 0
    print("", flush=True)  # Start on a new line; the earlier prints used end="".

    for i, azimuth in enumerate(range(start_azimuth, end_azimuth + 1, 1)):
        if verbose:
            print(f" {azimuth:03d}", end="", flush=True)
        mask_file = Path(work_dir) / f"mask_3d_{azimuth:03d}.png"
        axes.view_init(azim=azimuth, elev=camera_elevation)
        fig.savefig(mask_file)
        make_frame_with_stats(mask_file, clip, sr,
                              title=f"Subject {subject_id}",
                              leave_intermediates=leave_intermediates,
                              paint_debug_grids=paint_debug_grids_on_slides)
        frame_file = str(mask_file).replace("mask_3d", "frame")
        # Make another copy of the same file so the video can reverse
        dupe_num = end_azimuth + 1 + (end_azimuth - start_azimuth - i)
        if verbose:
            if pairs_printed_this_line > 8:
                end_char = None
                pairs_printed_this_line = 0
            else:
                end_char = ""
                pairs_printed_this_line += 1
            print(f"/{dupe_num:03d},", end=end_char, flush=True)
        dupe_frame_file = Path(work_dir) / f"frame_{dupe_num:03d}.png"
        shutil.copy(frame_file, dupe_frame_file)
    if verbose:
        print("", flush=True)


def combine_plots_into_movie(mp4_prefix, working_dir,
                             leave_intermediates=False, verbose=False):
    """ Find all png files in a directory, and combine them into a movie.
    """

    movie_filename = f"{mp4_prefix}_3d_mask.mp4"
    ffmpeg_command = [
        "ffmpeg", "-y", "-framerate", "30", "-pattern_type", "glob",
        "-i", "'frame_*.png'", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        movie_filename,
    ]
    p = subprocess.run(" ".join(ffmpeg_command),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       shell=True, cwd=str(working_dir))
    if p.returncode == 0:
        if verbose:
            printc(f"  {movie_filename} successful")
            # printc(p.stdout.decode("utf8"))
    else:
        print("ffmpeg failed")
        print(p.stderr.decode("utf8"))

    if not leave_intermediates:
        num_slides = 0
        num_stats = 0
        for individual_slide in Path(working_dir).glob("frame_*.png"):
            num_slides += 1
            individual_slide.unlink()
        for individual_stats in Path(working_dir).glob("stats_*.png"):
            num_stats += 1
            individual_stats.unlink()
        if verbose:
            printc(f"  removed {num_stats} stats figures and "
                   f"{num_slides} slides after making their movie.")

    return Path(working_dir) / movie_filename


def make_stats_plots(
        metadata, run_data, verbose=False
):
    """ Make a grid of plots describing metadata.

        This plot will be the full-size slide for each movie frame,
        leaving white space to be overlaid with the 3D plot.
    """

    # Build a figure amenable to a 1080p (1920x1080) movie resolution
    _fig = plt.figure(figsize=(19.2, 10.8), layout='tight')

    gs = GridSpec(nrows=6, ncols=4)
    #         0                 1                 2                 3
    #   +---------------------------------------------------------------------+
    # 0 |                                Title                                |
    #   +---------------+ +---------------------------------+ +---------------+
    # 1 |     blobs     | |                                 | |     blobs     |
    #   +---------------+ |                                 | +---------------+
    # 2 |     voxels    | |        3D rotating image        | |     voxels    |
    #   +---------------+ |                                 | +---------------+
    # 3 |  voxels/blob  | |                                 | |  voxels/blob  |
    #   +---------------+ +---------------------------------+ +---------------+
    # 4 | bottom weight | |                                 | | bottom weight |
    #   +---------------+ |           TAC Peaks             | +---------------+
    # 5                   |                                 |
    #                     +---------------------------------+

    vasc_filter = metadata['feature_likely_vascular']
    step_filter = (metadata['step'] == 1)
    # We use k=1 sometimes when overriding the cluster with a custom mask.
    k_filter = ((metadata['k'] > 15) | (metadata['k'] == 1))
    best_filter = metadata['best_overall']
    run_filter = metadata['run'] == run_data['run']
    # if run_data['clip'] is not None:
    #     clip_filter = metadata['slices_clipped'] == run_data['clip']
    #     clip_ticks = [0, 5, 10, 15, 20, 25, 30]
    # else:
    #     clip_filter = metadata['slices_clipped'] >= 0  # all True
    #     clip_ticks = list()
    # if run_data['sr'] is not None:
    #     sr_filter = metadata['sr'] == run_data['sr']
    #     sr_ticks = [0, 5, 10, 15, 20]
    # else:
    #     sr_filter = metadata['sr'] >= 0  # all True
    #     sr_ticks = list()
    # print(f"  make_stats_plots got slices_clipped={slices_clipped}; "
    #       f"{len(metadata['slices_clipped'].unique())} unique values in data: "
    #       f"[{', '.join([f'{v}' for v in metadata['slices_clipped'].unique()])}]; "
    #       f"{len(metadata[clip_filter])} rows with {slices_clipped} clipped.")

    # Only handles clips OR sr, not both
    x_var = "run"
    # ticks = clip_ticks
    # if sparsity_reduced is not None and slices_clipped is not None:
    #     x_var = "sr"
    #     ticks = sr_ticks

    for row, y_var, y_name in [
            (0, "blob_count", "blobs"),
            (1, "voxel_count", "voxels"),
            (2, "voxels_per_blob", "voxels/blob"),
            (3, "feature_inf_weighted_score", "bottom weight")
    ]:
        if y_var not in metadata.columns:
            if verbose:
                print(f"  excluding {y_var} from stats plot; not in metadata")
            continue
        # These four plots land on rows 1-4, column 0
        ax = _fig.add_subplot(gs[row + 1, 0])
        sns.stripplot(
            data=metadata[vasc_filter & step_filter & k_filter],
            x=x_var, y=y_var, color='gray', alpha=0.50,
            native_scale=True, ax=ax,
        )
        hi_filter = metadata[y_var] > 250000
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & hi_filter],
            x=x_var, y=y_var, color='red', s=70, ax=ax,
        )
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter & run_filter],
            x=x_var, y=y_var, color='blue', s=100, ax=ax,
        )
        ax.set_title(f"{y_name} (vasc clusters, k>15)")
        ax.set_ylabel(y_name)
        # ax.set_xticks(ticks)
        # ax.set_xticklabels([f"{t}" for t in ticks])
        if row < 2:
            ax.set_xlabel("")

        # These four plots land on rows 1-4, column 3
        ax = _fig.add_subplot(gs[row + 1, 3])
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter],
            x=x_var, y=y_var, color='black', s=50, ax=ax,
        )
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter & run_filter],
            x=x_var, y=y_var, color='blue', s=150, ax=ax,
        )
        ax.set_title(f"{y_name} (best clusters)")
        ax.set_ylabel(y_name)
        # ax.set_xticks(ticks)
        # ax.set_xticklabels([f"{t}" for t in ticks])
        if row < 2:
            ax.set_xlabel("")

    # This TAC-like plot is larger, landing on the bottom, centered
    ax = _fig.add_subplot(gs[4:6, 1:3])
    sns.scatterplot(
        data=metadata[vasc_filter & step_filter & k_filter],
        x='peak_index', y='peak_value', color='gray', alpha=0.50, ax=ax,
    )
    sns.scatterplot(
        data=metadata[vasc_filter & step_filter & k_filter & best_filter],
        x='peak_index', y='peak_value', color='black', s=50, ax=ax,
    )
    hi_filter = (metadata['voxel_count'] > 250000)
    sns.scatterplot(
        data=metadata[vasc_filter & step_filter & k_filter & hi_filter],
        x='peak_index', y='peak_value', color='red', s=200, ax=ax,
    )
    sns.scatterplot(
        data=metadata[vasc_filter & step_filter & k_filter & best_filter & run_filter],
        x='peak_index', y='peak_value', color='blue', s=150, ax=ax,
    )
    max_x = metadata['peak_index'].max() + 1
    ax.set_xlim((0, max_x))
    ax.set_xticks([t for t in range(max_x)])
    ax.set_xticklabels([f"{t}" for t in range(max_x)])

    return _fig


def find_clipping_threshold(stare_out_path):
    """ Look in the log file to determine how many slices were clipped.
    """
    log_files = sorted(stare_out_path.glob("stare_pet_*.log"), reverse=True)
    if len(log_files) == 0:
        print("no log file found")
        return 0
    else:
        for log_file in log_files:
            print(f"Looking for clipping value in {log_file.parent.name} / {log_file.name}")
            with open(log_file, "r") as f:
                for line in f:
                    match = re.search(r".*--axial-slices-to-clip[\s]+([0-9]+).*", line)
                    if match:
                        print(f"  found it: {match.groups}")
                        return int(match.group(1))
    print(f"  never found it")
    return 0


def gather_subject_directories(subject_dirs, order_by=None):
    """ Collect paths for clipped clustering runs of one subject, in clip order
    """

    nosr_pattern = re.compile(r"_nosr")
    sr_pattern = re.compile(r"sr-([0-9]+)")
    clip_pattern = re.compile(r"clip-([0-9]+)")
    or_pattern = re.compile(r"or-([a-z0-9]+)")
    path_dicts = list()
    for sop in subject_dirs:
        sr = 0  # If no SR was specified, sparsity reduction was not applied
        clip = 0  # If no clip was specified, nothing was clipped
        override = ""  # If no override was specified, it was not applied
        if nosr_pattern.search(str(sop)):
            sr = 0
        elif sr_pattern.search(str(sop)):
            sr = int(sr_pattern.search(str(sop)).group(1))
        if clip_pattern.search(str(sop)):
            clip = int(clip_pattern.search(str(sop)).group(1))
        if or_pattern.search(str(sop)):
            override = or_pattern.search(str(sop)).group(1)

        path_dicts.append({
            'path': sop,
            'clip_subdir': sop.parent.name,
            'subject_id': sop.name,
            'output_dir': sop.name,
            'clip': clip,
            'sr': sr,
            'override': override,
            'run': f"{clip}(-{sr}%)",
        })

    if order_by == "sr":
        return sorted(path_dicts, key=lambda x: (x['sr'], x['clip']))
    elif order_by == "clip":
        return sorted(path_dicts, key=lambda x: (x['clip'], x['sr']))
    elif order_by is not None:
        printc(f"I can only sort by 'clip' or 'sr', not '{order_by}', "
               f"so the paths will remain in their original order.",
               'red')

    return path_dicts


def main():
    """ Entry point """

    args = get_arguments()

    # Find the directories we'd like to include in our assessment.
    stare_result_dirs = gather_subject_directories(
        args.subject_paths, args.order_by
    )
    if args.verbose:
        if args.order_by is None:
            order_str = ""
        else:
            order_str = f", in {args.order_by}- order"
        printc(f"The STARE runs found{order_str}:")
        for res_dir in stare_result_dirs:
            print(f" - {res_dir['path']}")

    # Clear any existing full movies or lists of movie parts
    subjects = set([d['subject_id'] for d in stare_result_dirs])
    if len(subjects) == 1:
        movie_subject = f"sub-{subjects.pop()}"
    elif len(subjects) > 1:
        movie_subject = "multisubject"
    else:
        movie_subject = "no_subs_prob_broken"
    vid_list_file = Path(args.work_path) / f"{movie_subject}_vids.txt"
    movie_file = Path(args.work_path) / f"{movie_subject}.mp4"

    if args.dry_run:
        if movie_file.exists():
            printc(f"{movie_file} already exists. Use --force to overwrite.")
        printc("Quitting before any work gets done, --dry-run was set.",
               c='cyan')
        sys.exit(0)

    if movie_file.exists() and not args.force:
        printc(f"{movie_file} already exists. Use --force to overwrite.",
               c='red')
        sys.exit(1)

    # We are either making a new movie, or forcing an overwrite.
    # So delete any final outputs so we can write clean copies.
    vid_list_file.unlink(missing_ok=True)
    movie_file.unlink(missing_ok=True)
    movie_pieces = list()

    # -------------------------------------------------------------------------
    # 0. Gather metadata (don't actually _DO_ anything)
    # -------------------------------------------------------------------------
    metadata_dataframes = list()
    for res_dir in stare_result_dirs:
        if args.force:
            remove_old_3d_files(res_dir['path'])
        for step in (1, 2):
            # Dataframes have a 'subject' and a 'step' column, so can be combined
            # after we add a 'slices_clipped' column.
            metadata_file = (
                Path(res_dir['path']) / "debug" /
                f"sub-{res_dir['subject_id']}_vasc_clust_step-{step}_metadata.csv"
            )
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                df['slices_clipped'] = res_dir['clip']
                df['sr'] = res_dir['sr']
                df['run'] = res_dir['run']
                metadata_dataframes.append(df)
            else:
                printc(f"skipping {res_dir['output_dir']} step {step} "
                       "because it doesn't exist.", c='yellow')
    blob_metadata = pd.concat(metadata_dataframes)
    if args.verbose:
        print(f"Built {blob_metadata.shape}-shaped metadata.")

    # -------------------------------------------------------------------------
    # 1. For each run found, build everything
    # -------------------------------------------------------------------------
    # On the second pass, build 3D plots, 2D plots, and annotations.
    for i, res_dir in enumerate(stare_result_dirs):
        if args.verbose:
            printc(f"{i + 1:02d}/{len(stare_result_dirs):02d}. "
                   f"sub-{res_dir['subject_id']} "
                   f"in subdir '{res_dir['clip_subdir']}' "
                   f"clip-{res_dir['clip']:02d} "
                   f"sr-{res_dir['sr']:02d}",
                   c="cyan")
        # ---------------------------------------------------------------------
        # 1 A. Make 3D meshes and render them in a 3D figure.
        # ---------------------------------------------------------------------
        # If PCA was requested, masks must be created then made into meshes
        pca_0_mesh, pca_1_mesh = None, None
        if args.include_pca:
            pca_masks = extract_pca_masks(res_dir['path'])
            if pca_masks is not None:
                pca_0_mesh = get_mesh_from_mask(
                    pca_masks[0], force_rebuild=args.force
                )
                # pca_1_mesh = get_mesh_from_mask(
                #     pca_masks[1], force_rebuild=args.force
                # )
        step_1_mesh = get_mesh_from_mask(
            best_mask(res_dir['path'], 1), force_rebuild=args.force
        )
        if args.verbose:
            print(f"m1 ", end="", flush=True)
        step_2_mesh = get_mesh_from_mask(
            best_mask(res_dir['path'], 2), force_rebuild=args.force
        )
        if args.verbose:
            print(f"m2 ", end="", flush=True)
        clip_mesh = make_clip_mesh(res_dir['path'], res_dir['clip'])
        if args.verbose:
            print(f"mC ", end="", flush=True)

        # Build the initial figure
        fig1, axes1 = build_initial_3d_figure(
            res_dir['path'],
            s1_mesh=step_1_mesh,
            s2_mesh=step_2_mesh,
            pca0_mesh=pca_0_mesh,
            pca1_mesh=pca_1_mesh,
            clip_mesh=clip_mesh,
        )
        if args.verbose:
            print(f"f1 ", end="", flush=True)

        # ---------------------------------------------------------------------
        # 1 B. Build statistical plots, which act as backgrounds for each frame
        # ---------------------------------------------------------------------
        fig2 = make_stats_plots(blob_metadata, res_dir)
        fig2.savefig(
            Path(args.work_path) /
            f"stats_{res_dir['clip']:02d}_{res_dir['sr']:02d}.png"
        )
        plt.close(fig2)
        if args.verbose:
            print(f"f2 ", end="", flush=True)

        # ---------------------------------------------------------------------
        # 1 C. With the 3D plot, move the camera around, rendering each frame
        # ---------------------------------------------------------------------
        build_rotated_plots(
            fig1,
            axes1,
            res_dir['clip'],
            res_dir['sr'],
            res_dir['subject_id'],
            middle_azimuth=args.azimuth_center,
            azimuth_range=args.azimuth_range,
            camera_elevation=args.camera_elevation,
            work_dir=args.work_path,
            leave_intermediates=args.leave_intermediates,
            paint_debug_grids_on_slides=args.paint_debug_grids_on_slides,
            verbose=args.verbose,
        )

        # ---------------------------------------------------------------------
        # 1 D. Build each frame, drawing the 3D image onto the stats background
        # ---------------------------------------------------------------------
        plt.close(fig1)
        one_clip_movie_file = Path(
            combine_plots_into_movie(
                f"sub-{res_dir['subject_id']}_clip-{res_dir['clip']}_sr-{res_dir['sr']}",
                args.work_path,
                leave_intermediates=args.leave_intermediates,
                verbose=args.verbose
            )
        )
        movie_pieces.append(one_clip_movie_file)

        if args.verbose:
            printc(f"  movie '{one_clip_movie_file.name}' made "
                   f"at '{one_clip_movie_file.parent}'", )
        # shutil.copy(movie_file,
        #             Path(res_dir['path']) / "masks" / movie_file.name)
        # subprocess.run("rm mask_3d_*.png", shell=True, cwd=str(work_dir))

    # -------------------------------------------------------------------------
    # 2. Collect each frame and concatenate them into an mp4 movie.
    # -------------------------------------------------------------------------
    # Make a list of movie parts for ffmpeg to concatenate.
    for movie_piece in movie_pieces:
        # print(f"copying {movie_piece.name} to {args.work_path}/")
        # shutil.copy(movie_piece, Path(args.work_path) / movie_piece.name)
        with open(vid_list_file, "a") as f:
            f.write(f"file '{movie_piece.name}'\n")

    # Concatenate them
    ffmpeg_concat_command = (
        f"ffmpeg -f concat -i {vid_list_file.name} -c copy {movie_file.name}"
    )
    if args.verbose:
        print(f"Executing '{ffmpeg_concat_command}'")
    ffmpeg_proc = subprocess.run(
        ffmpeg_concat_command, shell=True, cwd=args.work_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if ffmpeg_proc.returncode != 0:
        printc(ffmpeg_proc.stderr.decode("utf-8"), c='red')
    else:
        final_path = Path(args.output_path) / movie_file.name
        movie_file.rename(final_path)
        if not args.leave_intermediates:
            for movie_piece in movie_pieces:
                movie_piece.unlink()
            vid_list_file.unlink()
        printc(f"Final movie complete at '{str(final_path)}'", c='green')


if __name__ == "__main__":
    main()
