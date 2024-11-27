#!/usr/bin/env python3

# assess_clipping_effect_on_clustering.py

import os
import sys
import argparse
import re
import shutil
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

    # Accept subject id with or without the 'sub-'
    if args.subject.startswith("sub-"):
        setattr(args, "subject", args.subject[4:])

    # Ensure paths we need exist (input, output, work)
    if (    Path(args.input_path).exists() and
            Path(args.input_path).is_dir()
    ):
        if args.verbose:
            printc(f"starting the search in '{args.input_path}'", c='green')
    elif Path(args.input_path).is_file():
        errors.append(f"The input-path '{args.input_path}' is a file. I can't "
                      "search for subject directories within it.")
    else:
        errors.append(f"The input-path '{args.input_path}' doesn't exist.")

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
        sys.exit(1)

    return args


def get_arguments():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(
        description=(
            "Find STARE runs for a subject with different clipping. "
            "Create 3D plots of each k-means cluster, 2D plots of "
            "spatial statistics, and package them all into a video."
        ),
    )
    parser.add_argument(
        "subject",
        help=(
            "The subject's ID, matching the STARE output directory names. "
            "With or without 'sub-' work the same"
        ),
    )
    parser.add_argument(
        "--input-path",
        default=".",
        help=(
            "Any path upstream of the STARE output directories. "
            "This is the start point of a search, so the closer the better."
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
        default=60, type=int,
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


def get_mesh_from_mask(stare_out_path, step):
    """ Build a mesh from a 3D nifti mask.
    """

    stl_file = Path(stare_out_path) / "masks" / f"step-{step}_mask.stl"
    if not stl_file.exists():

        # Extract vertices and faces from the volumetric Nifti file.
        mri_tessellate_cmd = find_fs_command('mri_tessellate')
        src_mask = Path(stare_out_path) / "masks" / f"cluster_step-{step}_best_mask_orig.nii.gz"
        if not src_mask.exists():
            src_mask = Path(stare_out_path) / "masks" / f"cluster_step-{step}_best_mask.nii.gz"
        if not src_mask.exists():
            printc(f"ERROR: I can't find a step {step} mask at "
                   f"'{str(stare_out_path)}'.", c="red")
            return None
        surf_file = Path(stare_out_path) / "masks" / f"step-{step}_mask.surf"
        p1 = subprocess.run(
            [mri_tessellate_cmd, str(src_mask), "1", str(surf_file), ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if p1.returncode != 0:
            printc(f"{p1.stderr.decode('utf-8')}", c="red")
            printc(f"ERROR: I can't tessellate '{src_mask.name}' "
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


def build_initial_3d_figure(s1_mesh, s2_mesh, clip_mesh):
    """ Build a 3d image of the vascular clusters
    """

    _fig = plt.figure(figsize=(10, 9), layout='tight')
    _axes = _fig.add_subplot(projection='3d')

    ls = LightSource(azdeg=225, altdeg=45.0)

    # Create, color, and light the step 1 mesh
    if s1_mesh is not None:
        step_1_alpha = 0.20
        color_step_1 = np.array((1.0, 1.0, 57 / 255.0, step_1_alpha))
        mesh_step_1 = mplot3d.art3d.Poly3DCollection(
            s1_mesh.vectors,
            shade=False,
            color=color_step_1,
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

    _axes.view_init(azim=75, elev=20)  # This will be changed repeatedly later

    return _fig, _axes


def make_frame_with_stats(mask_3d_file, clip_thresh=0, title="",
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

    # Open the existing image and paste it into the new larger background.
    inset_image = Image.open(mask_3d_file)
    inset_left = margin
    inset_top = image.height - inset_image.height - margin
    image.paste(inset_image, (inset_left, inset_top))
    if paint_debug_grids:
        drawer.rectangle(
            (inset_left, inset_top,
             inset_left + inset_image.width, inset_top + inset_image.height),
            fill=None, outline='red'
        )

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
        Path(mask_3d_file).parent / f"stats_{clip_thresh:03d}.png"
    )
    stat_left = image.width - stat_image.width - margin
    stat_top = image.height - stat_image.height - margin
    image.paste(stat_image, (stat_left, stat_top))
    if paint_debug_grids:
        drawer.rectangle(
            (stat_left, stat_top,
             stat_left + stat_image.width, stat_top + stat_image.height),
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

    subtitle_text = f"clip-{clip_thresh}"
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
        fig, axes, clip_threshold, subject_id,
        middle_azimuth=90, azimuth_range=30, paint_debug_grids_on_slides=False,
        work_dir="/var/tmp", verbose=False
):
    """ Build each frame of a movie by rotating fig around middle_azimuth.
    """

    # Spin it around, saving images of each perspective
    start_azimuth = int(middle_azimuth - (azimuth_range / 2))
    end_azimuth = int(middle_azimuth + (azimuth_range / 2))

    for i, azimuth in enumerate(range(start_azimuth, end_azimuth + 1, 1)):
        if verbose:
            print(f" {azimuth:03d}", end="", flush=True)
        mask_file = Path(work_dir) / f"mask_3d_{azimuth:03d}.png"
        axes.view_init(azim=azimuth, elev=10)
        fig.savefig(mask_file)
        make_frame_with_stats(mask_file, clip_threshold,
                              title=f"Scanner Subject {subject_id}",
                              paint_debug_grids=paint_debug_grids_on_slides)
        frame_file = str(mask_file).replace("mask_3d", "frame")
        # Make another copy of the same file so the video can reverse
        dupe_num = end_azimuth + 1 + (end_azimuth - start_azimuth - i)
        if verbose:
            print(f"/{dupe_num:03d},", end="", flush=True)
        dupe_frame_file = Path(work_dir) / f"frame_{dupe_num:03d}.png"
        shutil.copy(frame_file, dupe_frame_file)
    if verbose:
        print("", flush=True)


def combine_plots_into_movie(subject_id, clip_thresh, working_dir,
                             leave_intermediates=False, verbose=False):
    """ Find all png files in a directory, and combine them into a movie.
    """

    movie_filename = f"sub-{subject_id}_clip-{clip_thresh:02d}_3d_mask.mp4"
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
            printc("ffmpeg good")
            printc(p.stdout.decode("utf8"))
    else:
        print("ffmpeg failed")
        print(p.stderr.decode("utf8"))

    if not leave_intermediates:
        num_slides = 0
        for individual_slide in Path(working_dir).glob("frame_*.png"):
            num_slides += 1
            individual_slide.unlink()
        if verbose:
            printc(f"  removed {num_slides} slides after making movie")

    return Path(working_dir) / movie_filename


def make_stats_plots(metadata, slices_clipped):
    """ Make a grid of plots describing metadata.
    """

    _fig = plt.figure(figsize=(8.5, 9), layout='tight')
    gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 2])

    vasc_filter = metadata['feature_likely_vascular']
    step_filter = (metadata['step'] == 1)
    k_filter = (metadata['k'] > 15)
    best_filter = metadata['best_overall']
    clip_filter = metadata['slices_clipped'] == slices_clipped

    clip_ticks = [0, 5, 10, 15, 20, 25, 30]

    for row, y_var in [(0, "blob_count"), (1, "voxel_count"), (2, "voxels_per_blob"), ]:
        ax = _fig.add_subplot(gs[row, 0])
        sns.stripplot(
            data=metadata[vasc_filter & step_filter & k_filter],
            x='slices_clipped', y=y_var, color='gray', alpha=0.50, ax=ax,
        )
        hi_filter = metadata[y_var] > 250000
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & hi_filter],
            x='slices_clipped', y=y_var, color='red', s=70, ax=ax,
        )
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter & clip_filter],
            x='slices_clipped', y=y_var, color='blue', s=100, ax=ax,
        )
        ax.set_title(f"{y_var} in vascular clusters with k>15")
        ax.set_xticks(clip_ticks)
        ax.set_xticklabels([f"{t}" for t in clip_ticks])
        if row < 2:
            ax.set_xlabel("")

        ax = _fig.add_subplot(gs[row, 1])
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter],
            x='slices_clipped', y=y_var, color='black', s=50, ax=ax,
        )
        sns.scatterplot(
            data=metadata[vasc_filter & step_filter & k_filter & best_filter & clip_filter],
            x='slices_clipped', y=y_var, color='blue', s=150, ax=ax,
        )
        ax.set_title(f"{y_var} in selected best clusters")
        ax.set_xticks(clip_ticks)
        ax.set_xticklabels([f"{t}" for t in clip_ticks])
        if row < 2:
            ax.set_xlabel("")

    ax = _fig.add_subplot(gs[3, :])
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
        data=metadata[vasc_filter & step_filter & k_filter & best_filter & clip_filter],
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
    log_files = sorted(stare_out_path.glob("stare_pet_*.log"))
    if len(log_files) == 0:
        print("no log file found")
        return 0
    else:
        with open(log_files[-1], "r") as f:
            for line in f:
                match = re.search(r".*--axial-slices-to-clip[\s]+([0-9]+).*", line)
                if match:
                    return int(match.group(1))


def write_fsl_script(img_path, any_stare_output_path):
    """ Write an fsl script to view a background and provided image.
    """

    # Find an averaged PET for background
    bg_paths = list(Path(any_stare_output_path).glob("sub-*_orig_mean.nii.gz"))
    if len(bg_paths) == 0:
        bg_paths = list(Path(any_stare_output_path).glob("sub-*_mean.nii.gz"))
    if len(bg_paths) == 0:
        bg_path = None
    else:
        bg_path = str(bg_paths[0])
    img_path = str(img_path)

    # Write a script to open images in fsleyes
    script_file = str(Path(img_path).parent / "view_in_fsl.sh")
    with open(script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("fsleyes \\\n")
        if bg_path is not None:
            f.write(f"  {bg_path} --name \"Average PET\" --overlayType volume \\\n")
        f.write(f"  {img_path} --name \"Clusters\" --overlayType label \\\n")
    os.chmod(script_file, 0o755)

    return script_file


def gather_subject_directories(starting_path, subject_id):
    """ Collect paths for clipped clustering runs of one subject, in clip order
    """

    # TODO: Make this more flexible, it's dependent on my strange directories.
    # TODO: Dedupe somehow, I only want one per clip. The 'si' thing is brittle

    stare_paths = list()
    for sop in Path(starting_path).glob(f"cluster*/*{subject_id}"):
        if (
                (sop.name in [subject_id, f"sub-{subject_id}"]) and
                ("si" not in sop.parent.name) and
                ("so" not in sop.parent.name)
        ):
                stare_paths.append({
                    'path': sop,
                    'clip_subdir': sop.parent.name,
                    'subject_id': sop.name,
                    'clip_thresh': find_clipping_threshold(sop),
                })

    return sorted(stare_paths, key=lambda x: x['clip_thresh'])


def main():
    """ Entry point """

    args = get_arguments()

    # Find the directories we'd like to include in our assessment.
    stare_result_dirs = gather_subject_directories(
        args.input_path, args.subject
    )
    if args.verbose:
        printc("The STARE runs found, in clip- order of presentation:")
        for res_dir in stare_result_dirs:
            print(f" - {res_dir['path']}")

    # Clear any existing full movies or lists of movie parts
    vid_list_file = Path(args.work_path) / f"sub-{args.subject}_vids.txt"
    movie_file = Path(args.work_path) / f"sub-{args.subject}.mp4"

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

    # On the first pass, just collect the blob metadata.
    metadata_dataframes = list()
    for res_dir in stare_result_dirs:
        for step in (1, 2):
            # Dataframes have a 'subject' and a 'step' column, so can be combined
            # after we add a 'slices_clipped' column.
            metadata_file = (
                Path(res_dir['path']) / "debug" /
                f"sub-{res_dir['subject_id']}_vasc_clust_step-{step}_metadata.csv"
            )
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                df['slices_clipped'] = res_dir['clip_thresh']
                metadata_dataframes.append(df)
            else:
                printc(f"skipping {res_dir['subject_id']} step {step} "
                       "because it doesn't exist.", c='yellow')
    blob_metadata = pd.concat(metadata_dataframes)
    if args.verbose:
        print(f"Built {blob_metadata.shape}-shaped metadata.")

    # On the second pass, build 3D plots, 2D plots, and annotations.
    for i, res_dir in enumerate(stare_result_dirs):
        if args.verbose:
            print(f"{i + 1:02d}/{len(stare_result_dirs):02d}. "
                  f"sub-{res_dir['subject_id']} "
                  f"in subdir '{res_dir['clip_subdir']}' "
                  f"clip-{res_dir['clip_thresh']:03d}")
        step_1_mesh = get_mesh_from_mask(res_dir['path'], 1)
        if args.verbose:
            print(f"m1 ", end="", flush=True)
        step_2_mesh = get_mesh_from_mask(res_dir['path'], 2)
        if args.verbose:
            print(f"m2 ", end="", flush=True)
        clip_mesh = make_clip_mesh(res_dir['path'], res_dir['clip_thresh'])
        if args.verbose:
            print(f"mC ", end="", flush=True)

        # Build the initial figure
        fig1, axes1 = build_initial_3d_figure(
            step_1_mesh, step_2_mesh, clip_mesh,
        )
        if args.verbose:
            print(f"f1 ", end="", flush=True)
        fig2 = make_stats_plots(
            blob_metadata, slices_clipped=res_dir['clip_thresh']
        )
        fig2.savefig(
            Path(args.work_path) / f"stats_{res_dir['clip_thresh']:03d}.png"
        )
        if args.verbose:
            print(f"f2 ", end="", flush=True)
        build_rotated_plots(
            fig1, axes1, res_dir['clip_thresh'], res_dir['subject_id'],
            middle_azimuth=args.azimuth_center,
            azimuth_range=args.azimuth_range,
            work_dir=args.work_path,
            verbose=args.verbose,
        )
        one_clip_movie_file = Path(
            combine_plots_into_movie(
                res_dir['subject_id'], res_dir['clip_thresh'], args.work_path,
                remove_individual_slides=False, verbose=args.verbose
            )
        )
        movie_pieces.append(one_clip_movie_file)

        if args.verbose:
            printc(f"movie '{one_clip_movie_file.name}' made "
                   f"at '{one_clip_movie_file.parent}'",
                   c='green')
        # shutil.copy(movie_file,
        #             Path(res_dir['path']) / "masks" / movie_file.name)
        # subprocess.run("rm mask_3d_*.png", shell=True, cwd=str(work_dir))

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
