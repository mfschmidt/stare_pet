import numpy as np
import pyvista as pv
import nibabel as nib
import re
import math
from scipy.ndimage import center_of_mass
from pathlib import Path


def find_clipping_threshold(stare_out_path):
    """
    Find the clipping threshold by parsing log files within the given path.

    This function looks through log files named using the pattern 'stare_pet_*.log'
    in the specified directory. It scans the contents of these files for a
    specific pattern indicating the axial slices to clip. If found, it
    extracts and returns the integer clipping threshold. If no such value
    is found or no log files are present, it defaults to returning 0.

    Parameters
    ----------
    stare_out_path : pathlib.Path
        The directory path containing the log files to be analyzed.

    Returns
    -------
    int
        The axial slicing clipping threshold extracted from the log files.
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


def pc_brain_likelihood(pc_img, save_img_as=None, verbose=False):
    """ """

    # TODO: Make the sphere round in world coordinates rather than voxel
    # TODO: Expand the sphere to twice that radius; pc0 is cortex not ventricles
    position = center_of_mass(pc_img.get_fdata())
    mass = np.sum(pc_img.get_fdata().astype(bool))
    # If the brain were a perfect sphere with all voxels packed perfectly inside:
    radius = ((3 * mass) / (4 * math.pi)) ** (1. / 3)
    # But we want to account for ventricles, nonsphericity, etc.,
    # and still capture most/all the brain.
    radius = 2 * radius

    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    dims = len(pc_img.shape)
    assert len(position) == dims
    semisizes = (radius,) * dims

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, pc_img.shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(pc_img.shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    sphere_mask = arr <= 1.0

    voxels_inside_sphere = np.sum(sphere_mask & pc_img.get_fdata().astype(bool))
    voxels_outside_sphere = np.sum(~sphere_mask & pc_img.get_fdata().astype(bool))

    if save_img_as is not None:
        sphere_img = nib.nifti1.Nifti1Image(sphere_mask.astype(np.uint8), pc_img.affine)
        sphere_img.to_filename(save_img_as)

    if verbose:
        print(f"{voxels_inside_sphere:,} voxels inside the sphere, {voxels_outside_sphere:,} outside")

    return voxels_inside_sphere / (voxels_inside_sphere +  voxels_outside_sphere)


def get_nifti_world_corners(nii_img, verbose=False):
    """
    Extracts the world-space coordinates of the corners of a 3D NIfTI image.

    This function computes the coordinates of the corners of a three-dimensional
    NIfTI image in world space using the affine transformation matrix stored in
    the NIfTI image. By default, it performs silently but can optionally provide
    details about the voxel and world-space corner coordinates through logs.

    Parameters
    ----------
    nii_img : nibabel.Nifti1Image
        The NIfTI image object to compute the world-space corner coordinates for.
        Must include the affine transformation matrix and the image data.
    verbose : bool, optional
        A flag to control whether detailed logs about voxel and world-space
        coordinates of the corners are printed. Defaults to False.

    Returns
    -------
    numpy.ndarray
        A 3 x 8 array containing the world-space coordinates of the image corners.
        Each column corresponds to a corner of the input NIfTI volume.

    """

    n_x, n_y, n_z = nii_img.shape
    voxel_corners = np.array([
        [0, n_x-1, 0, n_x-1, 0, n_x-1, 0, n_x-1],
        [0, 0, n_y-1, n_y-1, 0, 0, n_y-1, n_y-1],
        [0, 0, 0, 0, n_z-1, n_z-1, n_z-1, n_z-1],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ])
    world_corners_homogeneous = nii_img.affine @ voxel_corners
    world_corners = world_corners_homogeneous[:3, :]

    if verbose:
        print(voxel_corners)
        print(world_corners)

    return world_corners


def nifti_img_to_stage(nii_img, verbose=False):
    """
    Using a 3D NIfTI image for scene boundaries, return floor and back wall meshes.

    This function processes a 3D NIfTI image file to extract its spatial information,
    constructs corresponding bounding geometries (floor and back wall), and generates
    a combined PyVista mesh. Optionally, details about the image and its derived
    properties can be printed based on user preference.

    Parameters
    ----------
    nii_img : nib.nifti1.Nifti1Image
        The NIfTI file to be processed.
    verbose : bool, optional
        If True, prints detailed information about the NIfTI image and the resulting
        mesh. Default is False.

    Returns
    -------
    pyvista.PolyData
        A PyVista mesh containing the floor and back wall geometries derived from
        the spatial properties of the input image.
    """

    # Load the data and pull orientation info from the header/affine
    zooms = np.diag(nii_img.affine)[:3]
    origin = nii_img.affine[:3, 3]

    # Calculate the world-coordinate bounding box for the image
    world_corners = get_nifti_world_corners(nii_img)
    world_origin = (
        np.min(world_corners[0, :]),
        np.min(world_corners[1, :]),
        np.min(world_corners[2, :]),
    )

    # Report on our findings if the caller cares
    if verbose:
        print(f"  Image dimensions: {nii_img.shape}")
        print(f"  Image affine:\n{nii_img.affine}")
        print(f"  Image orientation: {nib.aff2axcodes(nii_img.affine)}")
        print(f"  Spacing: {zooms}")
        print(f"  Affine Origin : {origin}")
        print(f"  World Origin : {world_origin}")

    # Create a grid of ALL voxels, regardless of value
    min_x, max_x = np.min(world_corners[0, :]), np.max(world_corners[0, :]) + abs(zooms[0])
    min_y, max_y = np.min(world_corners[1, :]), np.max(world_corners[1, :]) + abs(zooms[1])
    min_z, max_z = np.min(world_corners[2, :]), np.max(world_corners[2, :]) + abs(zooms[2])
    # Build a floor at min_z (with S+, max_z is the top)
    floor_mesh = pv.PolyData(
        np.array([
            [min_x, min_y, min_z, ],
            [min_x, max_y, min_z, ],
            [max_x, max_y, min_z, ],
            [max_x, min_y, min_z, ],
        ]),
        np.array([4, 0, 1, 2, 3]),
    )
    # Build a back wall at min_y (with A+, max_y is the front)
    back_mesh = pv.PolyData(
        np.array([
            [min_x, min_y, min_z, ],
            [max_x, min_y, min_z, ],
            [max_x, min_y, max_z, ],
            [min_x, min_y, max_z, ],
        ]),
        np.array([4, 0, 1, 2, 3]),
    )
    mesh = pv.merge([floor_mesh, back_mesh, ])

    if verbose:
        print(f"  Bounds : {mesh.bounds}")
        print(f"  Center : {mesh.center}")

    return mesh


def clip_box_mesh(nii_img, clip_slices, verbose=False):
    """
    Return a mesh representation of the clipped portion of the volume.

    This function extracts world-coordinate bounding box information from a 3D NIfTI
    image and generates a rectangular mesh representing to the defined slice
    range in the z-dimension. Optionally provides verbose information about
    the image and clipping parameters.

    Parameters
    ----------
    nii_img : nib.Nifti1Image
        The NIfTI format image containing the 3D imaging data.
    clip_slices : int
        Number of slices to include in the clipped bounding box along the z-axis.
    verbose : bool
        If True, prints detailed information about image properties, world
        orientation, and clipping bounds.

    Returns
    -------
    pyvista.PolyData
        A 3D rectangular mesh model (cube) with bounds clipped along the z-axis
        up to the specified `clip_slices` slices in world coordinates.
    """

    # Load the data and pull orientation info from the header/affine
    zooms = np.diag(nii_img.affine)[:3]
    origin = nii_img.affine[:3, 3]

    # Calculate the world-coordinate bounding box for the image
    world_corners = get_nifti_world_corners(nii_img)
    world_origin = (
        np.min(world_corners[0, :]),
        np.min(world_corners[1, :]),
        np.min(world_corners[2, :]),
    )

    # Report on our findings if the caller cares
    if verbose:
        print(f"  Image dimensions: {nii_img.shape}")
        print(f"  Image affine:\n{nii_img.affine}")
        print(f"  Image orientation: {nib.aff2axcodes(nii_img.affine)}")
        print(f"  Spacing: {zooms}")
        print(f"  Affine Origin : {origin}")
        print(f"  World Origin : {world_origin}")

    # Create a grid of ALL voxels, regardless of value
    min_x, max_x = np.min(world_corners[0, :]), np.max(world_corners[0, :]) + abs(zooms[0])
    min_y, max_y = np.min(world_corners[1, :]), np.max(world_corners[1, :]) + abs(zooms[1])
    min_z = np.min(world_corners[2, :])
    max_z = min_z + clip_slices * abs(zooms[2])

    # Build a cube around the clipped data
    mesh = pv.Cube(
        bounds=(min_x, max_x, min_y, max_y, min_z, max_z),
    )

    if verbose:
        print(f"  Bounds : {mesh.bounds}")
        print(f"  Center : {mesh.center}")

    return mesh


def nifti_mask_to_mesh(mask_img, method="voxel", verbose=False):
    """
    Converts a NIfTI mask file to a 3D surface mesh.

    This function processes a NIfTI mask image to generate a 3D mesh either
    using a voxel thresholding approach or marching cubes algorithm. It handles
    NIfTI file loading, orientation adjustments based on affine transformations,
    and creates a `pyvista` mesh for visualization. Verbose mode provides
    additional details about the loaded NIfTI file and mesh generation process.

    Parameters
    ----------
    mask_img : nibabel.nifti1.Nifti1Image
        The NIfTI mask image to be processed.

    method : str, optional
        The method for mesh generation. Must be one of:
        - 'voxel': Uses thresholding to create the mesh directly from voxel values.
        - 'marching cubes': Applies the marching cubes algorithm for continuous
          surface generation. Default method is 'voxel'.

    verbose : bool, optional
        If True, logs detailed information about the NIfTI file, orientation,
        and generated mesh properties. If False, suppresses logging messages.
        Default is False.

    Returns
    -------
    pyvista.DataSet
        The generated 3D mesh object, using either voxel thresholding or the
        marching cubes algorithm, based on the specified `method`.
    """

    # Load the data and pull orientation info from the header/affine
    img_data = mask_img.get_fdata()
    zooms = np.diag(mask_img.affine)[:3]
    origin = mask_img.affine[:3, 3]

    # Calculate the world-coordinate bounding box for the image
    world_corners = get_nifti_world_corners(mask_img)
    world_origin = (
        np.min(world_corners[0, :]),
        np.min(world_corners[1, :]),
        np.min(world_corners[2, :]),
    )

    # Report on our findings if the caller cares
    if verbose:
        print(f"  Image dimensions: {mask_img.shape}")
        print(f"  Image affine:\n{mask_img.affine}")
        print(f"  Image orientation: {nib.aff2axcodes(mask_img.affine)}")
        print(f"  Spacing: {zooms}")
        print(f"  Affine Origin : {origin}")
        print(f"  World Origin : {world_origin}")

    # Order the matrix to RAS+ world space (scaled via spacing later)
    if zooms[0] < 0:
        if verbose:
            print(f"  -- detected negative zoom (x), flipping image data & "
                  f"shifting origin by {mask_img.shape[0] * zooms[0]:0.2f}")
        img_data = np.flip(img_data, axis=0)
        # origin[0] = origin[0] + ((mask_img.shape[0] - 1) * zooms[0])
    if zooms[1] < 0:
        if verbose:
            print(f"  -- detected negative zoom (y), flipping image data & "
                  f"shifting origin by {mask_img.shape[1] * zooms[1]:0.2f}")
        img_data = np.flip(img_data, axis=1)
        # origin[1] = origin[1] + ((mask_img.shape[1] - 1) * zooms[1])
    if zooms[2] < 0:
        if verbose:
            print(f"  -- detected negative zoom (z), flipping image data & "
                  f"shifting origin by {mask_img.shape[2] * zooms[2]:0.2f}")
        img_data = np.flip(img_data, axis=2)
        # origin[2] = origin[2] + ((mask_img.shape[2] - 1) * zooms[2])

    # if verbose:
    #     print(f"  Modified Origin : {origin}")

    # Create a grid of ALL voxels, regardless of value
    grid = pv.ImageData(
        dimensions=mask_img.shape,
        spacing=abs(zooms),
        origin=world_origin
    )
    grid.point_data['mask_values'] = img_data.flatten(order='F')

    # Either bound the voxels explicitly or find a boundary between 0s and 1s
    if method == "voxel":
        mesh = grid.threshold(
            1, scalars="mask_values"
        )
    elif method == "marching cubes":
        mesh = grid.contour(
            [0.5, ], scalars="mask_values", method="marching_cubes"
        )
    else:
        raise ValueError(f"Unrecognized method: '{method}'. "
                         f"Only 'voxel' and 'marching cubes' are supported.")

    if verbose:
        print(f"  Bounds ({method}): {mesh.bounds}")
        print(f"  Center ({method}): {mesh.center}")

    return mesh


def render_masks(stare_output_path,
                 window_size=(1920, 1072), background_color="aliceblue",
                 brain_color="cornsilk", clip_color="dodgerblue",
                 step_1_color="lemonchiffon", step_2_color="red",
                 include_pca=True,
                 step_1_nii_file=None, step_2_nii_file=None, pc_file=None,
                 camera_up_angle = 15.0, camera_left_angle = 20.0,
                 title=None, save_image=True, save_movie=False, output_file=None,
                 show_in_notebook=False,
                 verbose=False):
    """
    Renders 3D masks and related anatomical structures from provided or default neuroimaging
    data files to create visualizations. The rendering includes optional PCA visualization,
    specific neurovascular clusters, a clipping plane, and labeled orientation markers. Camera
    angles and position can be adjusted to create custom viewing perspectives, and the resulting
    scene can be saved as a static image or a movie.

    Parameters
    ----------
    stare_output_path : Path
        Path to the directory containing STARE output data.
    window_size : tuple of int, optional
        Size of the rendering window in pixels as (width, height). Default is (1920, 1080).
    background_color : str, optional
        Color of the stage's background in the visualization. Default is "aliceblue".
    brain_color : str, optional
        Color of the brain shape in the scene. Default is "cornsilk".
    clip_color : str, optional
        Color of the clipping plane in the scene. Default is "dodgerblue".
    include_pca : bool, optional
        Whether to include PCA visualization using the `pc_file`. Default is True.
    step_1_nii_file : Path, optional
        Path to the NIfTI file for the vascular cluster produced during step 1. If not provided,
        a default path relative to `stare_output_path` is used.
    step_2_nii_file : Path, optional
        Path to the NIfTI file for the vascular cluster produced during step 2. If not provided,
        a default path relative to `stare_output_path` is used.
    pc_file : Path, optional
        Path to the NIfTI file for the principal component analysis mask. This is used if
        `include_pca` is True. If not provided, a default path relative to `stare_output_path`
        is used.
    camera_up_angle : float, optional
        Angle in degrees to tilt the camera upward from the center plane. Default is 15.0.
    camera_left_angle : float, optional
        Angle in degrees to offset the camera to the left of the scene. Default is 20.0.
    title : str, optional
        If provided, the title will be used as a label on the back wall of the visualization.
    save_image : bool, optional
        Whether to save the resulting scene as a static png image.
        Default is True.
        The image will be named the same as the step_1_nii_file, but with .png, if unspecified
    save_movie : bool, optional
        Whether to save the resulting scene as a movie file in addition to a static image.
        Default is False. The movie will be named the same as the output png file, with
        an mp4 extension.
    output_file : str, optional
        Path to the output png file. If not provided, the file will be written to the
        stare_output_path/figures/ directory alongside other figures.
    show_in_notebook : bool, optional
        If set to True, pyvista should render an interactive window in the calling
        Jupyter notebook. By default, False, rendering will happen quietly in the background.
        Default is False.
    verbose : bool, optional
        Whether to print detailed diagnostic information during processing and rendering.
        Default is False.

    Returns
    -------
    a dict containing the paths and plotter used to create the visualization.
    """

    # Resolve files
    if step_2_nii_file is None and step_1_nii_file is not None:
        # Explicitly render only the provided step_1, ignoring step_2
        step_2_nii_file = Path("/DOES/NOT/EXIST/FILE.nii.gz")
    if step_1_nii_file is None:
        step_1_nii_file = stare_cluster_path / "cluster_step-1_best_mask_orig.nii.gz"
        if not step_1_nii_file.exists():
            step_1_nii_file = stare_output_path / "masks/cluster_step-1_best_mask.nii.gz"
    if step_2_nii_file is None:
        step_2_nii_file = stare_output_path / "masks/cluster_step-2_best_mask_orig.nii.gz"
        if not step_2_nii_file.exists():
            step_2_nii_file = stare_output_path / "masks/cluster_step-2_best_mask.nii.gz"
    if include_pca and pc_file is None:
        pc_file = stare_output_path / "debug/components/pca_6.nii.gz"

    exemplar_file = step_1_nii_file
    for file in [step_1_nii_file, step_2_nii_file, pc_file, ]:
        if file is not None and file != False and file.exists():
            exemplar_file = file
            break
    exemplar_img = nib.nifti1.Nifti1Image.from_filename(exemplar_file)
    if verbose:
        print(f"  Exemplar file: {exemplar_file.name}")

    # Build the plotter and set the background color to white.
    if show_in_notebook:
        pv.set_jupyter_backend("trame")
    else:
        pv.OFF_SCREEN=True

    p = pv.Plotter(window_size=window_size,
                   off_screen=True,
                   line_smoothing=True,
                   polygon_smoothing=True, )
    p.set_background(color=(1.0, 1.0, 1.0))

    # Add a "stage", a floor and a back wall, to the scene.
    stage_mesh = nifti_img_to_stage(exemplar_img)
    _ = p.add_mesh(
        stage_mesh,
        color=background_color,
        opacity=0.3,
        show_scalar_bar=False,
        show_edges=True,
    )
    # Add a "brain" shape to the scene.
    if include_pca and pc_file.exists():
        if verbose:
            print(f"  Adding PCA mask: {pc_file.name}")


        pc_img = nib.nifti1.Nifti1Image.from_filename(pc_file)
        mask_images = list()
        for i in range(6):
            pc_data = pc_img.get_fdata()[:, :, :, i].squeeze()
            pc_mask = np.array(
                pc_data > np.percentile(pc_data, 95),
                dtype=np.uint8
            )
            pc_mask_img = nib.Nifti1Image(pc_mask, pc_img.affine)
            _pct_in = pc_brain_likelihood(
                pc_mask_img,
                save_img_as=str(pc_file).replace(".nii.gz", "_sphere.nii.gz"),
                verbose=True
            )
            print(f"writing sphere to {str(pc_file).replace('.nii.gz', '_sphere.nii.gz')}")
            print(f"pc {i} {_pct_in:.2%} score indicating brain.")
            # file_path = str(pc_file).replace("_6", f"_{i}_mask")
            # pca_mask_img.to_filename(file_path)
            if _pct_in > 0.90:
                mask_images.append(pc_mask_img)
                # Use the first PC that isn't scattered noise


        _ = p.add_mesh(
            nifti_mask_to_mesh(mask_images[0], method="marching cubes", verbose=verbose),
            color=brain_color, opacity=0.1, show_edges=False,
        )


    # Add step 1 vascular cluster to the scene
    if step_1_nii_file.exists():
        if verbose:
            print(f"  Adding step 1 mask: {step_1_nii_file.name}")
        step_1_nii_img = nib.nifti1.Nifti1Image.from_filename(step_1_nii_file)
        _ = p.add_mesh(
            nifti_mask_to_mesh(step_1_nii_img, method="voxel", verbose=verbose),
            color=step_1_color, opacity=0.3, show_edges=False
        )
        _ = p.add_mesh(
            nifti_mask_to_mesh(step_1_nii_img, method="marching cubes", verbose=verbose),
            color=step_1_color, opacity=0.3, show_edges=False  # was "yellow"
        )
    # Add step 2 vascular cluster to the scene
    if step_2_nii_file.exists():
        if verbose:
            print(f"  Adding step 2 mask: {step_2_nii_file.name}")
        step_2_nii_img = nib.nifti1.Nifti1Image.from_filename(step_2_nii_file)
        _ = p.add_mesh(
            nifti_mask_to_mesh(step_2_nii_img, method="voxel", verbose=verbose),
            color=step_2_color, opacity=0.4, show_edges=False  # was "coral"
        )
        _ = p.add_mesh(
            nifti_mask_to_mesh(step_2_nii_img, method="marching cubes", verbose=verbose),
            color=step_2_color, opacity=0.6, show_edges=True
        )

    # Label L, R, S on the back wall of the scene.
    world_corners = get_nifti_world_corners(exemplar_img)
    min_x, max_x = np.min(world_corners[0, :]), np.max(world_corners[0, :])
    min_y, max_y = np.min(world_corners[1, :]), np.max(world_corners[1, :])
    min_z, max_z = np.min(world_corners[2, :]), np.max(world_corners[2, :])

    # around 70 pt for a 1072-sized window
    font_size = int(min(window_size[0], window_size[1]) / 20)
    # p.add_point_labels(
    #     [(min_x + (max_x - min_x) / 2.0, min_y, max_z,), ],
    #     ["S", ],
    #     point_size=0, font_size=font_size,
    #     show_points=False, fill_shape=False, shape_opacity=0.0,
    #     justification_horizontal="center", justification_vertical="top",
    # )
    p.add_point_labels(
        [(max_x, min_y, min_z + (max_z - min_z) / 2.0,), ],
        ["R", ],
        point_size=0, font_size=font_size,
        show_points=False, fill_shape=False, shape_opacity=0.0,
        justification_horizontal="left", justification_vertical="center",
    )
    p.add_point_labels(
        [(min_x, min_y, min_z + (max_z - min_z) / 2.0,), ],
        ["L", ],
        point_size=0, font_size=font_size,
        show_points=False, fill_shape=False, shape_opacity=0.0,
        justification_horizontal="right", justification_vertical="center",
    )

    # Add a clipping plane to the scene.
    axial_slices_to_clip = find_clipping_threshold(stare_output_path)
    if axial_slices_to_clip > 0:
        print(f"  Detected {axial_slices_to_clip} axial slices to clip. "
              f"Adding clipping plane.")
        clip_mesh = clip_box_mesh(exemplar_img, axial_slices_to_clip)
        _ = p.add_mesh(
            clip_mesh,
            color=clip_color,
            opacity=0.3,
            show_scalar_bar=False,
            show_edges=True,
        )
        z_mm_per_voxel = exemplar_img.affine[2, 2]
        p.add_point_labels(
            [(max_x, min_y, min_z + ((axial_slices_to_clip + 2) * z_mm_per_voxel), ), ],
            [f"clip = {axial_slices_to_clip}", ],
            point_size=0, font_size=font_size,
            show_points=False, fill_shape=False, shape_opacity=0.0,
            justification_horizontal="left", justification_vertical="bottom",
        )

    # Add a title to the scene.
    if title is not None:
        p.add_point_labels(
            [(max_x, min_y, max_z, ), ],
            [title, ],
            point_size=0, font_size=font_size,
            show_points=False, fill_shape=False, shape_opacity=0.0,
            justification_horizontal="left", justification_vertical="top",
        )

    # Adjust the camera and save the scene as a png.
    p.show()

    if verbose:
        print("Default Camera Position:")
        print(p.camera_position)
    # Let's move the camera 2.5 image-depths anterior from the brain,
    # along y, and offset it slightly left and above the scene
    center = ((min_x + (max_x - min_x) / 2.0),
              (min_y + (max_y - min_y) / 2.0),
              (min_z + (max_z - min_z) / 2.0))
    # The distance from the image center to the camera is the hypotenuse,
    # and everything else can be calculated from that with trigonometry.
    raw_hyp = (max_y - min_y) * 2.5
    # The elevation 'up in z' reduces the hypotenuse in the xy plane
    hyp = raw_hyp * math.cos(camera_up_angle * math.pi / 180.0)
    # Angle leftward from the center plane by camera_left_angle degrees.
    delta_x_in_xy = hyp * math.sin(camera_left_angle * math.pi / 180.0)
    camera_x = center[0] + delta_x_in_xy
    # Angle upward from the center plane by camera_up_angle degrees.
    delta_y_in_xy = hyp * math.cos(camera_left_angle * math.pi / 180.0)
    camera_y = center[1] + delta_y_in_xy
    delta_z = hyp * math.sin(camera_up_angle * math.pi / 180.0)
    camera_z = center[2] + delta_z
    # I stole the 2nd and 3rd rows from prior auto-iso camera positioning
    p.camera_position = [(camera_x, camera_y, camera_z),
                         p.camera_position.focal_point,
                         p.camera_position.viewup, ]

    if verbose:
        print("Modified Camera Position:")
        print("  img center", center)
        print("  hyp", hyp)
        print("  deltas", delta_x_in_xy, delta_y_in_xy, delta_z)
        print("  up angle", camera_up_angle)
        print("  left angle", camera_left_angle)
        print("  calculated position", camera_x, camera_y, camera_z)
        print("  p.position", p.camera_position.position)
        print("  p.focal_point", p.camera_position.focal_point)
        print("  p.viewup", p.camera_position.viewup)

    p.show(auto_close=False)

    if output_file is None:
        output_file = (stare_output_path / "figures" /
                f"sub-{stare_output_path.name}_3d_selected_clusters.png")

    if save_image:
        _ = p.screenshot(output_file)

    if save_movie:
        p.open_movie(str(output_file).replace("png", "mp4"), quality=8)
        # Swing from camera_left_angle through 0 to neg camera_left_angle.
        steps = int(camera_left_angle * 2.0)
        p.write_frame()
        for i in range(steps * 2):
            if i < steps:
                camera_left_angle -= 1.0
            elif i > steps:
                camera_left_angle += 1.0
            if verbose:
                print(f"  writing frame {i} / {steps * 2}: {camera_left_angle}")
            # Angle leftward from the center plane by camera_left_angle degrees.
            delta_x_in_xy = hyp * math.sin(camera_left_angle * math.pi / 180.0)
            camera_x = center[0] + delta_x_in_xy
            # Angle upward from the center plane by camera_up_angle degrees.
            delta_y_in_xy = hyp * math.cos(camera_left_angle * math.pi / 180.0)
            camera_y = center[1] + delta_y_in_xy
            p.camera_position = [(camera_x, camera_y, camera_z),
                                 p.camera_position.focal_point,
                                 p.camera_position.viewup, ]
            p.write_frame()

    p.close()
    p.deep_clean()

    return {
        'output_file': output_file if save_image else None,
        'plotter': p,
        'step_1_nii_file': step_1_nii_file,
        'step_2_nii_file': step_2_nii_file,
        'pc_file': pc_file,
        'clip': axial_slices_to_clip,
    }
