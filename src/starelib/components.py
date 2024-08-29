import datetime
import numpy as np
from sklearn.decomposition import PCA, FastICA
import nibabel as nib
from nilearn.image import mean_img

from .util import flatten_4d_to_2d, unflatten_2d_to_4d
from .plotting import plot_components, plot_mixing_matrix, plot_pca_variance


def run_pca(x, num_components, logger=None):
    """ Decompose x into num_components principal components.

        :param np.array x: Voxels rows by Time columns PET data
        :param int num_components: Number of principal components
        :param logger: If not None, one line will be logged to the logger.

        :returns tuple: returns transformer, signal matrix, and mixing matrix
    """

    # Perform the PCA
    start_time = datetime.datetime.now()
    _transformer = PCA(n_components=num_components)
    _S = _transformer.fit_transform(x.T)
    _A = _transformer.components_.T
    _pred_X = (np.dot(_S, _A.T) + _transformer.mean_).T
    end_time = datetime.datetime.now()
    if logger is not None:
        logger.info(f"  Fitting {num_components}-component PCA "
                    f"took {end_time - start_time}")

    return _transformer, _S, _A


def run_ica(x, num_components, logger=None):
    """ Do all necessary steps for complete ICA

        :param np.array x: Voxels rows by Time columns PET data
        :param int num_components: Number of principal components
        :param logger: If not None, one line will be logged to the logger.

        :returns tuple: returns transformer, signal matrix, and mixing matrix
    """

    # Perform the ICA
    start_time = datetime.datetime.now()
    _transformer = FastICA(n_components=num_components, whiten="unit-variance")
    # ICA needs to run on T x V, but our matrix is V x T, so transpose it
    _S = _transformer.fit_transform(x.T)
    _A = _transformer.mixing_
    # And this should closely recreate X, transposed back into V x T
    _pred_X = (np.dot(_S, _A.T) + _transformer.mean_).T
    end_time = datetime.datetime.now()
    if logger is not None:
        logger.info(f"  Fitting {num_components}-component ICA "
                    f"took {end_time - start_time}")

    return _transformer, _S, _A


def decompose_components(results, logger):
    """ Orchestrate the PCA and ICA routines

        :param results: the main results object
        :param logger: a logger for output handling
    """

    out_path = results.args.debug_path / "components"
    out_path.mkdir(exist_ok=True)
    _img_4d = results.input_4D
    _img_3d = mean_img(results.input_4D)
    max_t = _img_4d.shape[3] - 1

    # Do the decompositions
    _x = flatten_4d_to_2d(_img_4d.get_fdata())
    _pca_6, _pca_s_6, _pca_a_6 = run_pca(_x, 6)
    _ica_6, _ica_s_6, _ica_a_6 = run_ica(_x, 6)
    _ica_n, _ica_s_n, _ica_a_n = run_ica(_x, max_t)

    # Save the results as nifti files, and some stats as a csv
    with open(out_path / "component_stats.csv", "w") as f:
        f.write("algorithm,components,component,row,cols,mean,sd,min,max\n")
    for t, a, s, save_path, num_components in [
        (_pca_6, _pca_a_6, _pca_s_6, out_path / "pca_6.nii.gz", 6),
        (_ica_6, _ica_a_6, _ica_s_6, out_path / "ica_6.nii.gz", 6),
        (_ica_n, _ica_a_n, _ica_s_n, out_path / f"ica_{max_t}.nii.gz", max_t),
    ]:
        # Write stats to csv file
        with open(out_path / "component_stats.csv", "a") as f:
            for i in range(a.shape[1]):
                vals = a[:, i].ravel()
                f.write(",".join([
                    save_path.name[0:3],
                    str(num_components),
                    str(i),
                    str(a.shape[0]),
                    str(a.shape[1]),
                    f"{np.mean(vals):0.6f}",
                    f"{np.std(vals):0.6f}",
                    f"{np.min(vals):0.6f}",
                    f"{np.max(vals):0.6f}",
                ]) + "\n")

        # Write maps to nifti files
        _map_shape = (
            _img_4d.shape[0],
            _img_4d.shape[1],
            _img_4d.shape[2],
            num_components,
        )
        _comp_img = nib.Nifti1Image(
            unflatten_2d_to_4d(a, _map_shape), affine=_img_4d.affine
        )
        _comp_img.to_filename(save_path)
        logger.info(f"Saved component map as {save_path.name}")

        # Plot the TAC of each component
        comp_plot_filename = save_path.name.replace(".nii.gz", "_comps.png")
        _fig = plot_components(a, _x, results.mid_times,
                               title="ICA-component-masked PET TACs",
                               save_as=save_path.parent / comp_plot_filename)
        logger.info(f"  Plotted component TACs as {comp_plot_filename}")

        # Plot the TAC of each component
        mix_plot_filename = save_path.name.replace(".nii.gz", "_mix.png")
        _fig = plot_mixing_matrix(s, title="ICA Components",
                                  save_as=save_path.parent / mix_plot_filename)
        logger.info(f"  Plotted component mixing matrix as {mix_plot_filename}")

        if save_path.name.startswith("p"):
            # Plot the variance explained by each successive principal component
            var_plot_filename = save_path.name.replace(".nii.gz", "_var.png")
            _fig_v = plot_pca_variance(
                t, title="PCA Variance Explained",
                save_as=save_path.parent / var_plot_filename
            )
