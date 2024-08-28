import datetime
import numpy as np
from sklearn.decomposition import PCA, FastICA


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

