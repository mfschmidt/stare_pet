import pathlib
import numpy as np


def kmeans(in_file):
    """ Perform k means clustering on in_file data.

        :param str in_file: Some file with some data, not yet used.

        :return numpy.array: array of zeros
    """

    data = np.zeros((4, 4))

    if pathlib.Path(in_file).exists():
        return data

    return None
