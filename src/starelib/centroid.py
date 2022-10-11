from .timeactivitycurve import TimeActivityCurve


class Centroid(TimeActivityCurve):
    """ Object representation of a centroid from k-means clustering
    """

    def __init__(self,
                 activity,
                 timepoints,
                 label,  # should be non-zero as zero indicates background
                 k,
                 labels,
                 name=None,
                 vascular=False,
                 best_in_k=False,
                 best_overall=False, ):
        """ Centroid constructor """

        # Specified properties
        super().__init__(activity, timepoints, "k-means", name=name)
        self.label = label  # int, one of k clusters
        self.k = k  # int, how many clusters
        self.labels = labels  # ndarray shaped like (1000000,)
        self.vascular = vascular
        self.best_in_k = best_in_k
        self.best_overall = best_overall

        # Each centroid can also contain {label: value} features,
        # initialized as an empty dict.
        # An example would be
        # {"likely_noise": True, "likely_irreversible": False}
        # to label this centroid with those features.
        self.features = {}
