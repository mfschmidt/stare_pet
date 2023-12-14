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
                 source="",
                 blob_count=0,
                 voxels_per_blob=0,
                 name=None,
                 best_in_k=False,
                 best_overall=False, ):
        """ Centroid constructor """

        # Specified properties
        super().__init__(activity, timepoints, "k-means", name=name)
        self.label = label  # int, one of k clusters
        self.k = k  # int, how many clusters
        self.labels = labels  # ndarray shaped like (1000000,)
        self.best_in_k = best_in_k
        self.best_overall = best_overall
        self.source = source
        self.blob_count = blob_count
        self.voxels_per_blob = voxels_per_blob

    def __str__(self):
        return "Centroid {} of k={}{}{}".format(
            self.label, self.k,
            f" (best in k={self.k})" if self.best_in_k else "",
            f" (best overall)" if self.best_overall else "",
        )

    def to_dict(self):
        d = super().to_dict()
        d["label"] = self.label
        d["k"] = self.k
        d["best_in_k"] = self.best_in_k
        d["best_overall"] = self.best_overall
        d["voxels_per_blob"] = self.voxels_per_blob
        d["blob_count"] = self.blob_count
        d["source"] = self.source
        return d

    def description(self):
        if self.best_overall:
            asterisks = " (**)"
        elif self.best_in_k:
            asterisks = " (*)"
        else:
            asterisks = ""
        d = "{}: peak={:0.4f} @ t={}/{}, {:d} blobs w/~{} voxels each{}".format(
            self.name,
            self.peak_value,
            int(self.peak_index + 1),
            len(self.timepoints),
            int(self.blob_count),
            int(self.voxels_per_blob),
            asterisks
        )
        return d
