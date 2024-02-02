import numpy as np

from .timeactivitycurve import TimeActivityCurve
from .util import reshape_labels_to_3d, get_cluster_blobs


class Centroid(TimeActivityCurve):
    """ Object representation of a centroid from k-means clustering
    """

    def __init__(
            self,
            activity,
            timepoints,
            label,  # should be non-zero as zero indicates background
            k,
            source="",
            labels=None,
            original_shape=None,
            blob_count=0,
            voxels_per_blob=0,
            name=None,
            best_in_k=False,
            best_overall=False,
    ):
        """ Centroid constructor """

        # Specified properties
        super().__init__(activity, timepoints, "k-means", name=name)
        self.label = label  # int, one of k clusters
        self.k = k  # int, how many clusters
        self.labels = labels  # ndarray shaped like (1000000,)
        self.original_shape = original_shape
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

    def to_3d(self):
        if any([self.labels is None, self.original_shape is None, ]):
            return None
        return reshape_labels_to_3d(self.labels, self.original_shape)

    def update_spatial_clusters(
            self, labels=None, force_update=False, verbose=0, logger=None
    ):
        if labels is None:
            labels = self.labels
        message_list = []
        if self.blob_count == 0 or force_update:
            blob_df, blob_ids, voxel_counts = get_cluster_blobs(
                reshape_labels_to_3d(labels, self.original_shape),
                label=self.label, verbose=verbose, messages=message_list,
            )
            self.blob_count = len(blob_ids)
            self.voxels_per_blob = np.mean(voxel_counts)
        if logger is not None:
            for message in message_list:
                logger.debug(message)

    def description(self):
        # Determine whether centroid's rank merits asterisks
        if self.best_overall:
            asterisks = " (**)"
        elif self.best_in_k:
            asterisks = " (*)"
        else:
            asterisks = ""

        # Determine whether spatial analysis has been done
        if self.blob_count == 0:
            blob_str = "no sparsity data"
        else:
            blob_str = "{:d} blobs w/~{} voxels each".format(
                int(self.blob_count), int(self.voxels_per_blob)
            )

        # Return description of centroid
        d = "{}: peak={:0.4f} @ t={}/{}, {}{}".format(
            self.name,
            self.peak_value,
            int(self.peak_index + 1),
            len(self.timepoints),
            blob_str,
            asterisks
        )
        return d
