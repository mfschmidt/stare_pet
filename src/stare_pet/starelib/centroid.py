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
            **kwargs,
    ):
        """ Centroid constructor """

        # Specified properties
        super().__init__(
            activity,
            timepoints,
            kwargs.get("source", ""),
            missing_timepoints=kwargs.get("missing_timepoints", None),
            sd=kwargs.get("sd", None),
            name=kwargs.get("name", None)
        )
        self.label = label  # int, one of k clusters
        self.k = k  # int, how many clusters
        self.labels = kwargs.get("labels", None)  # ndarray shaped like (1000000,)
        self.original_shape = kwargs.get("original_shape", None)
        self.original_affine = kwargs.get("original_affine", None)
        self.voxels_in_img = kwargs.get("voxels_in_img", 0)
        self.best_in_k = kwargs.get("best_in_k", False)
        self.best_overall = kwargs.get("best_overall", False)
        self.source = kwargs.get("source", None)
        self.voxel_count = kwargs.get("voxel_count", 0)
        self.blob_count = kwargs.get("blob_count", 0)
        self.voxels_per_blob = kwargs.get("voxels_per_blob", 0.0)
        self.sparsity = kwargs.get("sparsity", 0)
        self.blob_data = kwargs.get("blob_data", None)

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
        d["voxel_count"] = self.voxel_count
        d["voxels_in_img"] = self.voxels_in_img
        d["blob_count"] = self.blob_count
        d["voxels_per_blob"] = self.voxels_per_blob
        d["sparsity"] = self.sparsity
        return d

    def labels_in_3d(self):
        if any([self.labels is None, self.original_shape is None, ]):
            return None
        return reshape_labels_to_3d(self.labels, self.original_shape)

    def mask_in_3d(self, sparsity_threshold=0, logger=None):
        if any([self.labels is None, self.original_shape is None, ]):
            return None
        if sparsity_threshold == 0:
            return reshape_labels_to_3d(
                np.array(self.labels == self.label).astype(np.uint8),
                self.original_shape
            )
        else:
            self.update_spatial_clusters()
            real_threshold = 1.0 - (sparsity_threshold / 100.0)
            counts = (self.blob_data.groupby("blob")['blob']
                      .agg('count').sort_values(ascending=False))
            blobs_consumed, voxels_consumed = 0, 0
            ratio = 0.0
            keepers = set()
            for idx, voxels in counts.items():
                ratio = voxels_consumed / self.voxel_count
                if ratio <= real_threshold:
                    keepers.add(idx)
                else:
                    break
                blobs_consumed += 1
                voxels_consumed += voxels

            self.features['reduced_ratio'] = 1.0 - ratio
            if logger is not None:
                plural = "s" if blobs_consumed > 1 else ""
                logger.debug(f"Reduced cluster mask from {self.voxel_count:,}"
                             f" voxels in {self.blob_count} blobs, to "
                             f"{voxels_consumed:,} voxels in "
                             f"{blobs_consumed} blob{plural} ({ratio:0.1%}).")

            keeper_filter = self.blob_data['blob'].isin(keepers)
            df_in = self.blob_data.loc[keeper_filter, :]
            new_mask = np.zeros(self.original_shape[:3], dtype=np.uint8)

            # df_out = self.blob_data.loc[~keeper_filter, :]
            for idx, row in df_in.iterrows():
                new_mask[row['x'], row['y'], row['z']] = 1
            return new_mask

    def update_spatial_clusters(
            self, labels=None, force_update=False,
            message_list=None, verbose=0, logger=None
    ):
        if labels is None:
            labels = self.labels
        if message_list is None:
            message_list = list()
        if self.blob_count == 0 or force_update:
            self.voxel_count = np.sum(self.labels == self.label)
            blob_df, blob_ids, voxel_counts = get_cluster_blobs(
                reshape_labels_to_3d(labels, self.original_shape),
                label=self.label, verbose=verbose, messages=message_list,
            )
            self.blob_data = blob_df
            self.blob_count = len(blob_ids)
            if self.blob_count > 0:
                self.voxels_per_blob = np.mean(voxel_counts)

                # Sparsity is the smallest number of blobs to hold 95% of the voxels.
                sparsity_threshold = 0.95
                counts = (self.blob_data.groupby("blob")['blob']
                          .agg('count').sort_values(ascending=False))
                blobs_consumed, voxels_consumed = 0, 0
                for idx, voxels in counts.items():
                    ratio = voxels_consumed / self.voxel_count
                    if ratio > sparsity_threshold:
                        break
                    blobs_consumed += 1
                    voxels_consumed += voxels
                self.sparsity = blobs_consumed
            else:
                self.voxels_per_blob = 0.0
                self.sparsity = 0
        else:
            message_list.append(f"Centroid {self.label}/{self.k} did not update "
                                f"because it already has {self.blob_count} blobs.")
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
            blob_str = "{:d} blobs w/~{:0.1f} voxels each".format(
                int(self.blob_count), float(self.voxels_per_blob)
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
