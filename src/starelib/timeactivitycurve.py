import numpy as np
from .util import characterize_mid_times


class TimeActivityCurve:
    """ Object representation of a Time Activity Curve
    """

    def __init__(self,
                 activity,
                 timepoints,
                 source,
                 missing_timepoints=None,
                 sd=None,
                 name=None, ):
        """ Centroid constructor """

        # Specified properties
        self.activity = np.asarray(activity)  # ndarray shaped ~ (25,)
        self.timepoints = np.asarray(timepoints)  # ndarray shaped ~ (1000000,)
        self.sd = None if sd is None else np.asarray(sd)  # ndarray ~ (25,)
        self.source = source  # where did this centroid come from
        self.name = name  # what I should call this TAC in a figure legend

        if missing_timepoints is None:
            self.missing_timepoints = None
        else:
            self.missing_timepoints = np.array(missing_timepoints)

        # Each centroid can also contain {label: value} features,
        # initialized as an empty dict.
        # An example would be
        # {"likely_noise": True, "likely_irreversible": False}
        # to label this centroid with those features.
        # To use this, first initialize the Centroid with __init__,
        # then add features to it like
        #   c.features["label"] = value
        self.features = {}

        # Calculated properties
        self.peak_value = np.max(self.activity)
        self.peak_index = np.argmax(self.activity)

        # Determine smoothness of time axis, and save value
        self.has_uniform_time_delta = False
        self.uniform_time_delta = None
        self._find_uniform_time_delta()

    def __str__(self):
        return f"{len(self.activity)} samples from {self.source}"

    def to_dict(self):
        return {
            "timepoints": self.timepoints,
            "activity": self.activity,
            "source": self.source,
            "name": self.name,
        }

    def _find_uniform_time_delta(self):
        # * Note that 'i' intentionally accesses the item just prior to
        #   the value retrieved in enumeration, which is offset from zero.
        # * Also note that the 10000 limits the precision of the deltas so
        #   float-encoding error is not responsible for mismatched time deltas.
        deltas = set([
            round((t - self.timepoints[i]) * 10000)
            for i, t in enumerate(self.timepoints[1:])
        ])
        if len(deltas) == 1:
            self.has_uniform_time_delta = True
            self.uniform_time_delta = deltas.pop() / 10000

    def pre_peak_activity(self):
        return self.activity[:self.peak_index]

    def post_peak_activity(self):
        return self.activity[self.peak_index:]

    def pre_peak_timepoints(self):
        return self.timepoints[:self.peak_index]

    def post_peak_timepoints(self):
        return self.timepoints[self.peak_index:]

    def pre_peak_sd(self):
        return None if self.sd is None else self.sd[:self.peak_index]

    def post_peak_sd(self):
        return None if self.sd is None else self.sd[self.peak_index:]

    def weights(self, method='sqrt'):
        """ Determine weights, based on timepoints, by method
        """

        # Calculate start, mid, and end timepoints for each sample
        time_df = characterize_mid_times(
            self.timepoints, self.missing_timepoints
        )
        # But drop those from missing_timepoints
        time_df = time_df[time_df['used']]

        # Then return weights, based on used samples, not skipped
        if method == 'duration':
            return np.real(time_df['duration'].values)
        else:
            # For 'sqrt' and anything else not covered
            return np.real(np.sqrt(time_df['duration'].values))

    def sigmas(self, method='sqrt'):
        """ Determine sigmas, based on weights
        """

        return np.real(1 / np.sqrt(self.weights(method=method)))

    def pre_peak_weights(self, method='sqrt'):
        return self.weights(method=method)[:self.peak_index]

    def post_peak_weights(self, method='sqrt'):
        return self.weights(method=method)[self.peak_index:]

    def pre_peak_sigmas(self, method='sqrt'):
        return self.sigmas(method=method)[:self.peak_index]

    def post_peak_sigmas(self, method='sqrt'):
        return self.sigmas(method=method)[self.peak_index:]
