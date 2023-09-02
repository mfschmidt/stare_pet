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
        # At least one set of FDG timepoints has noise beyond the 4th decimal
        # point that causes an extra timepoint to be added to the hi-res TAC.
        # Rounding to the nearest millisecond prevents this problem.
        self.timepoints = np.asarray([
            round(t, 3) for t in timepoints
        ])  # ndarray shaped ~ (1000000,)
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

    def get_uniform_time_curve(self, spacing=0.10, interpolation='linear'):
        """ Evenly space timepoints from uneven sampling

            Betsy's matlab version only stored a peak_index value
            in the fit vascular tac, not the pvc-corrected vascular tac. So we
            needed two tacs to piece together a higher resolution interpolation.
            Because we use a TimeActivityCurve object where every TAC has both
            activity and timepoints and the ability to calculate its own peak,
            only one TAC is necessary here.
        """

        # Interpolate a higher-resolution x-axis time data from TAC data
        # In matlab test, results in a 551-length vector from 0.0 to 55.0
        # from 11 pre-peak 0.0 to 1.0 and 540 post-peak 1.1 to 55.0
        pre_peak_time_uniform = np.arange(
            start=0.0,
            stop=round(self.post_peak_timepoints()[0], 1),
            step=spacing,
        )
        post_peak_time_uniform = np.arange(
            start=round(self.post_peak_timepoints()[0], 1),
            stop=self.timepoints[-1] + spacing,
            step=spacing,
        )
        complete_time_uniform = np.concatenate([
            pre_peak_time_uniform, post_peak_time_uniform,
        ])

        # Interpolate higher-resolution y-axis activity from TAC data
        # Interpolate values from sparse to hi-res, then clip low end to 0.0.
        # DIFF:
        #   Numpy's interpolator flattens at the end; matlab's shoots higher.
        # NOTE:
        #   pvc_mean_tac is the best estimate of pre-peak activity so far.
        #   vascular_tac has been interpolated to high-res and back.
        #   xp & fp must have same # of samples, so align both to time_curve.
        if interpolation == 'linear':
            pre_peak_vasctac_uniform = np.interp(
                pre_peak_time_uniform,
                self.pre_peak_timepoints(),
                self.activity[:self.peak_index],
            )
        else:
            pre_peak_vasctac_uniform = np.interp(
                pre_peak_time_uniform,
                self.pre_peak_timepoints(),
                self.activity[:self.peak_index],
            )
        # Clean up any errant points
        pre_peak_vasctac_uniform[(
            (pre_peak_vasctac_uniform < 0) | np.isnan(pre_peak_vasctac_uniform)
        )] = 0.0
        num_post_peak = len(complete_time_uniform) - len(pre_peak_vasctac_uniform)
        # We model only the pre-peak data, leave post-peak for later
        post_peak_vasctac_uniform = np.array([np.nan, ] * num_post_peak)
        boot_curve_activity_uniform = np.concatenate([
            pre_peak_vasctac_uniform, post_peak_vasctac_uniform,
        ])

        # Ensure the fit is uniformly sampled.
        deltas = []
        last_t = 0.0
        for j, t in enumerate(complete_time_uniform):
            if j > 0:
                deltas.append(t - last_t)
            last_t = t
        if (np.max(np.array(deltas)) - np.min(np.array(deltas))) > 0.0000001:
            raise ValueError(
                "Impossibly, the predetermined times are non-uniform!"
            )

        return TimeActivityCurve(
            activity=np.array(boot_curve_activity_uniform),
            timepoints=np.array(complete_time_uniform),
            source="uniform_interpolator",
            name="uniform_time_only",
        )

