import numpy as np


class TimeActivityCurve:
    """ Object representation of a Time Activity Curve
    """

    def __init__(self,
                 activity,
                 timepoints,
                 source, ):
        """ Centroid constructor """

        # Specified properties
        self.activity = activity  # ndarray shaped like (25,)
        self.timepoints = timepoints  # ndarray shaped like (1000000,)
        self.source = source  # where did this centroid come from

        # Calculated properties
        self.peak_value = np.max(self.activity)
        self.peak_index = np.argmax(self.activity)

    def __str__(self):
        return f"{len(self.activity)} samples from {self.source}"
