import numpy as np


class TimeActivityCurve:
    """ Object representation of a Time Activity Curve
    """

    def __init__(self,
                 activity,
                 timepoints,
                 source,
                 name=None, ):
        """ Centroid constructor """

        # Specified properties
        self.activity = activity  # ndarray shaped like (25,)
        self.timepoints = timepoints  # ndarray shaped like (1000000,)
        self.source = source  # where did this centroid come from
        self.name = name  # what I should call this TAC in a figure legend

        # Calculated properties
        self.peak_value = np.max(self.activity)
        self.peak_index = np.argmax(self.activity)

    def __str__(self):
        return f"{len(self.activity)} samples from {self.source}"

    def to_dict(self):
        return {
            "timepoints": self.timepoints,
            "activity": self.activity,
            "source": self.source,
            "name": self.name,
        }
