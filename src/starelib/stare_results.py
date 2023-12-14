from .results import Results


class StareResults(Results):
    """ A Results container, specific to the STARE pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # This object will need member variables to store things that don't
        # yet exist. Make room here.

        # Data loaded from disk (some modified slightly)
        self.tacs = None  # previously extracted
        self.original_tacs = None
        self.source_tacs_path = None
        self.mid_times = None
        self.original_mid_times = None
        self.source_mid_times_path = None
        self.ignored_mid_times = None
        self.plasma_tac = None
        self.source_plasma_tac_file = None
        self.input_4D = None
        self.cropped_4D = None
        self.volume_files = []
        self.region_weights = None

        # Data from two-step clustering
        self.cluster_centroids = {1: [], 2: [], }  # contains best TAC so far
        self.cluster_model_fits = {1: [], 2: [], }
        self.best_vascular_mask_path = {}

        # Data from partial volume correction
        self.pvc_mean_vascular_tac = None  # pvc-corrected best cluster TAC

        # Data from stacked exponential model fitting
        self.fitting_successes = []
        self.fitted_tac = None
        self.fitted_hires_tac = None

        # Data from vascular correction of original regional TACs
        self.corrected_tacs = None

        # Data from bootstrapping randomized TACs
        self.kde_lower_bounds = None
        self.kde_upper_bounds = None
        self.bootstrap_curves = []
        self.bootstrap_rate_constants = []
        self.bootstrap_kis = None
        self.bootstrap_ki_fwhm = None

        # Data from simulated annealing
        self.annealer_bounds = None
        self.annealer_results = {}
        self.final_rate_df = None

    def __str__(self):
        return "{} {} results from {}".format(
            self._name, self.report.app_version, self.start_time_str
        )

    @property
    def regions(self):
        if self.tacs is None:
            return None
        else:
            return self.tacs.columns

    @property
    def input_volumes(self):
        return [self.input_4D.slice[:, :, :, t]
                for t in self.input_4D.shape[3]]

    @property
    def cropped_volumes(self):
        return [self.cropped_4D.slice[:, :, :, t]
                for t in self.cropped_4D.shape[3]]

    def best_centroid(self, step=2):
        for centroid in self.cluster_centroids[step]:
            if centroid.best_overall:
                return centroid
        return None
