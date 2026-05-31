# Define error codes for consistent usage
class FailureCodes:
    def __init__(self):
        self.CONVERGED_TO_NAN = 2
        self.CONVERGED_TO_INF = 3
        self.ERROR_TOO_HIGH = 4
        self.RUNTIME_ERROR = 11
        self.RUNTIME_WARNING = 12
        self.OPTIMIZE_WARNING = 14
        self.TAC_HAS_NEGATIVES = 15
        self.TAC_GENERIC = 21
        self.TAC_VALUE_ERROR = 31
        self.TAC_ZERO_PARAM = 41
        self.TAC_ONE_PARAM = 42

    def description(self, code):
        return {
            self.CONVERGED_TO_NAN: "3-exp fit converged, but converged to NaN.",
            self.CONVERGED_TO_INF: "3-exp fit converged, but converged to Inf.",
            self.ERROR_TOO_HIGH: "3-exp fit converged, but error wts too high.",
            self.RUNTIME_ERROR: "3-exp fitting raised RuntimeError exception.",
            self.RUNTIME_WARNING: "3-exp fitting raised RuntimeWarning.",
            self.OPTIMIZE_WARNING: "3-exp fitting raised OptimizeWarning.",
            self.TAC_HAS_NEGATIVES: "Boot curve fit, but TAC has negatives.",
            self.TAC_GENERIC: "Fitting TAC with LS failed.",
            self.TAC_VALUE_ERROR: "Fitting TAC with LS raised ValueError.",
            self.TAC_ZERO_PARAM: "Fit TAC with LS, but got a 0.0 in constants.",
            self.TAC_ONE_PARAM: "Fit TAC with LS, but got a 1.0 in constants.",
        }[code]


failure_codes = FailureCodes()


class Failure:
    def __init__(self, code, params=None):
        self.code = int(code)
        self.params = params

    def as_dict(self, include_params=True):
        d = {
            'code': self.code,
            'description': self.description,
        }
        if include_params and self.params is not None:
            d['params'] = self.params
        return d

    def as_csv_line(self):
        return ",".join([str(self.code), ] + [f"{p:0.5f}" for p in self.params])

    @property
    def description(self):
        return failure_codes.description(self.code)


class FailureCollection:
    """ Store a dict of failure types, with a count for each. """

    def __init__(self):
        self._failures = dict()

    def __len__(self):
        return sum([data['count'] for code, data in self._failures.items()])

    def append(self, new_failure):
        """ Count new_failure in this collection.

            Completely ignore the parameters that may be stored in each
            'new_failure'. We store them into each Failure object initially,
            and we may come back in the future and store them in the collection
            if we choose to further refine that selection, but for now, it takes
            a ridiculous amount of RAM for hard-to-fit curves,
            and we've never done anything with them, so let them vanish.
        """
        if new_failure.code in self._failures.keys():
            self._failures[new_failure.code]['count'] += 1
        else:
            self._failures[new_failure.code] = {
                'description': new_failure.description,
                'count': 1,
            }

    def join(self, new_failures):
        """ Combine new FailureCollection object with self. """
        for code, data in new_failures.items():
            if code in self._failures.keys():
                self._failures[code]['count'] += data['count']
            else:
                self._failures[code] = data

    def count(self, code):
        if code in self._failures.keys():
            return self._failures[code]['count']
        else:
            return 0

    def description(self, code):
        if code in self._failures.keys():
            return self._failures[code]['description']
        else:
            return None

    def items(self):
        return self._failures.items()
