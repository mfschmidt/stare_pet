import logging
import sys
from pathlib import Path
from datetime import datetime
import time
import pickle

from .report import Report


class Results:
    """ A base class for storing results throughout a pipeline run.
    """

    def __init__(self, name, title, parsed_args, dt_format="%Y-%m-%d_%H-%M"):
        self._dt_format = dt_format
        self._name = name
        self._title = title
        self._args = parsed_args
        self._start_datetime = datetime.now()
        self._end_datetime = None

        self._logger = logging.getLogger(self._name)
        self.setup_logger()

        self.report = Report(
            self._title,
            parsed_args.output_path / f"{parsed_args.subject}_stare_report.html",
            self._logger,
        )

    @property
    def name(self):
        """ The name of the pipeline """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def args(self):
        """ The arguments passed to the pipeline """
        return self._args

    @property
    def logger(self):
        return self._logger

    @property
    def start_time(self):
        return self._start_datetime

    @property
    def start_time_str(self):
        return self._start_datetime.strftime(self._dt_format)

    @property
    def end_time(self):
        return self._end_datetime

    @property
    def end_time_str(self):
        if self._end_datetime is None:
            return "active"
        else:
            return self._end_datetime.strftime(self._dt_format)

    @property
    def datetime_format(self):
        return self._dt_format

    @datetime_format.setter
    def datetime_format(self, value):
        self._dt_format = value

    def setup_logger(self):
        """ Create and configure logger with handlers. """

        # Set up a logger to handle output, and attach two handlers
        # The logger emits everything (DEBUG) and each handler can
        # filter it as arguments specify.
        self._logger.setLevel(logging.DEBUG)

        # Create a handler to write out to the terminal
        # This handler adapts to the verbosity in the command line.
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s : %(message)s",
            datefmt="%H:%M:%S %Z",
        ))
        terminal_handler.converter = time.localtime
        if (self._args.verbose > 1) or self._args.debug:
            terminal_handler.setLevel(logging.DEBUG)
        elif self._args.verbose > 0:
            terminal_handler.setLevel(logging.INFO)
        else:
            terminal_handler.setLevel(logging.WARNING)
        self._logger.addHandler(terminal_handler)

        # Create a handler to write detailed information to a log file.
        # This handler always captures all info, debug and higher
        # Windows cannot handle colons in filenames
        file_handler = logging.FileHandler(
            Path(self._args.output_path) /
            f"stare_pet_{self._start_datetime.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s : %(levelname)s : %(message)s",
            datefmt=self._dt_format,
        ))
        file_handler.converter = time.localtime
        file_handler.setLevel(logging.INFO)
        if (self._args.verbose > 1) or self._args.debug:
            file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)

        self._logger.info(f"Begin {self._name} at {self.start_time_str}.")

    def end(self):
        self._end_datetime = datetime.now()
        self.report.end()
        self._logger.info(f"{str(self.elapsed())} elapsed.")
        self._logger.info(f"End {self._name} at {self.end_time_str}.")
        for handler in self._logger.handlers:
            handler.flush()

    def elapsed(self):
        return datetime.now() - self._start_datetime

    def write_report(self):
        self.report.write(
            self._args.output_path / f"sub-{self._args.subject}.html"
        )
        if self._args.debug:
            with open(
                self._args.debug_path / f"sub-{self._args.subject}_results.pickle",
                "wb"
            ) as f:
                pickle.dump(self, f)

    def save(self):
        with open(
            self._args.output_path / f"sub-{self._args.subject}_results.pickle",
            "wb"
        ) as f:
            pickle.dump(self, f)
