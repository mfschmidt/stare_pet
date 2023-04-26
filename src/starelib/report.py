from datetime import datetime
import logging
from pathlib import Path
import numpy as np


class Section:
    """ A class to track each section of a report.
    """

    def __init__(self, title, report, dt_format="%Y-%m-%d %I:%M"):
        self._start_datetime = datetime.now()
        self._end_datetime = None
        self._dt_format = dt_format
        self._report = report
        self.title = title
        self.items = []

    @property
    def start_time(self):
        return self._start_datetime

    def end(self):
        self._end_datetime = datetime.now()

    def duration(self):
        if self._start_datetime is None or self._end_datetime is None:
            return 0
        else:
            return self._end_datetime - self._start_datetime

    def add_line(self, line):
        self.items.append(f"<p>{line}</p>\n")
        self._report.logger.info(line)

    def add_figure(self, fig_path, caption):
        rel_path = str(Path(fig_path).relative_to(self._report.path.parent))
        img_tag = f"<img src=\"{rel_path}\" style=\"width: 90%\">"
        caption_tag = f"<figcaption>{caption}</figcaption>"
        img_html = "\n".join([
            f"<a href=\"{rel_path}\">",
            "<figure>", img_tag, caption_tag, "</figure>",
            "</a>",
        ])
        self.items.append(f"<p>{img_html}</p>")

    def add_table(self, dataframe):
        self.items.append(f"<p>{dataframe.to_html()}</p>")

    def html(self):
        if self._end_datetime is None:
            return f"<h2>{self.title} not yet complete.</h2>"
        else:
            relative_start = self._start_datetime - self._report.start_time
            duration = self._end_datetime - self._start_datetime
            return "\n".join([
                f"<h2>{self.title}</h2>",
                "<p>Started at {} ({} in) and took {}</p>".format(
                    self._start_datetime.strftime(self._dt_format),
                    str(relative_start).split('.', 2)[0],
                    str(duration).split('.', 2)[0],
                ),
            ] + self.items) + "\n"


class Report:
    """ A class to keep track of events in time and write a final report.
    """

    def __init__(self, title, file, logger=None, dt_format="%Y-%m-%d %I:%M:%S"):
        self._start_datetime = datetime.now()
        self._end_datetime = None
        self._dt_format = dt_format
        self.title = title
        self.path = Path(file).resolve()
        self.sections = []
        if logger is None:
            self.logger = logging.getLogger(title)
        else:
            self.logger = logger

    def __str__(self):
        return (f"A report, titled \"{self.title}\", "
                f"with {len(self.sections)} sections")

    @property
    def start_time(self):
        return self._start_datetime

    def end(self):
        self._end_datetime = datetime.now()

    def begin_section(self, title):
        new_section = Section(title, self, dt_format=self._dt_format)
        self.sections.append(new_section)
        self.logger.info(f"Started section '{title}'.")
        return new_section

    def write(self, file):
        total_duration = np.sum([_.duration() for _ in self.sections])
        with open(file, "w") as f:
            f.write(f"<h1>{self.title}</h1>\n")
            for sect in sorted(self.sections, key=lambda x: x.start_time):
                f.write(sect.html())
            f.write("<footer>"
                    f"completed {self._end_datetime.strftime(self._dt_format)},"
                    f" taking {str(total_duration).split('.', 2)[0]}"
                    "</footer>\n")
            f.write("\n")
