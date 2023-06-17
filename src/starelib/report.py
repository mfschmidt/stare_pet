from datetime import datetime
import logging
from pathlib import Path
import numpy as np
import re
import platform


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

    def add_line(self, line, style=None, log=True):
        if style is None:
            self.items.append(f"<p>{line}</p>")
        else:
            self.items.append(f"<p style='{style}'>{line}</p>")
        if log:
            self._report.logger.info(line)

    def add_figure(self, fig_path, caption, style=None):
        rel_path = str(Path(fig_path).relative_to(self._report.path.parent))
        if style is None:
            width = "90%"
            style_str = ""
        else:
            width = "50%"
            style_str = f" style='{style}'"
        img_tag = f"<img src=\"{rel_path}\" style=\"width: {width}\">"
        caption_tag = f"<figcaption>{caption}</figcaption>"
        img_html = "\n".join([
            f"<a href=\"{rel_path}\">",
            f"<figure{style_str}>", img_tag, caption_tag, "</figure>",
            "</a>",
        ])
        self.items.append(f"{img_html}")

    def add_table(self, dataframe):
        self.items.append(f"{dataframe.to_html()}")

    def html(self):
        if self._end_datetime is None:
            return f"\n<div><h2>{self.title} not yet complete.</h2></div>\n"
        else:
            relative_start = self._start_datetime - self._report.start_time
            duration = self._end_datetime - self._start_datetime
            return "\n".join([
                f"\n<div>",
                "<h2>{self.title}</h2>",
                "<p class='{}'>Started at {} ({} in) and took {}</p>".format(
                    'subtext',
                    self._start_datetime.strftime(self._dt_format),
                    str(relative_start).split('.', 2)[0],
                    str(duration).split('.', 2)[0],
                ),
            ] + self.items) + "\n</div>\n"


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
        self.app_name = "N/A"
        self.app_version = "N/A"
        if logger is None:
            self.logger = logging.getLogger(title)
        else:
            self.logger = logger
        self.find_version()

    def __str__(self):
        return (f"A report, titled \"{self.title}\", "
                f"with {len(self.sections)} sections")

    def find_version(self):
        depth = 0
        here = Path(__file__).parent
        while depth < 5 and not Path(here / "setup.cfg").exists():
            depth += 1
            here = here.parent
        if (here / "setup.cfg").exists():
            with open(here / "setup.cfg", "r") as f:
                for line in f:
                    match_name = re.match(
                        r"name = ([A-Za-z_]*)", line
                    )
                    if match_name:
                        self.app_name = match_name.group(1)
                    match_version = re.match(
                        r"version = ([0-9]\.[0-9]\.[0-9])", line
                    )
                    if match_version:
                        self.app_version = match_version.group(1)

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

    @staticmethod
    def _get_styles():
        return "\n".join([
            "<style>",
            "table {padding: 4px; }",
            "td {padding: 4px; }",
            "h2 {clear: both}",
            "figure {",
            "  clear: both;",
            "  text-align: center; font-style: italic; font-size: smaller;",
            "  border: thin silver solid; margin: 0.5em; padding: 0.5em;",
            "}",
            ".subtext {color: black; font-size: small; font-style: italic; }",
            ".equation {text-align: center; }",
            ".left_fig {float: left; width: 50%}",
            ".right_fig {float: right; width: 50%}",
            ".clearfix::after {content: ''; clear: both; display: table;}",
            "</style>\n",
        ])

    @staticmethod
    def _get_js():
        return "\n".join([
            "<script src=\"{}\"></script>".format(
                "https://polyfill.io/v3/polyfill.min.js?features=es6"
            ),
            "<script id=\"{}\" async src=\"{}\"></script>".format(
                "MathJax - script",
                "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
            ),
            "",
        ])

    def write(self, file):
        total_duration = np.sum([_.duration() for _ in self.sections])
        with open(file, "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<head>\n")
            f.write(f"<title>{self.title}</title>\n")
            f.write(self._get_styles())
            f.write(self._get_js())
            f.write("</head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{self.title}</h1>\n")
            f.write(f"<p class='subtext'>running {self.app_name} version "
                    f"{self.app_version} on {platform.platform()}.</p>\n")
            for sect in sorted(self.sections, key=lambda x: x.start_time):
                f.write(sect.html())
            f.write("<footer class='subtext'><br />STARE "
                    f"completed {self._end_datetime.strftime(self._dt_format)},"
                    f" in {str(total_duration).split('.', 2)[0]}."
                    "</footer>\n")
            f.write("</body>\n")
            f.write("</html>\n")
