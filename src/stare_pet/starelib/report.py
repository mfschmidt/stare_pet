from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np
import platform
import multiprocessing
import os
from importlib.metadata import version, PackageNotFoundError

# This can be overridden when someone creates a Report or Section object,
# but all datetimes will be represented as text with this format otherwise.
# '1999-01-30 23:59' - 24-hour hours (%H rather than %I) allow later html mining
default_dt_fmt = "%Y-%m-%d %H:%M"


class Section:
    """ A class to track each section of a report.
    """

    def __init__(self, title, report, dt_format=default_dt_fmt):
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
            return timedelta(0)
        else:
            return self._end_datetime - self._start_datetime

    def add_line(self, line, css_class=None, log=True):
        if log:
            loggable_lines = (line.replace("<br />", "\n")
                              .replace("<table>", "")
                              .replace("</table>", "")
                              .replace("<thead>", "")
                              .replace("</thead>", "")
                              .replace("<tbody>", "")
                              .replace("</tbody>", "")
                              .replace("<tfoot>", "")
                              .replace("</tfoot>", "")
                              .replace("<th>", " ")
                              .replace("</th>", ",")
                              .replace("<tr>", "")
                              .replace("</tr>", "\n")
                              .replace("<td>", " ")
                              .replace("</td>", ",")).split("\n")
            for loggable_line in loggable_lines:
                if len(loggable_line.strip()) > 0:
                    self._report.logger.info(loggable_line.strip())
        if css_class is None:
            self.items.append(f"<p>{line}</p>")
        else:
            self.items.append(f"<p class='{css_class}'>{line}</p>")

    def add_link(self, url, text=None, css_class=None):
        css_class_str = ""
        if css_class is not None:
            css_class_str = " class='{css_class}'"
        if text is None:
            text = url
        self.items.append(f"<p><a{css_class_str} href=\"{url}\">{text}</a></p>")

    def add_figure(self, fig_path, caption, css_class=None):
        rel_path = str(Path(fig_path).relative_to(self._report.path.parent))
        if css_class is None:
            css_class_str = ""
        else:
            css_class_str = f" class='{css_class}'"
        img_tag = f"<img src=\"{rel_path}\" style=\"width: 90%\">"
        caption_tag = f"<figcaption>{caption}</figcaption>"
        img_html = "\n".join([
            f"<a href=\"{rel_path}\">",
            f"<figure{css_class_str}>", img_tag, caption_tag, "</figure>",
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
                f"<h2>{self.title}</h2>",
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

    def __init__(self, title, file, logger=None, dt_format=default_dt_fmt):
        self._start_datetime = datetime.now()
        self._end_datetime = None
        self._dt_format = dt_format
        self.title = title
        self.path = Path(file).resolve()
        self.sections = []
        self.app_name = "stare_pet"
        self.app_version = self.find_version()
        if logger is None:
            self.logger = logging.getLogger(title)
        else:
            self.logger = logger


    def __str__(self):
        return (f"A report, titled \"{self.title}\", "
                f"with {len(self.sections)} sections")

    def find_version(self):
        try:
            return version(self.app_name)
        except PackageNotFoundError as e:
            return "N/A"

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
            "table {border-color: gray; border-spacing: 0; border: 1px solid; }",
            "thead,tfoot { background-color: gainsboro; }",
            "thead th {"
            "  font-weight: bold; "
            "}",
            "tfoot th {"
            "  font-weight: lighter; font-style: italic; text-align: right; "
            "}",
            "th,td {padding: 4px; }",
            "h2 {clear: both; }",
            "figure {",
            "  clear: both;",
            "  text-align: center; font-style: italic; font-size: smaller;",
            "  border: thin silver solid; margin: 0.5em; padding: 0.5em;",
            "}",
            ".subtext {color: black; font-size: small; font-style: italic; }",
            ".warning {color: red; font-size: small; font-style: bold; }",
            ".equation {text-align: center; }",
            ".left_fig {float: left; width: 49%; }",
            ".right_fig {float: right; width: 49%; }",
            ".clearfix::after {content: ''; clear: both; display: table; }",
            "</style>\n",
        ])

    @staticmethod
    def _get_js():
        v3_url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        v4_url = "https://jsdelivr.net"
        return "\n".join([
            "<script id=\"{}\" async src=\"{}\"></script>".format(
                "MathJax-script", v4_url,
            ),
            "",
        ])

    def write(self, file):
        if self._end_datetime is not None:
            ts_str = self._end_datetime.strftime(self._dt_format)
            fin_str = "completed"
            verb_str = "in"
        else:
            ts_str = datetime.now().strftime(self._dt_format)
            fin_str = "writing partial report"
            verb_str = "so far took"
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
            # First, try using hostname from environment, which is set if
            # we're in a randomly named docker container. Otherwise,
            # ask the platform for its hostname
            f.write(f"<p class='subtext'>Running {self.app_name} version "
                    f"{self.app_version} on host "
                    f"{os.environ.get('HOST_NAME', platform.node())} "
                    f"({platform.platform()} with "
                    f"{multiprocessing.cpu_count()} CPUs).</p>\n")
            for sect in sorted(self.sections, key=lambda x: x.start_time):
                f.write(sect.html())
            f.write("<footer class='subtext'><br />STARE "
                    f"{fin_str} {ts_str}, {verb_str} "
                    f"{str(total_duration).split('.', 2)[0]}."
                    "</footer>\n")
            f.write("</body>\n")
            f.write("</html>\n")
