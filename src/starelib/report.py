from datetime import datetime


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

    def add_line(self, line):
        self.items.append(f"<p>{line}</p>\n")

    def add_figure(self, fig_path, caption):
        img_tag = f"<img src=\"{fig_path}\" style=\"width: 90%\">"
        caption_tag = f"<figcaption>{caption}</figcaption>"
        img_html = "\n".join([
            f"<a href=\"{fig_path}\">",
            "<figure>", img_tag, caption_tag, "</figure>",
            "</a>",
        ])
        self.items.append(f"<p>{img_html}</p>")

    def html(self):
        if self._end_datetime is None:
            return f"<h2>{self.title} not yet complete.</h2>"
        else:
            return "\n".join([
                f"<h2>{self.title}</h2>",
                "<p>Started at {} ({} in) and took {}</p>".format(
                    self._start_datetime.strftime(self._dt_format),
                    self._start_datetime - self._report.start_time,
                    self._end_datetime - self._start_datetime,
                ),
            ] + self.items) + "\n"


class Report:
    """ A class to keep track of events in time and write a final report.
    """

    def __init__(self, title, dt_format="%Y-%m-%d_%I-%M"):
        self._start_datetime = datetime.now()
        self._end_datetime = None
        self._dt_format = dt_format
        self.title = title
        self.sections = []

    def __str__(self):
        return "A report"

    @property
    def start_time(self):
        return self._start_datetime

    def begin_section(self, title):
        new_section = Section(title, self, dt_format=self._dt_format)
        self.sections.append(new_section)
        return new_section

    def write(self, file):
        with open(file, "w") as f:
            for sect in sorted(self.sections, key=lambda x: x.start_time):
                f.write(sect.html())
