"""
A terminal FITS viewer with interactive tables.

Author: Giuseppe Dilillo
Date:   August 2024
"""

from asyncio import sleep, to_thread
from contextlib import asynccontextmanager
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Callable

from astropy.io import fits
import click
from textual import events
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.app import DEFAULT_COLORS
from textual.containers import Horizontal
from textual.design import ColorSystem
from textual.message import Message
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import Tree
from textual.worker import get_current_worker

from misfits.data import _validate_fits
from misfits.data import filter_array
from misfits.data import get_fits_content
from misfits.screens import EscapableFileExplorerScreen, HeaderEntry
from misfits.screens import FileExplorerScreen
from misfits.screens import InfoScreen
from misfits.screens import LogScreen


DEFAULT_COLORS["dark"] = ColorSystem(
    primary="#03A062",  # matrix green
    secondary="#03A062",
    warning="#03A062",
    error="#ff0000",
    success="#00ff00",
    accent="#00ff00",
    dark=True,
)


class FitsTable(DataTable):
    """Display numpy structured array as a table."""
    def update_arr(self, arr: fits.fitsrec, cols):
        """Update with a new table."""
        self.clear(columns=True)
        self.add_columns(*cols)
        self.add_rows([tuple(row[c] for c in cols) for row in arr])

    def on_mount(self):
        self.border_title = "Table"
        self.cursor_type = "cell"


class LabelInput(Input):
    """A prompt widget with an input and a label"""

    def __init__(self, label_text: str):
        self.label_text = label_text
        super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"[dim italic]{self.label_text}: ")
            yield Input()

    def set_input_value(self, value: str):
        self.query_one(Input).value = value


class InputFilter(Static):
    """A prompt widget for filtering a table"""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("[dim italic] query: ")
            yield Input(placeholder=f"COL1 > 42 & COL2 == 3")

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    """Hosts widgets for navigating a data table and filter it.
    Table entries are shown in pages for responsiveness."""

    BINDINGS = [
        ("shift+left", "back_page()", "Back"),
        ("shift+right", "next_page()", "Next"),
        ("ctrl+a", "first_page()", "First"),
        ("ctrl+e", "last_page()", "Last"),
    ]

    def __init__(self, arr: fits.FITS_rec, cols: list[str], page_len: int = 50, hide_filter: bool = False):
        """
        :param arr: The dataframe to show
        :param cols: Columns to show in table
        :param page_len: How many dataframe rows are shown for each page.
        :param hide_filter: Wether if to show the table filter or none. We do not show
        filter for tables which would require huge loading time, such as tables with
        variable length array columns.
        """
        super().__init__()
        self._arr = arr
        self.arr = arr
        self.cols = cols
        self.disable_filter = hide_filter
        self.mask = None
        self.page_len = page_len
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(arr) / page_len), 1)

    def compose(self) -> ComposeResult:
        yield FitsTable()
        if not self.disable_filter and len(self.arr) > 1:
            yield InputFilter()

    def on_mount(self):
        self.border_title = "Table"
        self.update_page_display()

    # this is an unfortunate solution to an unfortunate problem.
    # as per textual 0.77, `TabbedContent.clear_panes()` does not clear all references
    # to its tabs. since tabs are holding references to large tables this may cause
    # potentially huge memory leaks. best solution i've managed so far is to delete
    # references to tables on unmounting. i've tried different solutions such as
    # passing references to functions around, with no luck.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self._arr
        del self.arr
        del self.mask

    # async is needed since `filter_table` calls a worker
    async def on_input_changed(self, event: Input.Submitted):
        # noinspection PyAsyncCall
        self.filter_table(event.value)

    # runs possibly slow filter operation with a worker to avoid UI lags
    @work(exclusive=True, group="filter_table")
    async def filter_table(self, query: str):
        """
        Filters a table according to a query and shows it's fist page.

        :param query: the filter query
        :param delay: sets sleep time at start to prevent function calls while
        writing query.
        :return:
        """
        # noinspection PyBroadException
        try:
            filtered_arr = await to_thread(filter_array, query, self._arr)
        except Exception as e:
            self.query_one(InputFilter).add_class("error")
            return
        self.arr = filtered_arr
        self.page_no = 1
        self.page_tot = max(ceil(len(self.arr) / self.page_len), 1)
        self.update_page_display()
        self.query_one(InputFilter).remove_class("error")
        self.app.log_push(
            f"Filtered table by query {repr(query)}, "
            f"{len(filtered_arr)} entries matching the query."
        )

    def page_slice(self):
        """Returns a slice which can be used to index the present page."""
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        """Displays the present table page."""
        table = self.query_one(FitsTable)
        arr = self.arr[self.page_slice()]
        table.update_arr(arr, self.cols)
        self.query_one(FitsTable).border_subtitle = (
            f"page {self.page_no} / {self.page_tot} "
        )

    def action_next_page(self):
        """Scrolls to next page."""
        if self.page_no < self.page_tot:
            self.page_no += 1
            self.update_page_display()

    def action_back_page(self):
        """Scrolls to previous page."""
        if self.page_no > 1:
            self.page_no -= 1
            self.update_page_display()

    def action_last_page(self):
        """Scrolls to last page"""
        self.page_no = self.page_tot
        self.update_page_display()

    def action_first_page(self):
        """Scrolls to first page"""
        self.page_no = 1
        self.update_page_display()

    # TODO: add methods and binding for scrolling to `n` page.


class EmptyDialog(Static):
    """When a FITs HDU contains an image or no data, displays a placeholder."""

    def compose(self) -> ComposeResult:
        yield Label("No tables to show")

    def on_mount(self):
        self.border_title = "Table"


class HeaderDialog(Tree):
    """Displays a FITS header as a tree."""

    def __init__(self, header: dict, ellipsis: int = 14):
        """
        :param header:
        :param ellipsis: sets length after which apply an ellipsis.
        """
        super().__init__(label="root")
        self.leafs = []
        for key, value in header.items():
            node = self.root.add(label=key)
            label = (
                vstr
                if len(vstr := str(value).strip()) < ellipsis
                else vstr[:ellipsis] + ".."
            )
            leaf = node.add_leaf(label, data=str(value))
            self.leafs.append(leaf)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        """Opens a pop-up clicking on a header entry."""
        if event.node in self.leafs:
            self.app.push_screen(HeaderEntry(event.node.data))

    def on_mount(self):
        self.border_title = "Header"
        self.guide_depth = 3
        self.show_guides = True
        self.root.expand()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.leafs


class HDUPane(TabPane):
    """A container for header and table widgets"""

    def __init__(self, content: dict):
        self.content = content
        super().__init__(content["name"] if content["name"].strip() else "HDU")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield HeaderDialog(self.content["header"])
            if self.content["is_table"]:
                yield TableDialog(
                    self.content["data"],
                    self.content["columns"],
                    hide_filter=True if self.content["columns_arrays"] else False,
                )
            else:
                yield EmptyDialog()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.content


class FileInput(LabelInput):
    class Sent(Message):
        """A custom Sent message for FileInput."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self):
        super().__init__(label_text="path")

    def on_mount(self):
        self.border_title = "File"

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.post_message(self.Sent(self.query_one(Input).value))


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@contextmanager
def catchtime() -> Callable[[], float]:
    """A context manager for measuring computing times."""
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


class Misfits(App):
    """Misfits, the main app."""

    TITLE = "Misfits"
    CSS_PATH = "misfits.tcss"
    SCREENS = {"log": LogScreen, "file_explorer": FileExplorerScreen, "info": InfoScreen}
    BINDINGS = [
        ("ctrl+l", "push_screen('log')", "Log"),
        ("ctrl+j", "push_screen('info')", "Info"),
        ("ctrl+o", "open_explorer", "Open"),
    ]

    def __init__(self, filepath: Path | None, root_dir: Path = Path.cwd()) -> None:
        """
        :param filepath: the file to load at startup.
        If `None`, an unescapable file explorer is shown at startup.
        :param root_dir: will set the directory from which file explorers start.
        """
        super().__init__()
        self.filepath = filepath
        self.rootdir = root_dir
        self.fits_content = []
        self.logstack = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield TabbedContent()
        yield FileInput()
        yield Footer()

    @asynccontextmanager
    async def disable_inputs(self, fileinput_delay=0.25):
        """
        Disables input and shows a loading animation while tables are read into memory.

        :param fileinput_delay: seconds delay between end of loading indicator and
        file input prompt release.
        :return:
        """
        fileinput = self.query_one(FileInput)
        fileinput.disabled = True
        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        yield
        tabs.loading = False
        # we wait a bit before releasing the input because quick, repeated sends can
        # cause a tab to not load properly
        await sleep(fileinput_delay)
        fileinput.disabled = False

    # `push_screen_wait` requires a worker
    @work
    async def on_mount(self):
        if not self.filepath:
            self.filepath = await self.push_screen_wait(FileExplorerScreen(self.rootdir))
        self.query_one(FileInput).set_input_value(str(self.filepath))
        # noinspection PyAsyncCall
        self.populate_tabs()

    async def on_file_input_sent(self, message: FileInput.Sent) -> None:
        """Accepts and checks message from file input prompt."""
        input_path = Path(message.value)
        if not _validate_fits(input_path):
            self.query_one(FileInput).add_class("error")
            return
        self.query_one(FileInput).remove_class("error")
        self.filepath = input_path
        # noinspection PyAsyncCall
        self.populate_tabs()

    # `push_screen_wait` requires a worker
    @work
    async def action_open_explorer(self):
        self.filepath = await self.push_screen_wait(EscapableFileExplorerScreen(self.rootdir))
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).set_input_value(str(self.filepath))
        self.query_one(FileInput).remove_class("error")

    # calls CPU-heavy `get_fits_content`, requiring a worker
    # exclusive because otherwise would result in an error everytime we attempt
    # to open a new while one is still loading.
    @work(exclusive=True)
    async def populate_tabs(self, mintime=0.25) -> None:
        """
        Fills the tabs with data read from the FITS' HDUs.

        :param mintime: smallest time in which the loading indicator is displayed.
        """
        async with self.disable_inputs():
            with catchtime() as elapsed:
                tabs = self.query_one(TabbedContent)
                await tabs.clear_panes()
                self.log_push(f"Opening '{self.filepath}'")
                contents = await get_fits_content(self.filepath)
                for i, content in enumerate(contents):
                    await tabs.add_pane(HDUPane(content))
                    self.log_fitcontents(content)

            self.log_push(f"Reading FITS file took {elapsed():.3f} s")
            # to avoid flickering, we wait a bit when FITS reading is fast
            if elapsed() < mintime:
                await sleep(mintime - elapsed())

    # TODO: move these methods to a log class.
    def log_push(self, message: str, level: LogLevel | None = LogLevel.INFO):
        now_str = "[dim cyan]" + datetime.now().strftime("(%H:%M:%S)") + "[/]"
        match level:
            case LogLevel.INFO:
                prefix = f"{now_str} [dim green][INFO][/]: "
            case LogLevel.WARNING:
                prefix = f"{now_str} [dim yellow][WARNING][/]: "
            case LogLevel.ERROR:
                prefix = f"{now_str} [bold red][ERROR][/]: "
            case _:
                prefix = ""
        self.logstack.append(prefix + message)

    def log_pop(self) -> str | None:
        return self.logstack.pop(0) if self.logstack else None

    def log_fitcontents(self, content):
        # fmt: off
        self.log_push(f"Found HDU {repr(content['name'])} of type {repr(content['type'])}")
        if content["data"] is not None:
            ncols = len(content["data"].columns)
            self.log_push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns")
        # fmt: on


def click_validate_fits(
    ctx: click.Context, param: click.Parameter, filepath: Path
) -> Path:
    """Click callback validator."""
    if filepath.is_file() and not _validate_fits(filepath):
        raise click.FileError(
            f"Invalid input.",
            hint="Please, check misfits `INPUT_PATH` argument "
            "and make sure it points to a FITS file.",
        )
    return filepath


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    callback=click_validate_fits,
)
def main(input_path: Path):
    """Misfits is an interactive FITs viewer for the terminal."""
    filepath, rootdir = (
        (None, input_path) if input_path.is_dir() else (input_path, Path.cwd())
    )
    Misfits(filepath, rootdir).run(inline=False)


if __name__ == "__main__":
    app = Misfits(None, Path(r"D:/Dropbox/Progetti/"))
    app.run()
