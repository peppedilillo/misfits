"""
A terminal FITS viewer with interactive tables.

Author: Giuseppe Dilillo
Date:   August 2024
"""

import asyncio
from asyncio import sleep
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
from enum import Enum
from math import ceil
from pathlib import Path
from random import choice
from string import ascii_letters
from string import digits
from time import perf_counter
from typing import Callable, Iterable

from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.hdu.table import TableHDU
from astropy.table import Table
import click
import pandas as pd
from rich.text import Text
from textual import events
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.app import DEFAULT_COLORS
from textual.containers import Container
from textual.containers import Horizontal
from textual.design import ColorSystem
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import DirectoryTree
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import RichLog
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import TextArea
from textual.widgets import Tree

_LOGO = """    0           0                                                       
   0000000     000000               0000000000000   000000000000000     
   0000000    0000000               000000000000      000000000         
   0000000    000000  0000    0000     00000     0000  000000    0000   
   0000000   0000000  000   0000000    000000000 000   00000   00000000 
   00000000 00000000  000  0000    0   0000000   000    00000 00000   0 
   00000000 0000000   000  00000       0000      000    0000  00000     
   000000000000 0000  000   0000000    0000      000     000    00000   
    000 000000  0000  000       00000  0000      000    0000        0000
   0000  0000   000    00 0000000000   0000       00    0000 0000000000 
   0000  0000   000         00000      00               00      0000    
   0000   00    000                                                     
"""

LOGO = "".join([(choice(ascii_letters + digits) if s == "0" else s) for s in _LOGO])


DEFAULT_COLORS["dark"] = ColorSystem(
    primary="#03A062",  # matrix green
    secondary="#03A062",
    warning="#03A062",
    error="#ff0000",
    success="#00ff00",
    accent="#00ff00",
    dark=True,
)

SHARED_PROCESS_POOL = ProcessPoolExecutor(max_workers=1)


class DataFrameTable(DataTable):
    """Display Pandas dataframe in DataTable widget.
    Based on `https://github.com/dannywade/textual-pandas`"""

    def add_df(self, df: pd.DataFrame):
        """Add DataFrame data to DataTable."""
        self.add_columns(*tuple(df.columns.values.tolist()))
        self.add_rows(list(df.itertuples(index=False, name=None))[0:])
        return self

    def update_df(self, df: pd.DataFrame):
        """Update DataFrameTable with a new DataFrame."""
        # Clear existing datatable
        self.clear(columns=True)
        # Redraw table with new dataframe
        self.add_df(df)

    def on_mount(self):
        self.border_title = "Table"
        self.cursor_type = "row"


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

    def __init__(self, df: pd.DataFrame, page_len: int = 100):
        """
        :param df: The dataframe to show
        :param page_len: How many dataframe rows are shown for each page.
        """
        super().__init__()
        self.df = df
        self.mask = df.index
        self.page_len = page_len
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(df) / page_len), 1)

    def compose(self) -> ComposeResult:
        yield DataFrameTable()
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
        del self.df
        del self.mask

    # async is needed since `filter_table` calls a worker
    async def on_input_changed(self, event: Input.Submitted):
        # noinspection PyAsyncCall
        self.filter_table(event.value)

    # runs possibly slow filter operation with a worker to avoid UI lags
    @work(exclusive=True, group="filter_table")
    async def filter_table(self, query: str, delay=0.25):
        """
        Filters a table according to a query and shows it's fist page.

        :param query: the filter query
        :param delay: sets sleep time at start to prevent function calls while
        writing query.
        :return:
        """
        # noinspection PyBroadException
        await sleep(delay)
        try:
            fdf = await asyncio.to_thread(self.df.query, query) if query else self.df
        except Exception as e:
            self.query_one(InputFilter).add_class("error")
            return
        self.mask = fdf.index
        self.page_no = 1
        self.page_tot = max(ceil(len(self.mask) / self.page_len), 1)
        self.update_page_display()
        self.query_one(InputFilter).remove_class("error")
        self.app.log_push(
            f"Filtered table by query {repr(query)}, "
            f"{len(fdf)} entries matching the query."
        )

    def page_slice(self):
        """Returns a slice which can be used to index the present page."""
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        """Displays the present table page."""
        table = self.query_one(DataFrameTable)
        table.update_df(self.df.iloc[self.mask[self.page_slice()]])
        table.border_subtitle = f"page {self.page_no} / {self.page_tot} "

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


class HeaderEntry(ModalScreen):
    """Displays header's entries in a pop-up screen. Useful with long entries."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]
    TITLE = "Header entry"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Container():
            yield Header()
            yield TextArea.code_editor(self.text, read_only=True)
        yield Footer()


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
                yield TableDialog(self.content["data"])
            else:
                yield EmptyDialog()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.content


def _get_fits_content(fits_path: str | Path) -> tuple[dict]:
    """Retrieves content from a FITS file and stores it in a tuple dict.
    Each tuple's records referes to one FITS HDU. CPU-heavy."""

    def is_table(hdu):
        return type(hdu) in [TableHDU, BinTableHDU]

    def multicols(data):
        return [c.name for c in data.columns if len(c.array.shape) > 1]

    def to_pandas(data):
        scols = [c.name for c in data.columns if len(c.array.shape) == 1]
        # filters out multicolumns
        # TODO: improve how we deal with multicolumn tables
        return Table(data)[scols].to_pandas()

    with fits.open(fits_path) as hdul:
        content = tuple(
            {
                "name": hdu.name,
                "type": hdu.__class__.__name__,
                "header": dict(hdu.header) if hdu.header else None,
                "is_table": is_table,
                "data": to_pandas(hdu.data) if is_table else None,
                "multicols": multicols(hdu.data) if is_table else None,
            }
            for i, (is_table, hdu) in enumerate(zip(map(is_table, hdul), hdul))
        )
    return content


async def get_fits_content(filepath: Path):
    """Factors `get_fits_content` to a separate process.
    This precaution is taken to avoid staggering and lags with the UI."""
    loop = asyncio.get_event_loop()
    contents, *_ = (
        await loop.run_in_executor(SHARED_PROCESS_POOL, _get_fits_content, filepath),
    )
    return contents


class FilteredDirectoryTree(DirectoryTree):
    """A directory tree widget filtering hidden files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [path for path in paths if not path.name.startswith(".")]


class FileExplorer(ModalScreen):
    """A pop-up screen showing a file explorer so that the user may choose an
    input navigating the file system. To be used at main app's start-up,
     if no input file is provided. For this reason the screen is not escapable."""

    TITLE = "Open file"

    def __init__(self, rootdir: Path = Path.cwd()):
        super().__init__()
        self.rootdir = rootdir

    def compose(self) -> ComposeResult:
        with Container():
            yield Header(show_clock=False)
            yield FilteredDirectoryTree(self.rootdir)
        yield Footer()

    # threaded because validating requires IO which may cause laggish behaviour
    @work(exclusive=True, group="file_check")
    async def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        isvalid = await asyncio.to_thread(_validate_fits, event.path)
        if not isvalid:
            self.query_one(DirectoryTree).add_class("error")
            return
        self.query_one(DirectoryTree).remove_class("error")
        # noinspection PyAsyncCall
        self.dismiss(event.path)


class EscapableFileExplorer(FileExplorer):
    """Like `FileExplorer` but with bindings to leave the screen.
    To be used when a file input has already been provided."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]


class InfoScreen(ModalScreen):
    """A pop-up showing some cool infos on the program."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    def get_text(self):
        return Text.from_markup(
            f"A FITS table viewer by G.D.\n"
            f"[dim]https://github.com/peppedilillo\n"
            f"https://gdilillo.com",
            justify="center",
        )

    def compose(self) -> ComposeResult:
        with Container():
            yield Static(f"[green bold]{LOGO}")
            yield Static(self.get_text())
        yield Footer()


class FileInput(Input):
    """A prompt widget to input via file paths."""

    class Sent(Message):
        """A message to the main app so that the input can be loaded."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("[dim italic]path: ")
            yield Input()
            yield Button("Send", id="fi_send")
            yield Button("Clear", id="fi_clear")

    def on_mount(self):
        self.border_title = "File"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fi_clear":
            self.set_input_value("")
            self.remove_class("error")
        elif event.button.id == "fi_send":
            self.post_message(self.Sent(self.query_one(Input).value))

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.post_message(self.Sent(self.query_one(Input).value))

    def set_input_value(self, value: str):
        self.query_one(Input).value = value


class LogScreen(ModalScreen):
    """An alternative screen showing a log"""

    TITLE = "Log"
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True)
        yield Footer()

    def on_screen_resume(self):
        """When screen is shown, pushes message on the stack to the screen."""
        log = self.query_one(RichLog)
        while line := self.app.log_pop():
            log.write(line)


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
    CSS_PATH = "misfits.scss"
    SCREENS = {"log": LogScreen, "file_explorer": FileExplorer, "info": InfoScreen}
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
    async def disable_inputs(self, fileinput_delay=0.5):
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
            self.filepath = await self.push_screen_wait(FileExplorer(self.rootdir))
        self.query_one(FileInput).set_input_value(str(self.filepath))
        # noinspection PyAsyncCall
        self.populate_tabs()

    async def on_file_input_sent(self, message: FileInput.Sent) -> None:
        """Accepts and checks message from file input prompt."""
        input_path = Path(message.value)
        if not _validate_fits(input_path):
            self.query_one(FileInput).add_class("error")
            return
        self.filepath = input_path
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).remove_class("error")

    # `push_screen_wait` requires a worker
    @work
    async def action_open_explorer(self):
        self.filepath = await self.push_screen_wait(EscapableFileExplorer(self.rootdir))
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).set_input_value(str(self.filepath))
        self.query_one(FileInput).remove_class("error")

    # calls CPU-heavy `get_fits_content`, requiring a worker
    # exclusive because otherwise would result in an error everytime we attempt
    # to open a new while one is still loading.
    @work(exclusive=True)
    async def populate_tabs(self, mintime=0.5) -> None:
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
            ncols = len(content["data"].columns) + len(content["multicols"]) if content["multicols"] else len(content["data"].columns)
            self.log_push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns")
        if content["multicols"]:
            self.log_push(f"Dropping multilevel columns: {', '.join(map(repr, content['multicols']))}", LogLevel.WARNING)
        # fmt: on


FITS_SIGNATURE = b"SIMPLE  =                    T"


def _validate_fits(filepath: Path) -> bool:
    """Checks if a file is a FITS."""
    # Following the same approach of astropy.
    try:
        with open(filepath, "rb") as file:
            # FITS signature is supposed to be in the first 30 bytes, but to
            # allow reading various invalid files we will check in the first
            # card (80 bytes).
            simple = file.read(80)
    except OSError:
        return False
    match_sig = simple[:29] == FITS_SIGNATURE[:-1] and simple[29:30] in (b"T", b"F")
    return match_sig


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
    app = Misfits(
        None, Path("/Users/peppedilillo/Dropbox/Progetti/PerformancesPaper/data")
    )
    app.run()
