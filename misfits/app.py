import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from enum import Enum
from math import ceil
from pathlib import Path
from random import choice
from string import ascii_letters
from string import digits
from typing import Iterable, Callable
from contextlib import contextmanager
from asyncio import sleep
from time import perf_counter

from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.hdu.table import TableHDU
from astropy.table import Table
import click
import pandas as pd
from rich.text import Text
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.app import DEFAULT_COLORS
from textual.containers import Container
from textual.containers import Horizontal
from textual.design import ColorSystem
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
from textual.message import Message
from textual import events

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
    primary="#03A062",
    secondary="#03A062",
    warning="#03A062",
    error="#ff0000",
    success="#00ff00",
    accent="#00ff00",
    dark=True,
)

SHARED_PROCESS_POOL = ProcessPoolExecutor(max_workers=1)


class DataFrameTable(DataTable):
    """Display Pandas dataframe in DataTable widget."""
    def add_df(self, df: pd.DataFrame):
        """Add DataFrame data to DataTable."""
        self.df = df
        self.add_columns(*self._add_df_columns())
        self.add_rows(self._add_df_rows()[0:])
        return self

    def update_df(self, df: pd.DataFrame):
        """Update DataFrameTable with a new DataFrame."""
        # Clear existing datatable
        self.clear(columns=True)
        # Redraw table with new dataframe
        self.add_df(df)

    def _add_df_rows(self):
        return self._get_df_rows()

    def _add_df_columns(self):
        return self._get_df_columns()

    def _get_df_rows(self) -> list[tuple]:
        """Convert dataframe rows to iterable."""
        return list(self.df.itertuples(index=False, name=None))

    def _get_df_columns(self) -> tuple:
        """Extract column names from dataframe."""
        return tuple(self.df.columns.values.tolist())

    def on_mount(self):
        self.border_title = "Table"
        self.cursor_type = "row"


class InputFilter(Static):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("[dim italic] query: ")
            yield Input(placeholder=f"COL1 > 42 & COL2 == 3")

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    BINDINGS = [
        ("left", "back_page()", "Back"),
        ("right", "next_page()", "Next"),
        ("shift+left", "first_page()", "First"),
        ("shift+right", "last_page()", "Last"),
    ]

    def __init__(self, df: pd.DataFrame, page_len: int = 100):
        super().__init__()
        self.df = df
        self.page_len = page_len
        self.shown_df = df
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(df) / page_len), 1)

    def compose(self) -> ComposeResult:
        yield DataFrameTable()
        yield InputFilter()

    def on_mount(self):
        self.query_one(DataFrameTable).update_df(self.shown_df[self.page_slice()])
        self.border_title = "Table"
        self.update_page_display()

    async def on_input_changed(self, event: Input.Submitted):
        # noinspection PyAsyncCall
        self.filter_table(event.value)

    @work(exclusive=True, group="filter_table")
    async def filter_table(self, query: str):
        # noinspection PyBroadException
        try:
            filtered_df = await asyncio.to_thread(self.df.query, query) if query else self.df
        except Exception as e:
            self.query_one(InputFilter).add_class("error")
            return
        self.shown_df = filtered_df
        self.page_no = 1
        self.page_tot = max(ceil(len(self.shown_df) / self.page_len), 1)
        self.update_page_display()
        self.query_one(InputFilter).remove_class("error")
        self.app.log_push(
            f"Filtered table by query {repr(query)}, "
            f"{len(filtered_df)} entries matching the query."
        )

    def page_slice(self):
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        table = self.query_one(DataFrameTable)
        table.update_df(self.shown_df[self.page_slice()])
        table.border_subtitle = f"page {self.page_no} / {self.page_tot} "

    def action_next_page(self):
        if self.page_no < self.page_tot:
            self.page_no += 1
            self.update_page_display()

    def action_back_page(self):
        if self.page_no > 1:
            self.page_no -= 1
            self.update_page_display()

    def action_last_page(self):
        self.page_no = self.page_tot
        self.update_page_display()

    def action_first_page(self):
        self.page_no = 1
        self.update_page_display()


class EmptyDialog(Static):
    def compose(self) -> ComposeResult:
        yield Label("No tables to show")

    def on_mount(self):
        self.border_title = "Table"


class HeaderEntry(ModalScreen):
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]
    TITLE = "Header entry"
    SUB_TITLE = ""

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Container():
            yield Header()
            yield TextArea.code_editor(self.text, read_only=True)
        yield Footer()


class HeaderDialog(Tree):
    def __init__(self, header: dict, hide_over: int = 14):
        super().__init__(label="root")
        self.leafs = []
        for key, value in header.items():
            node = self.root.add(label=key)
            label = (
                vstr
                if len(vstr := str(value).strip()) < hide_over
                else vstr[:hide_over] + ".."
            )
            leaf = node.add_leaf(label, data=str(value))
            self.leafs.append(leaf)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node in self.leafs:
            self.app.push_screen(HeaderEntry(event.node.data))

    def on_mount(self):
        self.border_title = "Header"
        self.guide_depth = 3
        self.show_guides = True
        self.root.expand()


class HDUPane(TabPane):
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


def _get_fits_content(fits_path: str | Path) -> tuple[dict]:
    def is_table(hdu):
        return type(hdu) in [TableHDU, BinTableHDU]

    def multicols(data):
        return [c.name for c in data.columns if len(c.array.shape) > 1]

    def to_pandas(data):
        scols = [c.name for c in data.columns if len(c.array.shape) == 1]
        # will filter out multicolumns
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
    loop = asyncio.get_event_loop()
    contents, *_ = (
        await loop.run_in_executor(SHARED_PROCESS_POOL, _get_fits_content, filepath),
    )
    return contents


class FilteredDirectoryTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [path for path in paths if not path.name.startswith(".")]


class FileExplorer(ModalScreen):
    TITLE = "Open file"

    def __init__(self, rootdir: Path = Path.cwd()):
        super().__init__()
        self.rootdir = rootdir

    def compose(self) -> ComposeResult:
        with Container():
            yield Header(show_clock=False)
            yield FilteredDirectoryTree(self.rootdir)
        yield Footer()

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
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]


class InfoScreen(ModalScreen):
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
    class Sent(Message):
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

    @work
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "fi_clear":
            self.set_input_value("")
            self.remove_class("error")
        elif event.button.id == "fi_send":
            self.post_message(self.Sent(self.query_one(Input).value))

    @work
    async def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.post_message(self.Sent(self.query_one(Input).value))

    def set_input_value(self, value: str):
        self.query_one(Input).value = value


class LogScreen(ModalScreen):
    TITLE = "Log"
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True)
        yield Footer()

    def on_screen_resume(self):
        log = self.query_one(RichLog)
        while line := self.app.log_pop():
            log.write(line)


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@contextmanager
def catchtime() -> Callable[[], float]:
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


class Misfits(App):
    """Main app."""

    TITLE = "Misfits"
    CSS_PATH = "misfits.scss"
    SCREENS = {"log": LogScreen, "file_explorer": FileExplorer, "info": InfoScreen}
    BINDINGS = [
        ("ctrl+l", "push_screen('log')", "Log"),
        ("ctrl+i", "push_screen('info')", "Infos"),
        ("ctrl+o", "open_explorer", "Open file"),
    ]

    def __init__(self, filepath: Path | None, root_dir: Path = Path.cwd()) -> None:
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

    @contextmanager
    def disable_inputs(self):
        fileinput = self.query_one(FileInput)
        fileinput.disabled = True
        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        yield
        tabs.loading = False
        fileinput.disabled = False

    @work
    async def on_mount(self):
        if not self.filepath:
            self.filepath = await self.push_screen_wait(FileExplorer(self.rootdir))
        self.query_one(FileInput).set_input_value(str(self.filepath))
        # noinspection PyAsyncCall
        self.populate_tabs()

    async def on_file_input_sent(self, message: FileInput.Sent) -> None:
        input_path = Path(message.value)
        if not _validate_fits(input_path):
            self.query_one(FileInput).add_class("error")
            return
        self.filepath = input_path
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).remove_class("error")

    @work
    async def action_open_explorer(self):
        self.filepath = await self.push_screen_wait(EscapableFileExplorer(self.rootdir))
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).set_input_value(str(self.filepath))
        self.query_one(FileInput).remove_class("error")

    @work
    async def populate_tabs(self, mintime=0.5):
        with self.disable_inputs():
            with catchtime() as elapsed:
                tabs = self.query_one(TabbedContent)
                await tabs.clear_panes()
                self.log_push(f"Opening '{self.filepath}'")
                contents = await get_fits_content(self.filepath)
                for i, content in enumerate(contents):
                    await tabs.add_pane(HDUPane(content))
                    self.log_fitcontents(content)
            self.log_push(f"Reading FITS file took {elapsed():.3f} s")
            if elapsed() < mintime:
                await sleep(mintime - elapsed())

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
    filepath, rootdir = (
        (None, input_path) if input_path.is_dir() else (input_path, Path.cwd())
    )
    Misfits(filepath, rootdir).run(inline=False)


if __name__ == "__main__":
    app = Misfits(
        None, Path("/Users/peppedilillo/Dropbox/Progetti/PerformancesPaper/data")
    )
    app.run()
