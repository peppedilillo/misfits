import asyncio
from datetime import datetime
from enum import Enum
from math import ceil
from math import log10
from pathlib import Path
from random import choice
from string import ascii_letters
from string import digits
from typing import Iterable

from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.hdu.table import TableHDU
from astropy.table import Table
import click
import pandas as pd
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
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
from textual.design import ColorSystem
from textual.app import DEFAULT_COLORS

DEFAULT_COLORS["dark"] = ColorSystem(
    primary="#03A062",
    secondary="#03A062",
    warning="#03A062",
    error="#ff0000",
    success="#00ff00",
    accent="#00ff00",
    dark=True,
)

_LOGO = """            
     0           0                                                       
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
        self.border_title = "Data"
        self.cursor_type = "row"


class PageControls(Static):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("[ |◀─ ]", id="first_button")
            yield Button("[ ◀─  ]", id="back_button")
            yield Label("Hello", id="page_display")
            yield Button("[  ─▶ ]", id="next_button")
            yield Button("[ ─▶| ]", id="last_button")

    def on_mount(self):
        self.border_title = "Pages"


class InputFilter(Static):
    def compose(self) -> ComposeResult:
        with Container():
            yield Input(placeholder=f"Enter query (e.g. 'COL1 > 42 & COL2 == 3)'")

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    def __init__(self, df: pd.DataFrame, page_len: int = 100):
        super().__init__()
        self.df = df
        self.page_len = page_len
        self.shown_df = df
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(df) / page_len), 1)

    def compose(self) -> ComposeResult:
        yield DataFrameTable()
        yield PageControls()
        yield InputFilter()

    def on_mount(self):
        self.query_one(DataFrameTable).update_df(self.shown_df[self.page_slice()])
        self.update_page_display()

    async def on_input_changed(self, event: Input.Submitted):
        return self.filter_table(event.value)

    def on_button_pressed(self, event: Button.Pressed):
        """Event handler called when a button is pressed."""
        match event.button.id:
            case "next_button":
                self.next_page()
            case "back_button":
                self.back_page()
            case "first_button":
                self.first_page()
            case "last_button":
                self.last_page()
            case _:
                raise ValueError("Unknown button.")
        self.update_page_display()

    def page_slice(self):
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        zpad = int(log10(self.page_tot)) + 1
        self.query_one(Label).update(f" {self.page_no:0{zpad}} / {self.page_tot} ")
        self.query_one(DataFrameTable).update_df(self.shown_df[self.page_slice()])

    def next_page(self):
        if self.page_no < self.page_tot:
            self.page_no += 1

    def back_page(self):
        if self.page_no > 1:
            self.page_no -= 1

    def last_page(self):
        self.page_no = self.page_tot

    def first_page(self):
        self.page_no = 1

    @work(exclusive=True)
    async def filter_table(self, query: str):
        # noinspection PyBroadException
        try:
            filtered_df = await asyncio.to_thread(self.df.query, query)
        except Exception as e:
            return
        self.shown_df = filtered_df
        self.page_no = 1
        self.page_tot = max(ceil(len(self.shown_df) / self.page_len), 1)
        self.update_page_display()
        self.app.log_push(
            f"Filtered table by query {repr(query)}, "
            f"{len(filtered_df)} entries matching the query."
        )


class EmptyDialog(Static):
    def compose(self) -> ComposeResult:
        yield Label("No tables to show")

    def on_mount(self):
        self.border_title = "Table"


class MoreScreen(ModalScreen):
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
    def __init__(self, header: dict, hide_over: int = 12):
        super().__init__(label="header")
        self.leafs = []
        for key, value in header.items():
            node = self.root.add(label=key)
            label = vstr if len(vstr := str(value).strip()) < hide_over else vstr[:hide_over] + ".."
            leaf = node.add_leaf(label, data=str(value))
            self.leafs.append(leaf)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node in self.leafs:
            self.app.push_screen(MoreScreen(event.node.data))

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


def get_fits_content(fits_path: str | Path) -> tuple[dict]:
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


class FilteredDirectoryTree(DirectoryTree):
    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [path for path in paths if not path.name.startswith(".")]


class FileExplorer(ModalScreen):
    TITLE = "Open file"
    SUB_TITLE = ""

    def __init__(self, rootdir: Path = Path.cwd()):
        super().__init__()
        self.rootdir = rootdir

    def compose(self) -> ComposeResult:
        with Container():
            yield Header(show_clock=False)
            yield FilteredDirectoryTree(self.rootdir)
        yield Footer()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        if not _validate_fits(event.path):
            self.query_one(DirectoryTree).add_class("error")
            return
        self.query_one(DirectoryTree).remove_class("error")
        self.dismiss(event.path)


class EscapableFileExplorer(FileExplorer):
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]


class LogScreen(ModalScreen):
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


class Misfits(App):
    """Main app."""

    TITLE = "Misfits"
    CSS_PATH = "misfits.scss"
    SCREENS = {"log": LogScreen, "file_explorer": FileExplorer}
    BINDINGS = [
        ("ctrl+l", "push_screen('log')", "Show log"),
        ("ctrl+o", "open_explorer", "Open file"),
    ]

    def __init__(self, filepath: Path, root_dir: Path = Path.cwd()) -> None:
        super().__init__()
        self.filepath = filepath
        self.rootdir = root_dir
        self.fits_content = []
        self.logstack = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield TabbedContent()
        yield Footer()

    @work
    async def on_mount(self):
        if not self.filepath:
            self.filepath = await self.push_screen_wait(FileExplorer(self.rootdir))
        self.fits_content = self.populate_tabs()

    @work
    async def action_open_explorer(self):
        self.filepath = await self.push_screen_wait(EscapableFileExplorer(self.rootdir))
        self.fits_content = self.populate_tabs()

    @work
    async def populate_tabs(self):
        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        await tabs.clear_panes()
        self.log_push(f"Opening '{self.filepath}'")
        contents = await asyncio.to_thread(get_fits_content, self.filepath)
        for i, content in enumerate(contents):
            await tabs.add_pane(HDUPane(content))
            self.log_fitcontents(content)
        tabs.loading = False

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
        self.log_push(f"Found HDU {repr(content['name'])} of type {repr(content['type'])}.")
        if content["data"] is not None:
            ncols = len(content["data"].columns) + len(content["multicols"]) if content["multicols"] else len(content["data"].columns)
            self.log_push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns.")
        if content["multicols"]:
            self.log_push(f"Dropping multilevel columns: {', '.join(map(repr, content['multicols']))}", LogLevel.WARNING)
        # fmt: on


FITS_SIGNATURE = b"SIMPLE  =                    T"


def _validate_fits(filepath: Path) -> bool:
    # Following the same approach of astropy.
    with open(filepath, "rb") as file:
        # FITS signature is supposed to be in the first 30 bytes, but to
        # allow reading various invalid files we will check in the first
        # card (80 bytes).
        simple = file.read(80)
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
    filepath, rootdir = (None, input_path) if input_path.is_dir() else (input_path, Path.cwd())
    app = Misfits(filepath, rootdir)
    app.run(inline=False)


if __name__ == "__main__":
    app = Misfits(None, Path("/Users/peppedilillo/Dropbox/Progetti/PerformancesPaper/data"))
    app.run()
