import asyncio
from datetime import datetime
from enum import Enum
from math import ceil
from math import log10
from pathlib import Path
from random import choice
from string import ascii_letters, digits

from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU
from astropy.io.fits.hdu.table import TableHDU
from astropy.table import Table
import click
import pandas as pd
from textual import events
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Container
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import RichLog
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import TextArea
from textual.widgets import Tree, DirectoryTree


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

LOGO = "".join(
    [
        (
            choice(ascii_letters + digits)
            if s == "0"
            else s
        )
        for s in _LOGO
    ]
)


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
            yield Input(f"Enter query (e.g. 'COL1 > 42 & COL2 == 3)'")

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    def __init__(self, df: pd.DataFrame, page_len: int = 1000):
        super().__init__()
        self.df = df
        self.page_len = page_len
        self.shown_df = df
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(df) / page_len), 1)

    def page_slice(self):
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        zpad = int(log10(self.page_tot)) + 1
        page_display = self.query_one(Label)
        page_display.update(f" {self.page_no:0{zpad}} / {self.page_tot} ")

    def next_page(self):
        if self.page_no < self.page_tot:
            self.page_no += 1
            table = self.query_one(DataFrameTable)
            table.update_df(self.shown_df[self.page_slice()])
            self.update_page_display()

    def back_page(self):
        if self.page_no > 1:
            self.page_no -= 1
            table = self.query_one(DataFrameTable)
            table.update_df(self.shown_df[self.page_slice()])
            self.update_page_display()

    def last_page(self):
        self.page_no = self.page_tot
        table = self.query_one(DataFrameTable)
        table.update_df(self.shown_df[self.page_slice()])
        self.update_page_display()

    def first_page(self):
        self.page_no = 1
        table = self.query_one(DataFrameTable)
        table.update_df(self.shown_df[self.page_slice()])
        self.update_page_display()

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
        table = self.query_one(DataFrameTable)
        table.update_df(self.shown_df[self.page_slice()])
        self.update_page_display()
        self.app.log_push(
            f"Filtered table by query {repr(query)}, "
            f"{len(filtered_df)} entries matching the query."
        )

    def compose(self) -> ComposeResult:
        yield DataFrameTable()
        yield PageControls()
        yield InputFilter()

    def on_mount(self):
        table = self.query_one(DataFrameTable)
        table.add_df(self.shown_df[self.page_slice()])
        self.update_page_display()

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

    async def on_input_changed(self, event: Input.Submitted):
        return self.filter_table(event.value)


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
            if len(vstr := str(value).strip()) < hide_over:
                label = vstr
                node.expand()
            else:
                label = vstr[:hide_over] + ".."
            self.leafs.append(node.add_leaf(label, data=str(value)))

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
        super().__init__(content["name"])

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


class FileExplorer(ModalScreen):
    TITLE = "Open file"
    SUB_TITLE = ""
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    def compose(self) -> ComposeResult:
        with Container():
            yield Header(show_clock=False)
            yield DirectoryTree("./")
        yield Footer()

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        if not _validate_fits(event.path):
            dirtree = self.query_one(DirectoryTree)
            dirtree.add_class("error")
            return
        dirtree = self.query_one(DirectoryTree)
        dirtree.remove_class("error")
        self.app.pop_screen()
        self.app.populate_tabs(event.path)


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
    SUB_TITLE = "a terminal FITS viewer"
    CSS_PATH = "misfits.scss"
    SCREENS = {"log": LogScreen, "file_explorer": FileExplorer}
    BINDINGS = [
        ("ctrl+l", "push_screen('log')", "Show log"),
        ("ctrl+o", "push_screen('file_explorer')", "Open file"),
    ]

    def __init__(self, input_path: Path) -> None:
        super().__init__()
        self.input_path = input_path
        self.fits_content = []
        self.logstack = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TabbedContent()
        yield Footer()

    def on_mount(self):
        self.fits_content = self.populate_tabs(self.input_path)

    @work
    async def populate_tabs(self, input_path: Path):
        def log_fitcontents(content):
            # fmt: off
            self.log_push(f"Found HDU {repr(content['name'])} of type {repr(content['type'])}.")
            if content["data"] is not None:
                ncols = len(content["data"].columns) + len(content["multicols"]) if content["multicols"] else len(content["data"].columns)
                self.log_push(f"HDU contains a table with {len(content['data'])} rows and {ncols} columns.")
            if content["multicols"]:
                self.log_push(f"Dropping multilevel columns: {', '.join(map(repr, content['multicols']))}", LogLevel.WARNING)
            # fmt: on

        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        tabs.clear_panes()
        self.log_push(f"Opening '{input_path}'")
        contents = await asyncio.to_thread(get_fits_content, input_path)
        for i, content in enumerate(contents):
            await tabs.add_pane(HDUPane(content))
            log_fitcontents(content)
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


def _validate_fits(filepath: Path) -> bool:
    try:
        _ = fits.getheader(filepath)
    except OSError as e:
        return False
    return True


def click_validate_fits(ctx: click.Context, param: click.Parameter, filepath: Path) -> Path:
    if not _validate_fits(filepath):
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
    app = Misfits(input_path)
    app.run(inline=False)


if __name__ == "__main__":
    app = Misfits(
        Path("/Users/peppedilillo/Dropbox/Progetti/fits-tui/fermi-fits.fits.gz")
    )
    app.run()
