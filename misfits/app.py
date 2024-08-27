import asyncio
from datetime import datetime
from enum import Enum
from math import ceil
from math import log10
from pathlib import Path
from random import choice
import string

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
from textual.screen import Screen
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
from textual.widgets import Tree

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
            choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
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
        yield Container(
            Input(
                placeholder=f"Enter query (e.g. `COL1 > 42 &  COL2 == 3)`",
            )
        )

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
        button_id = event.button.id
        if button_id == "next_button":
            self.next_page()
        elif button_id == "back_button":
            self.back_page()
        elif button_id == "first_button":
            self.first_page()
        elif button_id == "last_button":
            self.last_page()

    async def on_input_changed(self, event: Input.Submitted):
        return self.filter_table(event.value)

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


class EmptyDialog(Static):
    def compose(self) -> ComposeResult:
        yield Label("No tables to show")

    def on_mount(self):
        self.border_title = "Data"


class MoreScreen(ModalScreen):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Container():
            yield TextArea.code_editor(self.text, read_only=True)
            yield Label("Press ESC to close.")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()


class HeaderDialog(Tree):
    def __init__(self, header: dict, hide_over: int = 12, *args, **kwargs):
        super().__init__(label="header", *args, **kwargs)
        self.border_title = "Header"
        self.truncated = {}
        self.guide_depth = 3
        self.show_guides = True
        self.root.expand()
        for key, value in header.items():
            node = self.root.add(label=key)
            if len(vstr := str(value).strip()) < hide_over:
                node.add_leaf(vstr)
                node.expand()
            else:
                leaf = node.add_leaf(vstr[:hide_over] + "..")
                self.truncated[leaf] = str(value)

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node in self.truncated:
            self.app.push_screen(MoreScreen(self.truncated[event.node]))


class HDUPane(TabPane):
    def __init__(self, content: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content = content

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield HeaderDialog(self.content["header"])
            if self.content["type"] == "table":
                yield TableDialog(self.content["data"])
            else:
                yield EmptyDialog()


def get_fits_content(fits_path: str | Path) -> tuple[dict]:
    def is_table(hdu):
        return type(hdu) in [TableHDU, BinTableHDU]

    def sanitize(table):
        names = [name for name in table.colnames if len(table[name].shape) <= 1]
        return table[names]

    with fits.open(fits_path) as hdul:
        content = tuple(
            {
                "type": "table" if is_table else "other",
                "header": dict(hdu.header) if hdu.header else None,
                "data": sanitize(Table(hdu.data)).to_pandas() if is_table else None,
            }
            for i, (is_table, hdu) in enumerate(zip(map(is_table, hdul), hdul))
        )
    return content


class LogScreen(ModalScreen):
    BINDINGS = [("escape", "app.pop_screen", "Show dashboard")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=False, markup=True)
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

    CSS_PATH = "misfits.scss"
    SCREENS = {"log": LogScreen}
    BINDINGS = [("ctrl+l", "push_screen('log')", "Show log")]

    def __init__(self, input_path: Path) -> None:
        super().__init__()
        self.input_path = input_path
        self.fits_content = []
        self.logstack = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield TabbedContent()
        yield Footer()

    def on_mount(self):
        self.log_push(
            f"\nHey, this is..\n[bold green]{LOGO}"
            "[/]\n\nThe spooky FITS viewer.\n"
            "Nice to meet you! Let's begin.\n",
            level=None,
        )
        self.fits_content = self.populate_tabs(self.input_path)

    @work
    async def populate_tabs(self, input_path: Path):
        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        self.log_push(f"Opening '{input_path.name}'")
        contents = await asyncio.to_thread(get_fits_content, input_path)
        for i, content in enumerate(contents):
            await tabs.add_pane(HDUPane(content, f"HDU-{i}"))
            self.log_push(f"Found HDU of type {repr(content['type'])}.")
        tabs.loading = False

    def log_push(self, message: str, level: LogLevel | None = LogLevel.INFO):
        now_str = "[dim cyan]" + datetime.now().strftime("(%H:%M:%S)") + "[/]"
        match level:
            case LogLevel.INFO:
                prefix = f"{now_str} [dim green][INFO][/]: "
            case LogLevel.WARNING:
                prefix = f"{now_str} [dim orange][WARNING][/]: "
            case LogLevel.ERROR:
                prefix = f"{now_str} [dim red][ERROR][/]: "
            case _:
                prefix = ""
        self.logstack.append(prefix + message)

    def log_pop(self) -> str | None:
        return self.logstack.pop(0) if self.logstack else None


def validate_fits(ctx: click.Context, param: click.Option, filepath: Path) -> Path:
    try:
        _ = fits.getheader(filepath)
    except OSError as e:
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
    callback=validate_fits,
)
def main(input_path: Path):
    app = Misfits(input_path)
    app.run(inline=False)


if __name__ == "__main__":
    app = Misfits(
        Path("/Users/peppedilillo/Dropbox/Progetti/fits-tui/fermi-fits.fits.gz")
    )
    app.run()
