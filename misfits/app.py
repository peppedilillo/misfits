import asyncio
from math import ceil
from math import log10
from pathlib import Path

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
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import TextArea
from textual.widgets import Tree

LOGO = """
000        00    
00000    00000
00000    00000          0000000000  000000000   
00000    00000  00  0000  000     0   000   0000
 00000  000000  00 000    000000  00  000 000
000000 000 00   00 000     00     0    00 000
000  0000  000  00    000  00     00   00    000
 00  000   000  00 00  000 00     00   00 00 000
 00   00   00   0   00     0      00   0
 00         0
 """


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
        with Horizontal(id="pagecontrol_container"):
            yield Button("[  |◀─  ]", id="first_button")
            yield Button("[  ◀─  ]", id="back_button")
            yield Label("Hello", id="page_display")
            yield Button("[  ─▶  ]", id="next_button")
            yield Button("[  ─▶|  ]", id="last_button")

    def on_mount(self):
        self.border_title = "Pages"


class InputFilter(Static):
    def compose(self) -> ComposeResult:
        yield Container(
            Input(
                placeholder=f"Enter query (e.g. `COL1 > 42 &  COL2 == 3)`",
                id="input_prompt",
            )
        )

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    def __init__(self, df: pd.DataFrame, page_len: int = 100):
        super().__init__()
        self.df = df
        self.page_len = page_len
        self.shown_df = df
        self.page_no = 1  # starts from one
        self.page_tot = ceil(len(df) / page_len)

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
        self.page_tot = ceil(len(self.shown_df) / self.page_len)
        table = self.query_one(DataFrameTable)
        table.update_df(self.shown_df[self.page_slice()])
        self.update_page_display()


class MoreScreen(ModalScreen):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Container():
            yield TextArea.code_editor(self.text, read_only=True)
            yield Label("Press ESC to close.", id="close_label")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()


class HeaderDialog(Tree):
    def __init__(self, header: dict, hide_over: int = 20, *args, **kwargs):
        super().__init__(label="header", *args, **kwargs)
        self.truncated = {}
        self.guide_depth = 4
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


class Banner(Static):
    def compose(self) -> ComposeResult:
        yield Label(LOGO)


class Misfits(App):
    """Main app."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]
    CSS_PATH = "misfits.scss"

    def __init__(self, input_path: Path | str) -> None:
        super().__init__()
        self.input_path = input_path
        self.fits_content = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield TabbedContent()
        yield Footer()

    def on_mount(self):
        self.fits_content = self.populate_tabs(self.input_path)

    @work
    async def populate_tabs(self, input_path):
        tabs = self.query_one(TabbedContent)
        tabs.loading = True
        contents = await asyncio.to_thread(get_fits_content, input_path)
        for i, content in enumerate(contents):
            if content["header"]:
                await tabs.add_pane(
                    TabPane(f"Header-{i}", HeaderDialog(content["header"]))
                )
            if content["type"] == "table":
                await tabs.add_pane(TabPane(f"Table-{i}", TableDialog(content["data"])))
        tabs.loading = False


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
def main(input_path: Path):
    app = Misfits(input_path)
    app.run(inline=False)


if __name__ == "__main__":
    app = Misfits("/Users/peppedilillo/Dropbox/Progetti/fits-tui/fermi-fits.fits.gz")
    app.run()
