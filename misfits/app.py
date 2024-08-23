import asyncio
from math import ceil
from pathlib import Path

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
from textual.containers import ScrollableContainer
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Pretty
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane


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

    def next_page(self):
        if self.page_no < self.page_tot:
            self.page_no += 1
            table = self.query_one(DataFrameTable)
            table.update_df(self.shown_df[self.page_slice()])

    def back_page(self):
        if self.page_no > 1:
            self.page_no -= 1
            table = self.query_one(DataFrameTable)
            table.update_df(self.shown_df[self.page_slice()])

    def compose(self) -> ComposeResult:
        yield Container(
            DataFrameTable(),
            Container(
                Input(
                    placeholder=f"Example: {self.df.columns[-1]} > 42",
                    id="input_prompt",
                ),
                Button("[ ◀─ ]", id="back_button"),
                Button("[ ─▶ ]", id="next_button"),
                id="control_bar",
            ),
        )

    def on_mount(self):
        table = self.query_one(DataFrameTable)
        table.add_df(self.shown_df[self.page_slice()])

    def on_button_pressed(self, event: Button.Pressed):
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        if button_id == "next_button":
            self.next_page()
        elif button_id == "back_button":
            self.back_page()

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


class HeaderDialog(Static):
    def __init__(self, header: dict):
        super().__init__()
        self.header = header

    def compose(self):
        yield ScrollableContainer(Pretty(self.header))


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

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

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
