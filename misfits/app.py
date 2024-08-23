from math import ceil
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.hdu.table import BinTableHDU, TableHDU
from astropy.table import Table
import click
import pandas as pd
from textual.app import App
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.validation import ValidationResult
from textual.validation import Validator
from textual.widgets import Button
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import Input
from textual.widgets import Static
from textual.widgets import TabbedContent, TabPane, Pretty


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


class DataframeMask(Validator):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.filtered_df = df

    def validate(self, value: str) -> ValidationResult:
        """Check a string is equal to its reverse."""
        try:
            self.filtered_df = self.df.query(value) if value else self.df
        except Exception as e:
            return self.failure()
        return self.success()

    def get_result(self) -> pd.DataFrame:
        return self.filtered_df

    @staticmethod
    def is_palindrome(value: str) -> bool:
        return value == value[::-1]


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
        """Add DataFrame data to DataTable."""
        if self.page_no < self.page_tot:
            self.page_no += 1
            table = self.query_one(DataFrameTable)
            table.update_df(self.shown_df[self.page_slice()])

    def back_page(self):
        """Add DataFrame data to DataTable."""
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
                    validate_on=["submitted"],
                    validators=[DataframeMask(self.df)],
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

    def on_input_submitted(self, event: Input.Submitted):
        if not event.validation_result.is_valid:
            return
        validator, *_ = event.input.validators
        table = self.query_one(DataFrameTable)
        self.shown_df = validator.filtered_df
        self.page_no = 1
        self.page_tot = ceil(len(self.shown_df) / self.page_len)
        table.update_df(self.shown_df[self.page_slice()])


class HeaderDialog(Static):
    def __init__(self, header: dict):
        super().__init__()
        self.header = header

    def compose(self):
        yield ScrollableContainer(
            Pretty(self.header)
        )


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
        self.fits_content = get_fits_content(input_path)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with TabbedContent():
            for i, content in enumerate(self.fits_content):
                if content["header"]:
                    with TabPane(f"Header-{i}"):
                        yield HeaderDialog(content["header"])
                if content["type"] == "table":
                    with TabPane(f"Table-{i}"):
                        yield TableDialog(content["data"])
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


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
