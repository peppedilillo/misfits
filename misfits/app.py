from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.widgets import DataTable, Input, Button
from textual.containers import Container
from textual.validation import ValidationResult, Validator


import pandas as pd
import click
from pathlib import Path

from math import ceil


class DataFrameTable(DataTable):
    """Display Pandas dataframe in DataTable widget."""

    DEFAULT_CSS = """
    DataFrameTable {
        height: 1fr
    }
    """

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

    def _add_df_rows(self) -> None:
        return self._get_df_rows()

    def _add_df_columns(self) -> None:
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
        page = (
            (self.page_no - 1) * self.page_len,
            self.page_no * self.page_len
        )
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
            id="dialog",
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


class Misfits(App):
    """Main app."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]
    CSS_PATH = "misfits.scss"

    def __init__(self, input_path: Path | str) -> None:
        super().__init__()
        self.input_path = input_path
        self.df = pd.read_csv(input_path)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield TableDialog(self.df)
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
    app = Misfits("/Users/peppedilillo/Dropbox/Progetti/fits-tui/BTC-2017min.csv")
    app.run()
