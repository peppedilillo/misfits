from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.widgets import DataTable, Input, Button
from textual.containers import Horizontal, Container
import pandas as pd
import click
from pathlib import Path


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



class MisfitsApp(App):
    """A Textual app to manage stopwatches."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]
    CSS_PATH = "misfits.scss"

    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path
        self.df = pd.read_csv(input_path)
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(
            DataFrameTable(),
            Container(
                Input("Input prompt", id="input_prompt"),
                Button("<", classes="arrows"),
                Button(">", classes="arrows"),
                id="control_bar",
            ),
            id="dialog",
        )
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataFrameTable)
        table.add_df(self.df)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
def main(input_path: Path):
    app = MisfitsApp(input_path)
    app.run()


if __name__ == "__main__":
    main()
