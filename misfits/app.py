"""
A terminal FITS viewer with interactive tables.

Author: Giuseppe Dilillo
Date:   August 2024
"""

from asyncio import to_thread
from math import ceil
from pathlib import Path

from astropy.io import fits
from astropy.table import Table
import click
import pandas as pd
from textual import on
from textual import work
from textual.app import App
from textual.app import ComposeResult
from textual.app import DEFAULT_COLORS
from textual.containers import Horizontal
from textual.design import ColorSystem
from textual.message import Message
from textual.widgets import DataTable
from textual.widgets import Footer
from textual.widgets import Input
from textual.widgets import Label
from textual.widgets import Static
from textual.widgets import TabbedContent
from textual.widgets import TabPane
from textual.widgets import Tree

from misfits.data import _validate_fits
from misfits.data import get_fits_content
from misfits.headers import MainHeader
from misfits.log import log
from misfits.screens import EscapableFileExplorerScreen
from misfits.screens import FileExplorerScreen
from misfits.screens import HeaderEntry
from misfits.screens import InfoScreen
from misfits.screens import LogScreen
from misfits.utils import catchtime
from misfits.utils import disable_inputs

THEME = {
    "primary": "#03A062",  # matrix green
    "secondary": "#03A062",
    "warning": "#03A062",
    "error": "#ff0000",
    "success": "#00ff00",
    "accent": "#00ff00",
    "dark": True,
}

DEFAULT_COLORS["dark"] = ColorSystem(**THEME)


class FitsTable(DataTable):
    """Displays fits records as a table organized in pages."""

    BINDINGS = [
        ("ctrl+p", "back_page()", "Back"),
        ("ctrl+n", "next_page()", "Next"),
        ("ctrl+a", "first_page()", "First"),
        ("ctrl+e", "last_page()", "Last"),
    ]

    class QuerySucceded(Message):
        """Color selected message."""

        def __init__(self, query_succeded: bool) -> None:
            self.value = query_succeded
            super().__init__()

    def __init__(
        self, fits_records: fits.FITS_rec, cols: list[str], page_len: int = 50
    ):
        """
        :param fits_records: The dataframe to show
        :param cols: Columns to show in table
        :param page_len: How many dataframe rows are shown for each page.
        filter for tables which would require huge loading time, such as tables with
        variable length array columns.
        """
        super().__init__()
        self.table: fits.FITS_rec | pd.DataFrame = fits_records
        self.cols = cols
        self.page_len = page_len
        self.mask = None
        # table gets promoted when first converted to dataframe.
        # this enables the usage of pandas queries. promotion happens at first filter call.
        # a promoted table cannot be demoted.
        self.promoted = False
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(fits_records) / page_len), 1)

    def on_mount(self):
        self.border_title = "Table"
        self.cursor_type = "cell"
        self.update_page_display()

    # this is an unfortunate solution to an unfortunate problem.
    # as per textual 0.77, `TabbedContent.clear_panes()` does not clear all references
    # to its tabs. since tabs are holding references to large tables this may cause
    # potentially huge memory leaks. best solution i've managed so far is to delete
    # references to tables on unmounting. i've tried different solutions such as
    # passing references to functions around, with no luck.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.table
        del self.mask
        del self.cols
        del self.page_len
        del self.page_no
        del self.page_tot

    def display_table(self, rows: list[tuple], cols):
        """Update with a new table."""
        self.clear(columns=True)
        self.add_columns(*cols)
        self.add_rows(rows)

    # runs possibly slow filter operation with a worker to avoid UI lags
    @work(exclusive=True, group="filter_table")
    async def filter_table(self, query: str):
        """
        Filters a table according to a query and shows it's fist page.

        :param query: the filter query
        :return:
        """
        # noinspection PyBroadException
        if not self.promoted:
            self.table = Table(self.table).to_pandas()
            self.promoted = True
        try:
            fdf = await to_thread(self.table.query, query) if query else self.table
        except Exception:
            self.post_message(self.QuerySucceded(False))
            return
        self.mask = fdf.index
        self.page_no = 1
        self.page_tot = max(ceil(len(self.mask) / self.page_len), 1)
        self.update_page_display()
        self.post_message(self.QuerySucceded(True))
        log.push(f"Filtered table by query {repr(query)}, {len(fdf)} matching entries.")

    def page_slice(self):
        """Returns a slice which can be used to index the present page."""
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def update_page_display(self):
        """Displays the present table page."""
        if self.promoted:
            table_slice = self.table.iloc[self.mask[self.page_slice()]]
            self.display_table(
                rows=table_slice.itertuples(index=False),
                cols=self.cols,
            )
        else:
            table_slice = self.table[self.page_slice()]
            self.display_table(
                # this looks eccentric but is faster than list comprehension and
                # has the benefit of having homogenous formatting with promoted table
                rows=Table(table_slice)[self.cols].to_pandas().itertuples(index=False),
                cols=self.cols,
            )
        self.border_subtitle = f"page {self.page_no} / {self.page_tot} "

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


class FilterInput(Static):
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

    def __init__(
        self,
        arr: fits.FITS_rec,
        cols: list[str],
        page_len: int = 50,
        hide_filter: bool = False,
    ):
        """
        :param arr: The dataframe to show
        :param cols: Columns to show in table
        :param page_len: How many dataframe rows are shown for each page.
        :param hide_filter: Wether if to show the table filter or none. We do not show
        filter for tables which would require huge loading time, such as tables with
        variable length array columns.
        """
        super().__init__()
        self.page_len = page_len
        self.arr = arr
        self.cols = cols
        self.hide_filter = hide_filter

    def compose(self) -> ComposeResult:
        yield FitsTable(self.arr, self.cols, self.page_len)
        if not self.hide_filter:
            yield FilterInput()

    def on_mount(self):
        self.border_title = "Table"

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.arr
        del self.cols
        del self.page_len

    # async is needed since `filter_table` calls a worker
    @on(Input.Submitted)
    async def maybe_filter_table(self, event: Input.Submitted):
        # noinspection PyAsyncCall
        self.query_one(FitsTable).filter_table(event.value)
        # we prevent bubbling up of the message, which would affect file input prompt
        event.stop()

    @on(FitsTable.QuerySucceded)
    def color_filter_border(self, message: FitsTable.QuerySucceded):
        if message.value:
            self.query_one(FilterInput).remove_class("error")
        else:
            self.query_one(FilterInput).add_class("error")


class EmptyDialog(Static):
    """When a FITs HDU contains an image or no data, displays a placeholder."""

    def compose(self) -> ComposeResult:
        yield Label("No tables to show")

    def on_mount(self):
        self.border_title = "Table"


class HeaderDialog(Tree):
    """Displays a FITS header as a tree."""

    BINDINGS = [
        ("ctrl+s", "colexp_all", "Collapse/Expand all"),
    ]

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

    def on_mount(self):
        self.border_title = "Header"
        self.guide_depth = 3
        self.show_guides = True
        self.root.expand()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.leafs

    @on(Tree.NodeSelected)
    def display_content_popup(self, event: Tree.NodeSelected):
        """Opens a pop-up clicking on a header entry."""
        if event.node in self.leafs:
            self.app.push_screen(HeaderEntry(event.node.data))

    def action_colexp_all(self):
        """Collaps or expand all header nodes together."""
        if all(node.is_expanded for node in self.root.children):
            for c in self.root.children:
                c.collapse()
        else:  # some of the node is expanded already
            for node in self.root.children:
                if not node.is_expanded:
                    node.expand()


class HDUPane(TabPane):
    """A container for header and table widgets."""

    class FocusedUnpromotableTable(Message):
        """Color selected message."""

        def __init__(self, table_name) -> None:
            self.table_name = table_name
            super().__init__()

    def __init__(self, content: dict, **kwargs):
        self.content = content
        self.focused_already = False
        super().__init__(
            content["name"] if content["name"].strip() else "HDU", **kwargs
        )

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield HeaderDialog(self.content["header"])
            if self.content["is_table"]:
                yield TableDialog(
                    self.content["data"],
                    self.content["columns"],
                    hide_filter=True if self.content["columns_arrays"] else False,
                )
            else:
                yield EmptyDialog()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.content

    @on(TabPane.Focused)
    def notify(self, _: TabPane.Focused) -> None:
        """This will alert main app to notify we are on a table with limitations."""
        if not self.focused_already and self.content["columns_arrays"]:
            self.post_message(self.FocusedUnpromotableTable(self.content["name"]))
        self.focused_already = True


class FileInput(Static):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"[dim italic] path: ")
            yield Input()

    def on_mount(self):
        self.border_title = "File"

    def set_input_value(self, value: str):
        self.query_one(Input).value = value


class Misfits(App):
    """Misfits, the main app."""

    CSS_PATH = "misfits.tcss"
    SCREENS = {
        "log": LogScreen,
        "file_explorer": FileExplorerScreen,
        "info": InfoScreen,
    }
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
        yield MainHeader()
        yield TabbedContent()
        yield FileInput()
        yield Footer()

    # `push_screen_wait` requires a worker
    @work
    async def on_mount(self):
        if not self.filepath:
            self.filepath = await self.push_screen_wait(
                FileExplorerScreen(self.rootdir)
            )
        self.query_one(FileInput).set_input_value(str(self.filepath))
        # noinspection PyAsyncCall
        self.populate_tabs()

    # `populate_tabs` requires a worker
    @on(Input.Submitted)
    async def load_file_content(self, event: Input.Submitted):
        """Accepts and checks message from file input prompt."""
        input_path = Path(event.value)
        if not _validate_fits(input_path):
            self.query_one(FileInput).add_class("error")
            return
        self.query_one(FileInput).remove_class("error")
        self.filepath = input_path
        # noinspection PyAsyncCall
        self.populate_tabs()

    @on(HDUPane.FocusedUnpromotableTable)
    def notify_limitatiion(self, message: HDUPane.FocusedUnpromotableTable):
        self.notify(
            f"Table {message.table_name} contains array columns. "
            f"Array columns cannot be displayed. Filter has been disabled.",
            severity="warning",
            timeout=5,
        )

    # `push_screen_wait` requires a worker
    @work
    async def action_open_explorer(self):
        self.filepath = await self.push_screen_wait(
            EscapableFileExplorerScreen(self.rootdir)
        )
        # noinspection PyAsyncCall
        self.populate_tabs()
        self.query_one(FileInput).set_input_value(str(self.filepath))
        self.query_one(FileInput).remove_class("error")

    # calls CPU-heavy `get_fits_content`, requiring a worker
    # exclusive because otherwise would result in an error everytime we attempt
    # to open a new while one is still loading.
    @work(exclusive=True)
    async def populate_tabs(self) -> None:
        """
        Fills the tabs with data read from the FITS' HDUs.
        """
        async with disable_inputs(
            loading=self.query_one(TabbedContent),
            disabled=[self.query_one(FileInput)],
        ):
            with catchtime() as elapsed:
                tabs = self.query_one(TabbedContent)
                await tabs.clear_panes()
                log.push(f"Opening '{self.filepath}'")
                contents = await get_fits_content(self.filepath)
                for i, content in enumerate(contents):
                    await tabs.add_pane(HDUPane(content, id=(tab_id := f"tab-{i}")))
                    # switches to a pane if that pane contains a table
                    if content["is_table"]:
                        self.query_one(TabbedContent).active = tab_id
                    log.push_fitcontents(content)
            log.push(f"Reading FITS file took {elapsed():.3f} s")
        self.query_one(MainHeader).maybe_run_effect()


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
    app = Misfits(None, Path(r"D:/Dropbox/Progetti/"))
    app.run()
