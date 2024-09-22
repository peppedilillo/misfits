"""
A terminal FITS viewer with interactive tables.

Author: Giuseppe Dilillo
Date:   August 2024
"""

from asyncio import sleep
from asyncio import to_thread
from math import ceil
from pathlib import Path
from typing import Iterable

from astropy.io.fits import FITS_rec
import click
from textual import on
from textual import work
from textual.app import App, SystemCommand
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
from textual.widgets.tabbed_content import ContentTabs
from textual.screen import Screen
from textual.reactive import reactive

from misfits.data import _validate_fits
from misfits.data import DataContainer
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

DARK_THEME = {
    "primary": "#03A062",  # matrix green
    "secondary": "#03A062",
    "warning": "#03A062",
    "error": "#ff0000",
    "success": "#00ff00",
    "accent": "#00ff00",
    "dark": True,
}

DEFAULT_COLORS["dark"] = ColorSystem(**DARK_THEME)

# fits table are displayed in small chunks (pages) to achieve better performances.
# this parameter set the number of rows displayed per page within a FitsTable.
PAGE_LEN = 100


class FitsTable(DataTable):
    """Displays fits data as a table arranged in pages."""

    BINDINGS = [
        ("shift+left", "back_page()", "Back"),
        ("shift+right", "next_page()", "Next"),
        ("shift+up", "first_page()", "First"),
        ("shift+down", "last_page()", "Last"),
    ]
    PAGE_DELAY = 1 / 60

    page_no = reactive(1, bindings=True)

    class QuerySucceded(Message):
        """A message to be sent when a query completes."""

        def __init__(self, query_succeded: bool) -> None:
            self.value = query_succeded
            super().__init__()

    def __init__(self, data: DataContainer):
        """
        :param data: a data container, see `misfits.data.DataContainer`.
        """
        super().__init__()
        self.data: DataContainer = data
        self.page_len = PAGE_LEN
        self.mask = None
        self.page_no = 1  # starts from one
        self.page_tot = max(ceil(len(self.data) / self.page_len), 1)
        log.push_data_info(data)

    def on_mount(self):
        self.border_title = "Table"
        self.cursor_type = "cell"
        self.add_columns(*self.data.get_columns())
        self.show_page()

    # this is unfortunate but necessary.
    # as per textual 0.77, `TabbedContent.clear_panes()` does not clear all references
    # to its tabs. since tabs are holding references to large tables this may cause
    # potentially huge memory leaks. best solution i've managed so far is to delete
    # references to tables on unmounting. i've tried different solutions such as
    # passing references to functions around, with no luck.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.data
        del self.mask

    # runs possibly slow filter operation with a worker to avoid UI lags
    @work(exclusive=True, group="filter_table")
    async def filter_table(self, query: str):
        """
        Filters a table according to a query and shows a table page.

        :param query: the filter query
        """
        # noinspection PyBroadException
        try:
            _ = await to_thread(self.data.query, query)
        except Exception:
            self.post_message(self.QuerySucceded(False))
            return
        self.page_tot = max(ceil(len(self.data) / self.page_len), 1)
        self.show_page()
        self.post_message(self.QuerySucceded(True))
        log.push(
            f"Filtered table by query {repr(query)}, {len(self.data)} matching entries."
        )

    def page_slice(self):
        """Returns a slice comprising entries to be displayed in present page."""
        page = ((self.page_no - 1) * self.page_len, self.page_no * self.page_len)
        return slice(*page)

    def show_page(self):
        """Displays a table page."""
        self.clear(columns=False)
        self.add_rows(rows=self.data.get_rows(self.page_slice()))
        self.border_subtitle = f"page {self.page_no} / {self.page_tot} "

    # the function is on a worker so that, if user keeps page turn pressed
    # page displays won't accumulate in queue, resulting in pages still getting
    # loaded after user releases the button.
    # sleep enforces a maximum turning rate to tot pages per seconds.
    # maybe we can live without that?
    # TODO: make sure `PAGE_DELAY` is needed
    @work(exclusive=True, group="turn_page")
    async def action_next_page(self):
        """Scrolls to next page."""
        await sleep(self.PAGE_DELAY)
        if self.page_no < self.page_tot:
            self.page_no += 1
            self.show_page()

    @work(exclusive=True, group="turn_page")
    async def action_back_page(self):
        """Scrolls to previous page."""
        await sleep(self.PAGE_DELAY)
        if self.page_no > 1:
            self.page_no -= 1
            self.show_page()

    @work(exclusive=True, group="turn_page")
    async def action_last_page(self):
        """Scrolls to last page"""
        await sleep(self.PAGE_DELAY)
        self.page_no = self.page_tot
        self.show_page()

    @work(exclusive=True, group="turn_page")
    async def action_first_page(self):
        """Scrolls to first page"""
        await sleep(self.PAGE_DELAY)
        self.page_no = 1
        self.show_page()

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Checks if an action may run."""
        if action in ["first_page", "back_page"] and self.page_no == 1:
            return None
        if action in ["last_page", "next_page"] and self.page_no == self.page_tot:
            return None
        return True

    # TODO: add methods and binding for scrolling to `n` page.


class FilterInput(Static):
    """A widget displaying an input prompt for filtering a table"""
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label("[dim italic] query: ")
            yield Input(placeholder=f"COL1 > 42 & COL2 == 3")

    def on_mount(self):
        self.border_title = "Filter"


class TableDialog(Static):
    """A widget containing the data table and its filter."""

    def __init__(
        self,
        arr: FITS_rec,
        hide_filter: bool = False,
    ):
        """
        :param arr: The fits records data.
        :param hide_filter: Wether if to show the table filter or none. We do not show
        filter for tables which would require huge loading time, such as tables with
        variable length array columns.
        """
        super().__init__()
        self.arr = arr
        self.hide_filter = hide_filter

    def compose(self) -> ComposeResult:
        data = DataContainer(self.arr)
        yield FitsTable(data)
        if data.can_promote:
            yield FilterInput()

    def on_mount(self):
        self.border_title = "Table"

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.arr

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
    """A placeholder widget for when an HDU contains images or no data."""

    # TODO: Add a separate placeholder for images.

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
        """Opens a pop-up when a header entry is selected."""
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
        """A message to be sent when loading tables we do not fully support."""

        def __init__(self, table_name) -> None:
            self.table_name = table_name
            super().__init__()

    def __init__(self, content: dict, **kwargs):
        self.content = content
        self._name = content["name"] if content["name"].strip() else "HDU"
        self.focused_already = False
        super().__init__(self._name, **kwargs)
        log.push_hdu_info(content)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield HeaderDialog(self.content["header"])
            if self.content["is_table"]:
                yield TableDialog(self.content["data"])
            else:
                yield EmptyDialog()

    # see note on `TableDialog.on_unmount`.
    # TODO: remove once textual resolves this bug.
    def on_unmount(self):
        del self.content
        del self._name

    @on(TabPane.Focused)
    def notify(self, _: TabPane.Focused) -> None:
        """This will alert main app to notify we are on a table with limitations."""
        if (
            not self.focused_already
            and self.content["is_table"]
            and not self.query_one(FitsTable).data.can_promote
        ):
            self.post_message(self.FocusedUnpromotableTable(self.content["name"]))
        self.focused_already = True


class FileInput(Static):
    """A widget showing an input for file paths."""
    BINDINGS = [
        ("ctrl+o", "open_explorer", "Open file explorer"),
    ]

    class RequestFileExplorer(Message):
        """A message for the main app, asking for the file explorer to be displayed."""

        def __init__(self) -> None:
            super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(f"[dim italic] path: ")
            yield Input()

    def on_mount(self):
        self.border_title = "File"

    def set_input_value(self, value: str):
        self.query_one(Input).value = value

    def action_open_explorer(self):
        self.post_message(self.RequestFileExplorer())


class Misfits(App):
    """Misfits, the main app."""

    CSS_PATH = "misfits.tcss"
    SCREENS = {
        "log": LogScreen,
        "file_explorer": FileExplorerScreen,
        "info": InfoScreen,
    }
    BINDINGS = [
        ("ctrl+l", "show_log", "Log"),
        ("ctrl+j", "show_info", "Info"),
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

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        # skips light mode toggle since, at present, it will mess with CSS and headers
        yield from (c for c in super().get_system_commands(screen) if c.title != "Light mode")
        yield SystemCommand("Show log", "Displays a log of misfits operations.", lambda: self.push_screen('log'))
        yield SystemCommand("More informations", "Displays information on misfits.", lambda: self.push_screen('info'))

    def action_show_log(self):
        self.push_screen('log')

    def action_show_info(self):
        self.push_screen('info')

    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        """Checks if an action may run."""
        if action in ["show_log", "show_info", "open_explorer"] and not isinstance(self.focused, ContentTabs):
            return False
        return True

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
    def notify_unpromotable(self, message: HDUPane.FocusedUnpromotableTable):
        self.notify(
            f"Table {message.table_name} contains array columns with variable length. "
            f"Unfortunately, these columns cannot be displayed. Filter has been disabled.",
            severity="warning",
            timeout=5,
        )

    @on(FileInput.RequestFileExplorer)
    # `push_screen_wait` requires a worker
    @work
    async def open_explorer(self):
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
    app = Misfits(None, Path("."))
    app.run()
