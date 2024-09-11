import asyncio
from pathlib import Path
from typing import Iterable

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree
from textual.widgets import Footer
from textual.widgets import Header
from textual.widgets import RichLog
from textual.widgets import Static
from textual.widgets import TextArea

from misfits.data import _validate_fits
from misfits.effects import EffectLabel
from misfits.log import log
from misfits.logo import LOGO


class LogScreen(ModalScreen):
    """An alternative screen showing a log"""

    TITLE = "Log"
    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True)
        yield Footer()

    def on_screen_resume(self):
        """When screen is shown, pushes message on the stack to the screen."""
        while line := log.pop():
            self.query_one(RichLog).write(line)


class InfoScreen(ModalScreen):
    """Shows an information screen with cool effects."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]

    @staticmethod
    def get_text():
        return Text.from_markup(
            f"A FITS table viewer by G.D.\n"
            f"[dim]https://github.com/peppedilillo\n"
            f"https://gdilillo.com",
            justify="center",
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield EffectLabel(
                LOGO,
                effect="BinaryPath",
                config={
                    "active_binary_groups": 0.1,
                    "movement_speed": 1.0,
                }
            )
            yield Static(self.get_text())
        yield Footer()


class FileExplorerScreen(ModalScreen):
    """A pop-up screen showing a file explorer so that the user may choose an
    input navigating the file system. To be used at main app's start-up,
     if no input file is provided. For this reason the screen is not escapable."""

    TITLE = "Open file"

    def __init__(self, rootdir: Path = Path.cwd()):
        super().__init__()
        self.rootdir = rootdir

    def compose(self) -> ComposeResult:
        with Container():
            yield Header(show_clock=False)
            yield FilteredDirectoryTree(self.rootdir)
        yield Footer()

    # threaded because validating requires IO which may cause laggish behaviour
    @work(exclusive=True, group="file_check")
    async def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected):
        isvalid = await asyncio.to_thread(_validate_fits, event.path)
        if not isvalid:
            self.query_one(DirectoryTree).add_class("error")
            return
        self.query_one(DirectoryTree).remove_class("error")
        # noinspection PyAsyncCall
        self.dismiss(event.path)


class EscapableFileExplorerScreen(FileExplorerScreen):
    """Like `FileExplorer` but with bindings to leave the screen.
    To be used when a file input has already been provided."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]


class FilteredDirectoryTree(DirectoryTree):
    """A directory tree widget filtering hidden files."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [path for path in paths if not path.name.startswith(".")]


class HeaderEntry(ModalScreen):
    """Displays header's entries in a pop-up screen. Useful with long entries."""

    BINDINGS = [("escape", "app.pop_screen", "Return to dashboard")]
    TITLE = "Header entry"

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Container():
            yield Header()
            yield TextArea.code_editor(self.text, read_only=True)
        yield Footer()
