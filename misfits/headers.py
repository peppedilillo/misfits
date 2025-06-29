from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label
from textual.widgets import Static

from misfits import __version__


def labelize(arg: Label | str | None):
    if isinstance(arg, Label):
        return arg
    elif isinstance(arg, str):
        return Label(arg)
    elif arg is None:
        return None
    else:
        raise ValueError()


class Header(Static):
    def __init__(
        self,
        *,
        left_label: Label | str | None = None,
        mid_label: Label | str | None = None,
        right_label: Label | str | None = None,
    ):
        self.left_label = labelize(left_label)
        self.mid_label = labelize(mid_label)
        self.right_label = labelize(right_label)
        super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal():
            if self.left_label:
                yield self.left_label
            yield Static()
            if self.mid_label:
                yield self.mid_label
            yield Static()
            if self.right_label:
                yield self.right_label


class MainHeader(Header):
    def __init__(self):
        self.has_run_before = False
        super().__init__(
            left_label=" misfits",
            right_label=Label(Text.from_markup(f"[italic dim]v.{__version__} ")),
        )
