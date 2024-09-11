from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Label
from terminaltexteffects.utils.graphics import Color

from misfits.effects import EffectLabel
from misfits import __version__


def labelize(arg: Label | EffectLabel | str | None):
    if isinstance(arg, Label) or isinstance(arg, EffectLabel):
        return arg
    elif isinstance(arg, str):
        return Label(arg)
    elif arg is None:
        return None
    else:
        raise ValueError()


class Header(Static):
    def __init__(
        self, *,
        left_label: Label | EffectLabel | str | None = None,
        mid_label: Label | EffectLabel | str | None = None,
        right_label: Label | EffectLabel | str | None = None,
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


MainHeader = Header(
    left_label=EffectLabel(
        text="  misfits",
        effect="BinaryPath",
        config={"final_gradient_stops": (Color("#ffffff"),)},
    ),
    right_label=Label(Text.from_markup(f"[italic dim]v.{__version__} "))
)
