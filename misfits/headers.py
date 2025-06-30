import asyncio

from rich.text import Text
from terminaltexteffects import Color
from terminaltexteffects import Gradient
from terminaltexteffects.effects.effect_binarypath import BinaryPath
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label
from textual.widgets import Static

from misfits import __version__


class Header(Static):
    def __init__(
        self,
        *,
        left_label: Label | str | None = None,
        mid_label: Label | str | None = None,
        right_label: Label | str | None = None,
    ):
        self.left_label = (
            Label(left_label) if isinstance(left_label, str) else left_label
        )
        self.mid_label = Label(mid_label) if isinstance(mid_label, str) else mid_label
        self.right_label = (
            Label(right_label) if isinstance(right_label, str) else right_label
        )
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


class AnimatedLabel(Static):
    def __init__(self, text: str):
        super().__init__()
        self.text = text
        effect = BinaryPath(self.text)
        effect.effect_config.final_gradient_stops = Color("FFFFFF")
        effect.effect_config.final_gradient_steps = 8
        effect.terminal_config.canvas_height = 1
        effect.terminal_config.canvas_width = len(self.text)
        self.effect = effect

    @work
    async def play(self) -> None:
        for frame in self.effect:
            self.update(Text.from_ansi(frame))
            await asyncio.sleep(0.0)


class MainHeader(Header):
    def __init__(self):
        self.has_run_before = False
        super().__init__(
            left_label=AnimatedLabel(" misfits"),
            right_label=Label(Text.from_markup(f"[italic dim]v.{__version__} ")),
        )
