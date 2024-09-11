from asyncio import sleep
from contextlib import contextmanager, asynccontextmanager
from time import perf_counter
from typing import Callable

from textual.widget import Widget


@contextmanager
def catchtime() -> Callable[[], float]:
    """A context manager for measuring computing times."""
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()


@asynccontextmanager
async def disable_inputs(loading: Widget, disabled: list[Widget], delay: float = 0.25):
    """
    Disables input and shows a loading animation while tables are read into memory.

    :param disabled:
    :param loading:
    :param delay: seconds delay between end of loading indicator and
    file input prompt release.
    :return:
    """
    for widget in disabled:
        widget.disabled = True
    loading.loading = True
    yield
    loading.loading = False
    # we wait a bit before releasing the input because quick, repeated sends can
    # cause a tab to not load properly
    await sleep(delay)
    for widget in disabled:
        widget.disabled = False
