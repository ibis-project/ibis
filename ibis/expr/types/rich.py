"""The public API for rendering Ibis expressions with rich.

Does not require rich to be installed to import this module, but DOES
require rich to be installed to use any of the functions in this module.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ibis.expr import types as ir

if TYPE_CHECKING:
    from collections.abc import Sequence

    import rich.panel
    import rich.table
    from rich.console import RenderableType


try:
    import rich
except ImportError:

    class RichJupyterMixin:
        """No-op when rich is not installed."""
else:

    class RichJupyterMixin:
        """Adds `_repr_mimebundle_()` to anything with a `__rich_console__()`."""

        def _repr_mimebundle_(
            self, include: Sequence[str], exclude: Sequence[str], **kwargs
        ) -> dict[str, str]:
            bundle = capture_rich_renderable(self, no_color=False)
            return {k: bundle[k] for k in (bundle.keys() & include).difference(exclude)}


def capture_rich_renderable(
    renderable: RenderableType, *, no_color: bool
) -> dict[str, str]:
    """Convert a rich renderable (has a __rich_console__(), etc) to text and html representations."""
    from rich.console import Console

    color_system = None if no_color else "auto"
    console = Console()
    width = console.width
    if console.is_jupyter:
        width = 1_000_000
    with _with_rich_configured(
        width=width, color_system=color_system, force_terminal=False
    ):
        return _RichMimeBundler(renderable).get_mimebundle()


def to_rich(
    expr: ir.Scalar | ir.Table | ir.Column,
    *,
    max_rows: int | None = None,
    max_columns: int | None = None,
    max_length: int | None = None,
    max_string: int | None = None,
    max_depth: int | None = None,
    console_width: int | float | None = None,
) -> rich.panel.Panel | rich.table.Table:
    """Truncate, evaluate, and render an Ibis expression as a rich object."""
    from ibis.expr.types._rich import to_rich_scalar, to_rich_table

    if isinstance(expr, ir.Scalar):
        return to_rich_scalar(
            expr, max_length=max_length, max_string=max_string, max_depth=max_depth
        )
    else:
        return to_rich_table(
            expr,
            max_rows=max_rows,
            max_columns=max_columns,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            console_width=console_width,
        )


@contextlib.contextmanager
def _with_rich_configured(**config):
    """Context manager to temporarily configure rich."""
    from rich.console import Console

    global_console = rich.get_console()
    new_console = Console(**config)
    original_config = global_console.__dict__

    try:
        global_console.__dict__ = new_console.__dict__
        yield
    finally:
        global_console.__dict__ = original_config


try:
    from rich.jupyter import JupyterMixin
except ImportError:
    JupyterMixin = object


class _RichMimeBundler(JupyterMixin):
    def __init__(self, renderable: RenderableType):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        yield self.renderable

    def get_mimebundle(self) -> dict[str, str]:
        with _with_rich_display_disabled():
            bundle = super()._repr_mimebundle_(include=None, exclude=None)
        bundle["text/plain"] = bundle["text/plain"].rstrip()  # Remove trailing newline
        return bundle


@contextlib.contextmanager
def _with_rich_display_disabled():
    """Workaround to keep rich from doing spurious display() calls in Jupyter.

    When you display(ibis.Table), without this, an extra output cell is created
    in the notebook. With this, there is no extra output cell.

    See https://github.com/Textualize/rich/pull/3329
    """
    try:
        from IPython import display as ipython_display
    except ImportError:
        # IPython is not installed, so nothing to do
        yield
    else:

        def noop_display(*args, **kwargs):
            pass
            # ipython_display.display(*args, **kwargs)

        original_display = ipython_display.display
        try:
            ipython_display.display = noop_display
            yield
        finally:
            ipython_display.display = original_display
