"""The public API for rendering Ibis expressions with rich.

Does not require rich to be installed to import this module, but DOES
require rich to be installed to use any of the functions in this module.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from ibis.expr import types as ir

if TYPE_CHECKING:
    import rich.panel
    import rich.table
    from rich.console import RenderableType


try:
    from rich.jupyter import JupyterMixin
except ImportError:

    class FixedTextJupyterMixin:
        """No-op when rich is not installed."""
else:

    class FixedTextJupyterMixin(JupyterMixin):
        """JupyterMixin adds a spurious newline to text, this fixes the issue."""

        def _repr_mimebundle_(self, *args, **kwargs):
            try:
                with _with_rich_display_disabled():
                    bundle = super()._repr_mimebundle_(*args, **kwargs)
            except Exception:  # noqa: BLE001
                return None
            else:
                bundle["text/plain"] = bundle["text/plain"].rstrip()
                return bundle


def capture_rich_renderable(renderable: RenderableType) -> str:
    """Convert a rich renderable (has a __rich_console__(), etc) to a string."""
    from rich.console import Console

    console = Console(force_terminal=False)
    with _with_rich_display_disabled(), console.capture() as capture:
        console.print(renderable)
    return capture.get().rstrip()


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

        original_display = ipython_display.display
        try:
            ipython_display.display = noop_display
            yield
        finally:
            ipython_display.display = original_display
