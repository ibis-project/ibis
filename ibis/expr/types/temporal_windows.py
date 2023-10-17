from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.deferred import Deferred
from ibis.selectors import Selector

if TYPE_CHECKING:
    from ibis.expr.types import Table


def _get_window_by_key(table, value):
    if isinstance(value, str):
        return table[value]
    elif isinstance(value, Deferred):
        return value.resolve(table)
    elif isinstance(value, Selector):
        matches = value.expand(table)
        if len(matches) != 1:
            raise com.IbisInputError(
                "Multiple columns match the selector; only 1 is expected"
            )
        return next(iter(matches))
    elif isinstance(value, ir.Expr):
        return an.sub_immediate_parents(value.op(), table.op()).to_expr()
    else:
        return value


@public
class WindowedTable:
    """An intermediate table expression to hold windowing information."""

    def __init__(self, table: ir.Table, time_col: ir.Value):
        self.table = table
        self.time_col = _get_window_by_key(table, time_col)

        if self.time_col is None:
            raise com.IbisInputError(
                "Window aggregations require `time_col` as an argument"
            )

    def tumble(
        self,
        window_size: ir.IntervalScalar,
        offset: ir.IntervalScalar | None = None,
    ) -> Table:
        """Compute a tumble table valued function.

        Tumbling windows have a fixed size and do not overlap. The size of the windows is
        determined by `window_size`, optionally shifted by a duration specified by `offset`.

        Parameters
        ----------
        window_size
            Width of the tumbling windows.
        offset
            An optional parameter to specify the offset which window start should be shifted by.

        Returns
        -------
        Table
            Table expression after applying tumbling table-valued function.
        """
        return ops.TumbleWindowingTVF(
            table=self.table,
            time_col=_get_window_by_key(self.table, self.time_col),
            window_size=window_size,
            offset=offset,
        ).to_expr()

    def hop(
        self,
        window_size: ir.IntervalScalar,
        window_slide: ir.IntervalScalar,
        offset: ir.IntervalScalar | None = None,
    ):
        """Compute a hop table valued function.

        Hopping windows have a fixed size and can be overlapping if the slide is smaller than the
        window size (in which case elements can be assigned to multiple windows). Hopping windows
        are also known as sliding windows. The size of the windows is determined by `window_size`,
        how frequently a hopping window is started is determined by `window_slide`, and windows can
        be optionally shifted by a duration specified by `offset`.

        For example, you could have windows of size 10 minutes that slides by 5 minutes. With this,
        you get every 5 minutes a window that contains the events that arrived during the last 10 minutes.

        Parameters
        ----------
        window_size
            Width of the hopping windows.
        window_slide
            The duration between the start of sequential hopping windows.
        offset
            An optional parameter to specify the offset which window start should be shifted by.

        Returns
        -------
        Table
            Table expression after applying hopping table-valued function.
        """
        return ops.HopWindowingTVF(
            table=self.table,
            time_col=_get_window_by_key(self.table, self.time_col),
            window_size=window_size,
            window_slide=window_slide,
            offset=offset,
        ).to_expr()

    def cumulate(
        self,
        window_size: ir.IntervalScalar,
        window_step: ir.IntervalScalar,
        offset: ir.IntervalScalar | None = None,
    ):
        """Compute a cumulate table valued function.

        Cumulate windows don't have a fixed size and do overlap. Cumulate windows assign elements to windows
        that cover rows within an initial interval of step size and expand to one more step size (keep window
        start fixed) every step until the max window size.

        For example, you could have a cumulating window for 1 hour step and 1 day max size, and you will get
        windows: [00:00, 01:00), [00:00, 02:00), [00:00, 03:00), …, [00:00, 24:00) for every day.

        Parameters
        ----------
        window_size
            Max width of the cumulating windows.
        window_step
            A duration specifying the increased window size between the end of sequential cumulating windows.
        offset
            An optional parameter to specify the offset which window start should be shifted by.

        Returns
        -------
        Table
            Table expression after applying cumulate table-valued function.
        """
        return ops.CumulateWindowingTVF(
            table=self.table,
            time_col=_get_window_by_key(self.table, self.time_col),
            window_size=window_size,
            window_step=window_step,
            offset=offset,
        ).to_expr()
