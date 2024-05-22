from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.expr.types.relations import unwrap_aliases

if TYPE_CHECKING:
    from collections.abc import Sequence


@public
class WindowedTable:
    """An intermediate table expression to hold windowing information."""

    def __init__(self, parent: ir.Table, time_col: ops.Column):
        if time_col is None:
            raise com.IbisInputError(
                "Window aggregations require `time_col` as an argument"
            )
        self.parent = parent
        self.time_col = time_col

    def tumble(
        self,
        size: ir.IntervalScalar,
        offset: ir.IntervalScalar | None = None,
    ) -> WindowedTable:
        self.window_type = "tumble"
        self.window_slide = None
        self.window_size = size
        self.window_offset = offset
        return self

    def hop(
        self,
        size: ir.IntervalScalar,
        slide: ir.IntervalScalar,
        offset: ir.IntervalScalar | None = None,
    ) -> WindowedTable:
        self.window_type = "hop"
        self.window_size = size
        self.window_slide = slide
        self.window_offset = offset
        return self

    def aggregate(
        self,
        metrics: Sequence[ir.Scalar] | None = (),
        by: Sequence[ir.Value] | None = (),
        **kwargs: ir.Value,
    ) -> ir.Table:
        groups = self.parent.bind(by)
        metrics = self.parent.bind(metrics, **kwargs)

        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)

        return ops.WindowAggregate(
            self.parent,
            self.window_type,
            self.time_col,
            groups=groups,
            metrics=metrics,
            window_size=self.window_size,
            window_slide=self.window_slide,
            window_offset=self.window_offset,
        ).to_expr()

    agg = aggregate
