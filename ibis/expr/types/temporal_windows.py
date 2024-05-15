from __future__ import annotations

from typing import TYPE_CHECKING

from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.grounds import Concrete
from ibis.expr.types.relations import unwrap_aliases

if TYPE_CHECKING:
    from collections.abc import Sequence


@public
class WindowedTable(Concrete):
    """An intermediate table expression to hold windowing information."""

    table: ops.Relation
    time_col: ops.Column

    def __init__(self, time_col: ops.Column, **kwargs):
        if not time_col:
            raise com.IbisInputError("No time column provided")
        super().__init__(time_col=time_col, **kwargs)


@public
class TumbleTable(WindowedTable):
    window_size: ir.IntervalScalar
    offset: ir.IntervalScalar | None = None

    def aggregate(
        self,
        metrics: Sequence[ir.Scalar] | None = (),
        by: Sequence[ir.Value] | None = (),
        **kwargs: ir.Value,
    ) -> ir.Table:
        table = self.table.to_expr()
        groups = table.bind(by)
        metrics = table.bind(metrics, **kwargs)

        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)

        return ops.WindowAggregate(
            self.table,
            "tumble",
            self.time_col,
            groups=groups,
            metrics=metrics,
            window_size=self.window_size,
            offset=self.offset,
        ).to_expr()

    agg = aggregate


@public
class HopTable(WindowedTable):
    window_size: ir.IntervalScalar
    window_slide: ir.IntervalScalar
    offset: ir.IntervalScalar | None = None

    def aggregate(
        self,
        metrics: Sequence[ir.Scalar] | None = (),
        by: Sequence[ir.Value] | None = (),
        **kwargs: ir.Value,
    ) -> ir.Table:
        table = self.table.to_expr()
        groups = table.bind(by)
        metrics = table.bind(metrics, **kwargs)

        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)

        return ops.WindowAggregate(
            self.table,
            "hop",
            self.time_col,
            groups=groups,
            metrics=metrics,
            window_size=self.window_size,
            offset=self.offset,
        ).to_expr()

    agg = aggregate
