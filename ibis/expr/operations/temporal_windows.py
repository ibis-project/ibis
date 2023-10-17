from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datatypes as dt
from ibis.expr.operations.core import Column, Scalar  # noqa: TCH001
from ibis.expr.operations.relations import Relation
from ibis.expr.schema import Schema


@public
class WindowingTVF(Relation):
    """Generic windowing table-valued function."""

    table: Relation
    time_col: Column[dt.Timestamp]  # enforce timestamp column type here

    @property
    def schema(self):
        names = list(self.table.schema.names)
        types = list(self.table.schema.types)

        # The return value of windowing TVF is a new relation that includes all columns
        # of original relation as well as additional 3 columns named “window_start”,
        # “window_end”, “window_time” to indicate the assigned window

        names.extend(["window_start", "window_end", "window_time"])
        # window_start, window_end, window_time have type TIMESTAMP(3) in Flink
        types.extend([dt.timestamp(scale=3)] * 3)

        return Schema.from_tuples(list(zip(names, types)))


@public
class TumbleWindowingTVF(WindowingTVF):
    """TUMBLE window table-valued function."""

    window_size: Scalar[dt.Interval]
    offset: Optional[Scalar[dt.Interval]] = None


@public
class HopWindowingTVF(WindowingTVF):
    """HOP window table-valued function."""

    window_size: Scalar[dt.Interval]
    window_slide: Scalar[dt.Interval]
    offset: Optional[Scalar[dt.Interval]] = None


@public
class CumulateWindowingTVF(WindowingTVF):
    """CUMULATE window table-valued function."""

    window_size: Scalar[dt.Interval]
    window_step: Scalar[dt.Interval]
    offset: Optional[Scalar[dt.Interval]] = None
