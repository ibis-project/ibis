from __future__ import annotations

from typing import Optional

from public import public

import ibis.expr.datatypes as dt
from ibis.common.annotations import attribute
from ibis.expr.operations.core import Column, Scalar  # noqa: TCH001
from ibis.expr.operations.relations import Relation
from ibis.expr.schema import Schema


@public
class WindowingTVF(Relation):
    """Generic windowing table-valued function.

    Table-valued functions return tables.

    Windowing TVFs in Ibis return the original columns plus `window_start`,
    `window_end`, and `window_time`.
    """

    parent: Relation
    time_col: Column[dt.Timestamp]  # enforce timestamp column type here

    @attribute
    def values(self):
        return self.parent.fields

    @property
    def schema(self):
        names = list(self.parent.schema.names)
        types = list(self.parent.schema.types)

        names.extend(["window_start", "window_end", "window_time"])
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
