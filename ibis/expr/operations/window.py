from __future__ import annotations

from abc import abstractmethod

from public import public

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations import Value


@public
class WindowBoundary(Value):
    # TODO(kszucs): consider to prefer Concrete base class here
    # pretty similar to SortKey and Alias operations which wrap a single value
    value = rlz.one_of([rlz.numeric, rlz.interval])
    preceding = rlz.bool_

    output_shape = rlz.shape_like("value")
    output_dtype = rlz.dtype_like("value")

    @property
    def following(self) -> bool:
        return not self.preceding


@public
class WindowFrame(Value):
    """A window frame operation bound to a table."""

    table = rlz.table
    group_by = rlz.optional(rlz.tuple_of(rlz.any), default=())
    order_by = rlz.optional(
        rlz.tuple_of(rlz.sort_key_from(rlz.ref("table"))), default=()
    )

    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, start, end, **kwargs):
        if start and end and start.output_dtype != end.output_dtype:
            raise com.IbisTypeError(
                "Window frame start and end boundaries must have the same datatype"
            )
        super().__init__(start=start, end=end, **kwargs)

    def output_dtype(self) -> dt.DataType:
        return dt.Array(dt.Struct.from_tuples(self.table.schema.items()))

    @property
    @abstractmethod
    def start(self):
        ...

    @property
    @abstractmethod
    def end(self):
        ...


@public
class RowsWindowFrame(WindowFrame):
    how = "rows"
    start = rlz.optional(rlz.row_window_boundary)
    end = rlz.optional(rlz.row_window_boundary)
    max_lookback = rlz.optional(rlz.interval)

    def __init__(self, max_lookback, order_by, **kwargs):
        if max_lookback:
            # TODO(kszucs): this should belong to a timeseries extension rather than
            # the core window operation
            if len(order_by) != 1:
                raise com.IbisTypeError(
                    "`max_lookback` window must be ordered by a single column"
                )
            if not order_by[0].output_dtype.is_timestamp():
                raise com.IbisTypeError(
                    "`max_lookback` window must be ordered by a timestamp column"
                )
        super().__init__(max_lookback=max_lookback, order_by=order_by, **kwargs)


@public
class RangeWindowFrame(WindowFrame):
    how = "range"
    start = rlz.optional(rlz.range_window_boundary)
    end = rlz.optional(rlz.range_window_boundary)


@public
class WindowFunction(Value):
    func = rlz.analytic
    frame = rlz.instance_of(WindowFrame)

    output_dtype = rlz.dtype_like("func")
    output_shape = rlz.Shape.COLUMNAR

    def __init__(self, func, frame):
        from ibis.expr.analysis import propagate_down_window, shares_all_roots

        func = propagate_down_window(func, frame)
        if not shares_all_roots(func, frame):
            raise com.RelationError(
                "Window function expressions doesn't fully originate from the "
                "dependencies of the window expression."
            )
        super().__init__(func=func, frame=frame)

    @property
    def name(self):
        return self.func.name


public(WindowOp=WindowFunction, Window=WindowFunction)
