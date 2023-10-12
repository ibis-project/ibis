from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from public import public
from typing_extensions import TypeVar

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.patterns import CoercionError
from ibis.common.typing import VarTuple  # noqa: TCH001
from ibis.expr.operations.analytic import Analytic  # noqa: TCH001
from ibis.expr.operations.core import Column, Value
from ibis.expr.operations.generic import Literal
from ibis.expr.operations.numeric import Negate
from ibis.expr.operations.reductions import Reduction  # noqa: TCH001
from ibis.expr.operations.relations import Relation  # noqa: TCH001
from ibis.expr.operations.sortkeys import SortKey  # noqa: TCH001

T = TypeVar("T", bound=dt.Numeric | dt.Interval, covariant=True)
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)


@public
class WindowBoundary(Value[T, S]):
    # TODO(kszucs): consider to prefer Concrete base class here
    # pretty similar to SortKey and Alias operations which wrap a single value
    value: Value[T, S]
    preceding: bool

    @property
    def following(self) -> bool:
        return not self.preceding

    @property
    def shape(self) -> S:
        return self.value.shape

    @property
    def dtype(self) -> T:
        return self.value.dtype

    @classmethod
    def __coerce__(cls, value, **kwargs):
        arg = super().__coerce__(value, **kwargs)

        if isinstance(arg, cls):
            return arg
        elif isinstance(arg, Negate):
            return cls(arg.arg, preceding=True)
        elif isinstance(arg, Literal):
            new = arg.copy(value=abs(arg.value))
            return cls(new, preceding=arg.value < 0)
        elif isinstance(arg, Value):
            return cls(arg, preceding=False)
        else:
            raise CoercionError(f"Invalid window boundary type: {type(arg)}")


@public
class WindowFrame(Value):
    """A window frame operation bound to a table."""

    table: Relation
    group_by: VarTuple[Column] = ()
    order_by: VarTuple[SortKey] = ()

    shape = ds.columnar

    def __init__(self, start, end, **kwargs):
        if start and end and start.dtype != end.dtype:
            raise com.IbisTypeError(
                "Window frame start and end boundaries must have the same datatype"
            )
        super().__init__(start=start, end=end, **kwargs)

    def dtype(self) -> dt.DataType:
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
    start: Optional[WindowBoundary[dt.Integer]] = None
    end: Optional[WindowBoundary] = None
    max_lookback: Optional[Value[dt.Interval]] = None

    def __init__(self, max_lookback, order_by, **kwargs):
        if max_lookback:
            # TODO(kszucs): this should belong to a timeseries extension rather than
            # the core window operation
            if len(order_by) != 1:
                raise com.IbisTypeError(
                    "`max_lookback` window must be ordered by a single column"
                )
            if not order_by[0].dtype.is_timestamp():
                raise com.IbisTypeError(
                    "`max_lookback` window must be ordered by a timestamp column"
                )
        super().__init__(max_lookback=max_lookback, order_by=order_by, **kwargs)


@public
class RangeWindowFrame(WindowFrame):
    how = "range"
    start: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None
    end: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None


@public
class WindowFunction(Value):
    func: Analytic | Reduction
    frame: WindowFrame

    dtype = rlz.dtype_like("func")
    shape = ds.columnar

    def __init__(self, func, frame):
        from ibis.expr.analysis import shares_all_roots

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
