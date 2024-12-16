"""Window operations."""

from __future__ import annotations

from typing import Literal as LiteralType
from typing import Optional

from public import public
from typing_extensions import TypeVar

import ibis.common.exceptions as com
import ibis.expr.datashape as ds
import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.common.patterns import CoercionError
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.analytic import Analytic  # noqa: TC001
from ibis.expr.operations.core import Column, Value
from ibis.expr.operations.generic import Literal
from ibis.expr.operations.numeric import Negate
from ibis.expr.operations.reductions import Reduction  # noqa: TC001
from ibis.expr.operations.sortkeys import SortKey  # noqa: TC001

T = TypeVar("T", bound=dt.Numeric | dt.Interval, covariant=True)
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)


@public
class WindowBoundary(Value[T, S]):
    """Window boundary object."""

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
class WindowFunction(Value):
    """Window function operation."""

    func: Analytic | Reduction
    how: LiteralType["rows", "range"] = "rows"
    start: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None
    end: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None
    group_by: VarTuple[Column] = ()
    order_by: VarTuple[SortKey] = ()

    dtype = rlz.dtype_like("func")
    shape = ds.columnar

    def __init__(self, how, start, end, **kwargs):
        if how == "rows":
            if start and not start.dtype.is_integer():
                raise com.IbisTypeError(
                    "Row-based window frame start boundary must be an integer"
                )
            if end and not end.dtype.is_integer():
                raise com.IbisTypeError(
                    "Row-based window frame end boundary must be an integer"
                )
        elif how == "range":
            if (
                start
                and end
                and not (
                    (start.dtype.is_interval() and end.dtype.is_interval())
                    or (start.dtype.is_numeric() and end.dtype.is_numeric())
                )
            ):
                raise com.IbisTypeError(
                    "Window frame start and end boundaries must have the same datatype"
                )
        else:
            raise com.IbisTypeError(
                f"Window frame type must be either 'rows' or 'range', got {how}"
            )
        super().__init__(how=how, start=start, end=end, **kwargs)

    @property
    def name(self):
        return self.func.name


public(WindowOp=WindowFunction, Window=WindowFunction)
