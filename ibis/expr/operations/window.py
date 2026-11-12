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
from ibis import util
from ibis.common.patterns import CoercionError
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.operations.analytic import Analytic  # noqa: TC001
from ibis.expr.operations.core import Value
from ibis.expr.operations.generic import Literal
from ibis.expr.operations.numeric import Negate
from ibis.expr.operations.reductions import Reduction  # noqa: TC001
from ibis.expr.operations.sortkeys import SortKey  # noqa: TC001
from ibis.expr.operations.temporal import IntervalAdd, IntervalSubtract

T = TypeVar("T", bound=dt.Numeric | dt.Interval, covariant=True)
S = TypeVar("S", bound=ds.DataShape, default=ds.Any, covariant=True)

# the shortest and longest possible length, in days, of calendar units
_CALENDAR_UNIT_DAYS = {"Y": (365, 366), "Q": (89, 92), "M": (28, 31)}


def _interval_bounds(op):
    """Bound the length, in nanoseconds, of an expression of interval literals.

    Returns `None` if `op` isn't built only from interval literals.
    """
    if isinstance(op, Literal) and op.dtype.is_interval():
        unit = op.dtype.unit.short
        if unit in _CALENDAR_UNIT_DAYS:
            shortest, longest = _CALENDAR_UNIT_DAYS[unit]
            low, high = sorted((op.value * shortest, op.value * longest))
            return util.convert_unit(low, "D", "ns"), util.convert_unit(high, "D", "ns")
        value = util.convert_unit(op.value, unit, "ns")
        return value, value
    elif isinstance(op, (IntervalAdd, IntervalSubtract)):
        left, right = _interval_bounds(op.left), _interval_bounds(op.right)
        if left is None or right is None:
            return None
        elif isinstance(op, IntervalAdd):
            return left[0] + right[0], left[1] + right[1]
        else:
            return left[0] - right[1], left[1] - right[0]
    return None


def _negate_interval_literals(op):
    if isinstance(op, Literal):
        return op.copy(value=-op.value)
    return op.copy(
        left=_negate_interval_literals(op.left),
        right=_negate_interval_literals(op.right),
    )


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
        elif (
            isinstance(arg, (IntervalAdd, IntervalSubtract))
            and (bounds := _interval_bounds(arg)) is not None
            and bounds[1] < 0
        ):
            # a compound interval that is always negative, such as
            # `ibis.interval(months=-1, days=1)`
            return cls(_negate_interval_literals(arg), preceding=True)
        elif isinstance(arg, Value):
            return cls(arg, preceding=False)
        else:
            raise CoercionError(f"Invalid window boundary type: {type(arg)}")


@public
class WindowFunction(Value):
    """A function evaluated over a window.

    Holds the function together with the window's frame bounds, grouping, and
    ordering, so it corresponds to all of `func(...) OVER (...)` in SQL, not
    just `func`. `func` is either an `Analytic`, which is only valid over a
    window, or a `Reduction`, which is evaluated over each row's window
    instead of over the whole table.
    """

    func: Analytic | Reduction
    how: LiteralType["rows", "range"] = "rows"
    start: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None
    end: Optional[WindowBoundary[dt.Numeric | dt.Interval]] = None
    group_by: VarTuple[Value] = ()
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
