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

_NS_PER_DAY = 24 * 60 * 60 * 1_000_000_000
_MONTH_UNITS = frozenset({"Y", "Q", "M"})


def _interval_components(value):
    """Return exact month and nanosecond components for a literal interval tree."""
    if isinstance(value, Literal) and value.dtype.is_interval():
        if value.value is None:
            return None
        unit = value.dtype.unit.short
        if unit in _MONTH_UNITS:
            return util.convert_unit(value.value, unit, "M"), 0
        return 0, util.convert_unit(value.value, unit, "ns")
    elif isinstance(value, IntervalAdd):
        left = _interval_components(value.left)
        right = _interval_components(value.right)
        if left is not None and right is not None:
            return left[0] + right[0], left[1] + right[1]
    elif isinstance(value, IntervalSubtract):
        left = _interval_components(value.left)
        right = _interval_components(value.right)
        if left is not None and right is not None:
            return left[0] - right[0], left[1] - right[1]
    elif isinstance(value, Negate):
        if (components := _interval_components(value.arg)) is not None:
            return -components[0], -components[1]

    return None


def _rewrite_interval(value, sign=1):
    """Distribute negation to interval literals for portable SQL generation."""
    if isinstance(value, Literal):
        return value.copy(value=sign * value.value)
    elif isinstance(value, IntervalAdd):
        return IntervalAdd(
            _rewrite_interval(value.left, sign),
            _rewrite_interval(value.right, sign),
        )
    elif isinstance(value, IntervalSubtract):
        return IntervalSubtract(
            _rewrite_interval(value.left, sign),
            _rewrite_interval(value.right, sign),
        )
    elif isinstance(value, Negate):
        return _rewrite_interval(value.arg, -sign)
    else:
        raise AssertionError(f"Unsupported literal interval operation: {type(value)}")


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
        elif isinstance(arg, Literal):
            new = arg.copy(value=abs(arg.value))
            return cls(new, preceding=arg.value < 0)
        elif (components := _interval_components(arg)) is not None:
            months, nanoseconds = components
            month_bounds = sorted(
                (months * 28 * _NS_PER_DAY, months * 31 * _NS_PER_DAY)
            )
            lower = month_bounds[0] + nanoseconds
            upper = month_bounds[1] + nanoseconds
            if lower < 0 and upper <= 0:
                return cls(_rewrite_interval(arg, sign=-1), preceding=True)
            return cls(_rewrite_interval(arg), preceding=False)
        elif isinstance(arg, Negate):
            return cls(arg.arg, preceding=True)
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
