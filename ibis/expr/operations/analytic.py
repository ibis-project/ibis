from public import public

from .. import datatypes as dt
from .. import rules as rlz
from .. import types as ir
from ..window import propagate_down_window
from .core import ValueOp, distinct_roots


@public
class AnalyticOp(ValueOp):
    pass


@public
class WindowOp(ValueOp):
    expr = rlz.analytic
    window = rlz.window(from_base_table_of="expr")
    output_type = rlz.array_like('expr')

    display_argnames = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(
            self,
            "expr",
            propagate_down_window(self.expr, self.window),
        )

    def over(self, window):
        new_window = self.window.combine(window)
        return WindowOp(self.expr, new_window)

    @property
    def inputs(self):
        return self.expr.op().inputs[0], self.window

    def root_tables(self):
        return distinct_roots(
            self.expr, *self.window._order_by, *self.window._group_by
        )


@public
class ShiftBase(AnalyticOp):
    arg = rlz.column(rlz.any)
    offset = rlz.optional(rlz.one_of((rlz.integer, rlz.interval)))
    default = rlz.optional(rlz.any)
    output_type = rlz.typeof('arg')


@public
class Lag(ShiftBase):
    pass


@public
class Lead(ShiftBase):
    pass


@public
class RankBase(AnalyticOp):
    def output_type(self):
        return dt.int64.column_type()


@public
class MinRank(RankBase):
    """Compute position of first element within each equal-value group in sorted
    order.

    Examples
    --------
    values   ranks
    1        0
    1        0
    2        2
    2        2
    2        2
    3        5

    Returns
    -------
    Int64Column
        The min rank
    """

    # Equivalent to SQL RANK()
    arg = rlz.column(rlz.any)


@public
class DenseRank(RankBase):
    """Compute the position of first element within each equal-value group in
    sorted order, ignoring duplicate values.

    Examples
    --------
    values   ranks
    1        0
    1        0
    2        1
    2        1
    2        1
    3        2

    Returns
    -------
    IntegerColumn
        The rank
    """

    # Equivalent to SQL DENSE_RANK()
    arg = rlz.column(rlz.any)


@public
class RowNumber(RankBase):
    """Compute the row number starting from 0.

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('values', dt.int64)])
    >>> w = ibis.window(order_by=t.values)
    >>> row_num = ibis.row_number().over(w)
    >>> result = t[t.values, row_num.name('row_num')]

    Returns
    -------
    IntegerColumn
        Row number
    """


@public
class CumulativeOp(AnalyticOp):
    pass


@public
class CumulativeSum(CumulativeOp):
    """Cumulative sum. Requires an ordering window."""

    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest
        return dtype.column_type()


@public
class CumulativeMean(CumulativeOp):
    """Cumulative mean. Requires an order window."""

    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest
        else:
            dtype = dt.float64
        return dtype.column_type()


@public
class CumulativeMax(CumulativeOp):
    """Cumulative max. Requires an order window."""

    arg = rlz.column(rlz.any)
    output_type = rlz.array_like('arg')


@public
class CumulativeMin(CumulativeOp):
    """Cumulative min. Requires an order window."""

    arg = rlz.column(rlz.any)
    output_type = rlz.array_like('arg')


@public
class PercentRank(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.shape_like('arg', dt.double)


@public
class NTile(AnalyticOp):
    arg = rlz.column(rlz.any)
    buckets = rlz.integer
    output_type = rlz.shape_like('arg', dt.int64)


@public
class FirstValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


@public
class LastValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


@public
class NthValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    nth = rlz.integer
    output_type = rlz.typeof('arg')


@public
class Any(ValueOp):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)
    arg = rlz.column(rlz.boolean)

    @property
    def _reduction(self):
        roots = self.arg.op().root_tables()
        return len(roots) < 2

    def output_type(self):
        if self._reduction:
            return dt.boolean.scalar_type()
        else:
            return dt.boolean.column_type()

    def negate(self):
        return NotAny(self.arg)


@public
class NotAny(Any):
    def negate(self):
        return Any(self.arg)


@public
class CumulativeAny(CumulativeOp):
    arg = rlz.column(rlz.boolean)
    output_type = rlz.typeof('arg')


@public
class CumulativeAll(CumulativeOp):
    arg = rlz.column(rlz.boolean)
    output_type = rlz.typeof('arg')
