import six
import toolz
import operator
import itertools
import collections

from ibis.expr.schema import HasSchema, Schema

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt

from ibis import util, compat
from ibis.compat import functools, map
from ibis.expr.signature import Annotable, Argument as Arg


def _safe_repr(x, memo=None):
    return x._repr(memo=memo) if isinstance(x, (ir.Expr, Node)) else repr(x)


# TODO: move to analysis
def distinct_roots(*expressions):
    roots = toolz.concat(
        expression._root_tables() for expression in expressions
    )
    return list(toolz.unique(roots, key=id))


class Node(Annotable):

    __slots__ = ('_expr_cached',)

    def __repr__(self):
        return self._repr()

    def _repr(self, memo=None):
        if memo is None:
            from ibis.expr.format import FormatMemo
            memo = FormatMemo()

        opname = type(self).__name__
        pprint_args = []

        def _pp(x):
            return _safe_repr(x, memo=memo)

        for x in self.args:
            if isinstance(x, (tuple, list)):
                pp = repr([_pp(y) for y in x])
            else:
                pp = _pp(x)
            pprint_args.append(pp)

        return '%s(%s)' % (opname, ', '.join(pprint_args))

    @property
    def inputs(self):
        return tuple(self.args)

    def blocks(self):
        # The contents of this node at referentially distinct and may not be
        # analyzed deeper
        return False

    def flat_args(self):
        for arg in self.args:
            if not isinstance(arg, six.string_types) and isinstance(
                arg, collections.Iterable
            ):
                for x in arg:
                    yield x
            else:
                yield arg

    def equals(self, other, cache=None):
        if cache is None:
            cache = {}

        if (self, other) in cache:
            return cache[(self, other)]

        if self is other:
            cache[(self, other)] = True
            return True

        if type(self) != type(other):
            cache[(self, other)] = False
            return False

        if len(self.args) != len(other.args):
            cache[(self, other)] = False
            return False

        for left, right in zip(self.args, other.args):
            if not all_equal(left, right, cache=cache):
                cache[(self, other)] = False
                return False
        cache[(self, other)] = True
        return True

    def is_ancestor(self, other):
        if isinstance(other, ir.Expr):
            other = other.op()

        return self.equals(other)

    def to_expr(self):
        if not hasattr(self, '_expr_cached'):
            self._expr_cached = self._make_expr()
        return self._expr_cached

    def _make_expr(self):
        klass = self.output_type()
        return klass(self)

    def output_type(self):
        """
        This function must resolve the output type of the expression and return
        the node wrapped in the appropriate ValueExpr type.
        """
        raise NotImplementedError


class ValueOp(Node):

    def root_tables(self):
        exprs = [arg for arg in self.args if isinstance(arg, ir.Expr)]
        return distinct_roots(*exprs)

    def resolve_name(self):
        raise com.ExpressionError('Expression is not named: %s' % repr(self))

    def has_resolved_name(self):
        return False


def all_equal(left, right, cache=None):
    if util.is_iterable(left):
        return util.is_iterable(right) and all(
            map(functools.partial(all_equal, cache=cache), left, right))

    if hasattr(left, 'equals'):
        return left.equals(right, cache=cache)
    return left == right


_table_names = ('unbound_table_{:d}'.format(i) for i in itertools.count())


def genname():
    return next(_table_names)


class TableNode(Node):

    def get_type(self, name):
        return self.schema[name]

    def output_type(self):
        return ir.TableExpr

    def aggregate(self, this, metrics, by=None, having=None):
        return Aggregation(this, metrics, by=by, having=having)

    def sort_by(self, expr, sort_exprs):
        return Selection(expr, [], sort_keys=sort_exprs)


class TableColumn(ValueOp):
    """Selects a column from a TableExpr"""

    name = Arg(six.string_types + six.integer_types)
    table = Arg(ir.TableExpr)

    def __init__(self, name, table):
        schema = table.schema()
        if isinstance(name, six.integer_types):
            name = schema.name_at_position(name)
        super(TableColumn, self).__init__(name, table)

    def _validate(self):
        if self.name not in self.table.schema():
            raise com.IbisTypeError(
                "'{}' is not a field in {}".format(
                    self.name,
                    self.table.columns
                )
            )

    def parent(self):
        return self.table

    def resolve_name(self):
        return self.name

    def has_resolved_name(self):
        return True

    def root_tables(self):
        return self.table._root_tables()

    def _make_expr(self):
        dtype = self.table._get_type(self.name)
        klass = dtype.array_type()
        return klass(self, name=self.name)


def find_all_base_tables(expr, memo=None):
    if memo is None:
        memo = {}

    node = expr.op()

    if isinstance(expr, ir.TableExpr) and node.blocks():
        if id(expr) not in memo:
            memo[id(node)] = expr
        return memo

    for arg in expr.op().flat_args():
        if isinstance(arg, ir.Expr):
            find_all_base_tables(arg, memo)

    return memo


class PhysicalTable(TableNode, HasSchema):

    def blocks(self):
        return True


class UnboundTable(PhysicalTable):
    schema = Arg(sch.Schema)
    name = Arg(six.string_types, default=genname)


class DatabaseTable(PhysicalTable):
    name = Arg(six.string_types)
    schema = Arg(sch.Schema)
    source = Arg(rlz.client)

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


class SQLQueryResult(TableNode, HasSchema):
    """A table sourced from the result set of a select query"""

    query = Arg(rlz.noop)
    schema = Arg(sch.Schema)
    source = Arg(rlz.client)

    def blocks(self):
        return True


class TableArrayView(ValueOp):

    """
    (Temporary?) Helper operation class for SQL translation (fully formed table
    subqueries to be viewed as arrays)
    """
    table = Arg(ir.TableExpr)
    name = Arg(six.string_types)

    def __init__(self, table):
        schema = table.schema()
        if len(schema) > 1:
            raise com.ExpressionError('Table can only have a single column')

        name = schema.names[0]
        return super(TableArrayView, self).__init__(table, name)

    def _make_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ctype.array_type()
        return klass(self, name=self.name)


class UnaryOp(ValueOp):
    arg = Arg(rlz.any)


class BinaryOp(ValueOp):
    """A binary operation"""

    left = Arg(rlz.any)
    right = Arg(rlz.any)


class Cast(ValueOp):
    arg = Arg(rlz.any)
    to = Arg(dt.dtype)

    # see #396 for the issue preventing this
    # def resolve_name(self):
    #     return self.args[0].get_name()

    def output_type(self):
        return rlz.shape_like(self.arg, dtype=self.to)


class TypeOf(UnaryOp):
    output_type = rlz.shape_like('arg', dt.string)


class Negate(UnaryOp):
    arg = Arg(rlz.one_of((rlz.numeric(), rlz.interval())))
    output_type = rlz.typeof('arg')


class IsNull(UnaryOp):
    """Returns true if values are null

    Returns
    -------
    isnull : boolean with dimension of caller
    """
    output_type = rlz.shape_like('arg', dt.boolean)


class NotNull(UnaryOp):
    """Returns true if values are not null

    Returns
    -------
    notnull : boolean with dimension of caller
    """
    output_type = rlz.shape_like('arg', dt.boolean)


class ZeroIfNull(UnaryOp):
    output_type = rlz.typeof('arg')


class IfNull(ValueOp):
    """Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """
    arg = Arg(rlz.any)
    ifnull_expr = Arg(rlz.any)
    output_type = rlz.shape_like('args')


class NullIf(ValueOp):
    """Set values to NULL if they equal the null_if_expr"""
    arg = Arg(rlz.any)
    null_if_expr = Arg(rlz.any)
    output_type = rlz.typeof('arg')


class NullIfZero(ValueOp):

    """
    Set values to NULL if they equal to zero. Commonly used in cases where
    divide-by-zero would produce an overflow or infinity.

    Equivalent to (value == 0).ifelse(ibis.NA, value)

    Returns
    -------
    maybe_nulled : type of caller
    """
    arg = Arg(rlz.numeric)
    output_type = rlz.typeof('arg')


class IsNan(ValueOp):
    arg = Arg(rlz.floating)
    output_type = rlz.shape_like('arg', dt.boolean)


class IsInf(ValueOp):
    arg = Arg(rlz.floating)
    output_type = rlz.shape_like('arg', dt.boolean)


class CoalesceLike(ValueOp):

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column
    arg = Arg(rlz.list_of(rlz.any))

    def output_type(self):
        first = self.arg[0]
        if isinstance(first, (ir.IntegerValue, ir.FloatingValue)):
            dtype = first.type().largest
        else:
            dtype = first.type()

        # self.arg is a list of value expressions
        return rlz.shape_like(self.arg, dtype)


class Coalesce(CoalesceLike):
    pass


class Greatest(CoalesceLike):
    pass


class Least(CoalesceLike):
    pass


class Abs(UnaryOp):
    """Absolute value"""
    output_type = rlz.typeof('arg')


class Ceil(UnaryOp):

    """
    Round up to the nearest integer value greater than or equal to this value

    Returns
    -------
    ceiled : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """
    arg = Arg(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shape_like(self.arg, dt.int64)


class Floor(UnaryOp):

    """
    Round down to the nearest integer value less than or equal to this value

    Returns
    -------
    floored : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """
    arg = Arg(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shape_like(self.arg, dt.int64)


class Round(ValueOp):
    arg = Arg(rlz.numeric)
    digits = Arg(rlz.numeric, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None:
            return rlz.shape_like(self.arg, dt.int64)
        else:
            return rlz.shape_like(self.arg, dt.double)


class Clip(ValueOp):
    arg = Arg(rlz.strict_numeric)
    lower = Arg(rlz.strict_numeric, default=None)
    upper = Arg(rlz.strict_numeric, default=None)
    output_type = rlz.typeof('arg')


class BaseConvert(ValueOp):
    arg = Arg(rlz.one_of([rlz.integer, rlz.string]))
    from_base = Arg(rlz.integer)
    to_base = Arg(rlz.integer)

    def output_type(self):
        return rlz.shape_like(tuple(self.flat_args()), dt.string)


class MathUnaryOp(UnaryOp):
    arg = Arg(rlz.numeric)

    def output_type(self):
        arg = self.arg
        if isinstance(self.arg, ir.DecimalValue):
            dtype = arg.type()
        else:
            dtype = dt.double
        return rlz.shape_like(arg, dtype)


class ExpandingTypeMathUnaryOp(MathUnaryOp):
    def output_type(self):
        if not isinstance(self.arg, ir.DecimalValue):
            return super(ExpandingTypeMathUnaryOp, self).output_type()
        arg = self.arg
        return rlz.shape_like(arg, arg.type().largest)


class Exp(ExpandingTypeMathUnaryOp):
    pass


class Sign(UnaryOp):
    arg = Arg(rlz.numeric)
    output_type = rlz.typeof('arg')


class Sqrt(MathUnaryOp):
    pass


class Logarithm(MathUnaryOp):
    arg = Arg(rlz.strict_numeric)


class Log(Logarithm):
    arg = Arg(rlz.strict_numeric)
    base = Arg(rlz.strict_numeric, default=None)


class Ln(Logarithm):
    """Natural logarithm"""


class Log2(Logarithm):
    """Logarithm base 2"""


class Log10(Logarithm):
    """Logarithm base 10"""


class Degrees(ExpandingTypeMathUnaryOp):
    """Converts radians to degrees"""
    arg = Arg(rlz.numeric)


class Radians(MathUnaryOp):
    """Converts degrees to radians"""
    arg = Arg(rlz.numeric)


# TRIGONOMETRIC OPERATIONS

class TrigonometricUnary(MathUnaryOp):
    """Trigonometric base unary"""
    arg = Arg(rlz.numeric)


class TrigonometricBinary(BinaryOp):
    """Trigonometric base binary"""
    left = Arg(rlz.numeric)
    right = Arg(rlz.numeric)
    output_type = rlz.shape_like('args', dt.float64)


class Acos(TrigonometricUnary):
    """Returns the arc cosine of x"""


class Asin(TrigonometricUnary):
    """Returns the arc sine of x"""


class Atan(TrigonometricUnary):
    """Returns the arc tangent of x"""


class Atan2(TrigonometricBinary):
    """Returns the arc tangent of x and y"""


class Cos(TrigonometricUnary):
    """Returns the cosine of x"""


class Cot(TrigonometricUnary):
    """Returns the cotangent of x"""


class Sin(TrigonometricUnary):
    """Returns the sine of x"""


class Tan(TrigonometricUnary):
    """Returns the tangent of x"""


class StringUnaryOp(UnaryOp):
    arg = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


class Uppercase(StringUnaryOp):
    """Convert string to all uppercase"""


class Lowercase(StringUnaryOp):
    """Convert string to all lowercase"""


class Reverse(StringUnaryOp):
    """Reverse string"""


class Strip(StringUnaryOp):
    """Remove whitespace from left and right sides of string"""


class LStrip(StringUnaryOp):
    """Remove whitespace from left side of string"""


class RStrip(StringUnaryOp):
    """Remove whitespace from right side of string"""


class Capitalize(StringUnaryOp):
    """Return a capitalized version of input string"""


class Substring(ValueOp):
    arg = Arg(rlz.string)
    start = Arg(rlz.integer)
    length = Arg(rlz.integer, default=None)
    output_type = rlz.shape_like('arg', dt.string)


class StrRight(ValueOp):
    arg = Arg(rlz.string)
    nchars = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


class Repeat(ValueOp):
    arg = Arg(rlz.string)
    times = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


class StringFind(ValueOp):
    arg = Arg(rlz.string)
    substr = Arg(rlz.string)
    start = Arg(rlz.integer, default=None)
    end = Arg(rlz.integer, default=None)
    output_type = rlz.shape_like('arg', dt.int64)


class Translate(ValueOp):
    arg = Arg(rlz.string)
    from_str = Arg(rlz.string)
    to_str = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


class LPad(ValueOp):
    arg = Arg(rlz.string)
    length = Arg(rlz.integer)
    pad = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


class RPad(ValueOp):
    arg = Arg(rlz.string)
    length = Arg(rlz.integer)
    pad = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


class FindInSet(ValueOp):
    needle = Arg(rlz.string)
    values = Arg(rlz.list_of(rlz.string, min_length=1))
    output_type = rlz.shape_like('needle', dt.int64)


class StringJoin(ValueOp):
    sep = Arg(rlz.string)
    arg = Arg(rlz.list_of(rlz.string, min_length=1))

    def output_type(self):
        return rlz.shape_like(tuple(self.flat_args()), dt.string)


class BooleanValueOp(object):
    pass


class FuzzySearch(ValueOp, BooleanValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.boolean)


class StringSQLLike(FuzzySearch):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    escape = Arg(six.string_types, default=None)


class StringSQLILike(StringSQLLike):
    """SQL ilike operation"""


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    index = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.string)


class RegexReplace(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    replacement = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


class StringReplace(ValueOp):
    arg = Arg(rlz.string)
    pattern = Arg(rlz.string)
    replacement = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


class StringSplit(ValueOp):
    arg = Arg(rlz.string)
    delimiter = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.Array(dt.string))


class StringConcat(ValueOp):
    arg = Arg(rlz.list_of(rlz.string))
    output_type = rlz.shape_like('arg', dt.string)


class ParseURL(ValueOp):
    arg = Arg(rlz.string)
    extract = Arg(
        rlz.isin({
            'PROTOCOL',
            'HOST',
            'PATH',
            'REF',
            'AUTHORITY',
            'FILE',
            'USERINFO',
            'QUERY'
        })
    )
    key = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.string)


class StringLength(UnaryOp):

    """
    Compute length of strings

    Returns
    -------
    length : int32
    """

    output_type = rlz.shape_like('arg', dt.int32)


class StringAscii(UnaryOp):

    output_type = rlz.shape_like('arg', dt.int32)


# ----------------------------------------------------------------------


class Reduction(ValueOp):
    _reduction = True


class Count(Reduction):
    arg = Arg((ir.ColumnExpr, ir.TableExpr))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


class Arbitrary(Reduction):
    arg = Arg(rlz.column(rlz.any))
    how = Arg(rlz.isin({'first', 'last', 'heavy'}), default='first')
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


class Sum(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest
        return dtype.scalar_type()


class Mean(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


class Quantile(Reduction):
    arg = Arg(rlz.any)
    quantile = Arg(rlz.strict_numeric)
    interpolation = Arg(rlz.isin({'linear', 'lower', 'higher',
                                  'midpoint', 'nearest'}),
                        default='linear')

    def output_type(self):
        return dt.float64.scalar_type()


class MultiQuantile(Quantile):
    arg = Arg(rlz.any)
    quantile = Arg(rlz.value(dt.Array(dt.float64)))
    interpolation = Arg(rlz.isin({'linear', 'lower', 'higher',
                                  'midpoint', 'nearest'}),
                        default='linear')

    def output_type(self):
        return dt.Array(dt.float64).scalar_type()


class VarianceBase(Reduction):
    arg = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest
        else:
            dtype = dt.float64
        return dtype.scalar_type()


class StandardDev(VarianceBase):
    pass


class Variance(VarianceBase):
    pass


class Correlation(Reduction):
    """Coefficient of correlation of a set of number pairs."""
    left = Arg(rlz.column(rlz.numeric))
    right = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.float64.scalar_type()


class Covariance(Reduction):
    """Covariance of a set of number pairs."""
    left = Arg(rlz.column(rlz.numeric))
    right = Arg(rlz.column(rlz.numeric))
    how = Arg(rlz.isin({'sample', 'pop'}), default=None)
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.float64.scalar_type()


class Max(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


class Min(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


class HLLCardinality(Reduction):
    """Approximate number of unique values using HyperLogLog algorithm.

    Impala offers the NDV built-in function for this.
    """
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return functools.partial(ir.IntegerScalar, dtype=dt.int64)


class GroupConcat(Reduction):
    arg = Arg(rlz.column(rlz.any))
    sep = Arg(rlz.string, default=',')
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.string.scalar_type()


class CMSMedian(Reduction):
    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)
    output_type = rlz.scalar_like('arg')


# ----------------------------------------------------------------------
# Analytic functions


class AnalyticOp(ValueOp):
    pass


class WindowOp(ValueOp):
    expr = Arg(rlz.noop)
    window = Arg(rlz.noop)
    output_type = rlz.array_like('expr')

    display_argnames = False

    def __init__(self, expr, window):
        from ibis.expr.window import propagate_down_window
        from ibis.expr.analysis import is_analytic
        if not is_analytic(expr):
            raise com.IbisInputError(
                'Expression does not contain a valid window operation'
            )

        table = ir.find_base_table(expr)
        if table is not None:
            window = window.bind(table)

        expr = propagate_down_window(expr, window)
        super(WindowOp, self).__init__(expr, window)

    def over(self, window):
        new_window = self.window.combine(window)
        return WindowOp(self.expr, new_window)

    @property
    def inputs(self):
        return self.expr.op().inputs[0], self.window

    def root_tables(self):
        result = list(toolz.unique(
            toolz.concatv(
                self.expr._root_tables(),
                distinct_roots(
                    *toolz.concatv(
                        self.window._order_by,
                        self.window._group_by
                    )
                )
            ),
            key=id
        ))
        return result


class ShiftBase(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    offset = Arg(rlz.one_of((rlz.integer, rlz.interval)), default=None)
    default = Arg(rlz.any, default=None)
    output_type = rlz.typeof('arg')


class Lag(ShiftBase):
    pass


class Lead(ShiftBase):
    pass


class RankBase(AnalyticOp):

    def output_type(self):
        return dt.int64.array_type()


class MinRank(RankBase):
    """
    Compute position of first element within each equal-value group in sorted
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
    ranks : Int64Column, starting from 0
    """
    # Equivalent to SQL RANK()
    arg = Arg(rlz.column(rlz.any))


class DenseRank(RankBase):
    """
    Compute position of first element within each equal-value group in sorted
    order, ignoring duplicate values.

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
    ranks : Int64Column, starting from 0
    """
    # Equivalent to SQL DENSE_RANK()
    arg = Arg(rlz.column(rlz.any))


class RowNumber(RankBase):
    """
    Compute row number starting from 0 after sorting by column expression

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('values', dt.int64)])
    >>> w = ibis.window(order_by=t.values)
    >>> row_num = ibis.row_number().over(w)
    >>> result = t[t.values, row_num.name('row_num')]

    Returns
    -------
    row_number : Int64Column, starting from 0
    """
    # Equivalent to SQL ROW_NUMBER()


class CumulativeOp(AnalyticOp):
    pass


class CumulativeSum(CumulativeOp):
    """Cumulative sum. Requires an order window."""
    arg = Arg(rlz.column(rlz.numeric))

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest
        return dtype.array_type()


class CumulativeMean(CumulativeOp):
    """Cumulative mean. Requires an order window."""
    arg = Arg(rlz.column(rlz.numeric))

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest
        else:
            dtype = dt.float64
        return dtype.array_type()


class CumulativeMax(CumulativeOp):
    """Cumulative max. Requires an order window."""
    arg = Arg(rlz.column(rlz.any))
    output_type = rlz.array_like('arg')


class CumulativeMin(CumulativeOp):
    """Cumulative min. Requires an order window."""
    arg = Arg(rlz.column(rlz.any))
    output_type = rlz.array_like('arg')


class PercentRank(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    output_type = rlz.shape_like('arg', dt.double)


class NTile(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    buckets = Arg(rlz.integer)
    output_type = rlz.shape_like('arg', dt.int64)


class FirstValue(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    output_type = rlz.typeof('arg')


class LastValue(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    output_type = rlz.typeof('arg')


class NthValue(AnalyticOp):
    arg = Arg(rlz.column(rlz.any))
    nth = Arg(rlz.integer)
    output_type = rlz.typeof('arg')


# ----------------------------------------------------------------------
# Distinct stuff


class Distinct(TableNode, HasSchema):
    """
    Distinct is a table-level unique-ing operation.

    In SQL, you might have:

    SELECT DISTINCT foo
    FROM table

    SELECT DISTINCT foo, bar
    FROM table
    """
    table = Arg(ir.TableExpr)

    def _validate(self):
        # check whether schema has overlapping columns or not
        assert self.schema

    @property
    def schema(self):
        return self.table.schema()

    def blocks(self):
        return True


class DistinctColumn(ValueOp):

    """
    COUNT(DISTINCT ...) is really just syntactic suger, but we provide a
    distinct().count() nicety for users nonetheless.

    For all intents and purposes, like Distinct, but can be distinguished later
    for evaluation if the result should be array-like versus table-like. Also
    for calling count()
    """
    arg = Arg(rlz.noop)
    output_type = rlz.typeof('arg')

    def count(self):
        """Only valid if the distinct contains a single column"""
        return CountDistinct(self.arg)


class CountDistinct(Reduction):
    arg = Arg(rlz.column(rlz.any))
    where = Arg(rlz.boolean, default=None)

    def output_type(self):
        return dt.int64.scalar_type()


# ---------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueOp):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)
    arg = Arg(rlz.column(rlz.boolean))

    @property
    def _reduction(self):
        roots = self.arg._root_tables()
        return len(roots) < 2

    def output_type(self):
        if self._reduction:
            return dt.boolean.scalar_type()
        else:
            return dt.boolean.array_type()

    def negate(self):
        return NotAny(self.arg)


class All(ValueOp):
    arg = Arg(rlz.column(rlz.boolean))
    output_type = rlz.scalar_like('arg')
    _reduction = True

    def negate(self):
        return NotAll(self.arg)


class NotAny(Any):

    def negate(self):
        return Any(self.arg)


class NotAll(All):

    def negate(self):
        return All(self.arg)


class CumulativeAny(CumulativeOp):
    arg = Arg(rlz.column(rlz.boolean))
    output_type = rlz.typeof('arg')


class CumulativeAll(CumulativeOp):
    arg = Arg(rlz.column(rlz.boolean))
    output_type = rlz.typeof('arg')


# ---------------------------------------------------------------------


class SimpleCaseBuilder(object):

    def __init__(self, expr, cases=None, results=None, default=None):
        self.base = expr
        self.cases = cases or []
        self.results = results or []
        self.default = default

    def when(self, case_expr, result_expr):
        """
        Add a new case-result pair.

        Parameters
        ----------
        case : Expr
          Expression to equality-compare with base expression. Must be
          comparable with the base.
        result : Expr
          Value when the case predicate evaluates to true.

        Returns
        -------
        builder : CaseBuilder
        """
        case_expr = ir.as_value_expr(case_expr)
        result_expr = ir.as_value_expr(result_expr)

        if not rlz.comparable(self.base, case_expr):
            raise TypeError('Base expression and passed case are not '
                            'comparable')

        cases = list(self.cases)
        cases.append(case_expr)

        results = list(self.results)
        results.append(result_expr)

        # Maintain immutability
        return SimpleCaseBuilder(self.base, cases=cases, results=results,
                                 default=self.default)

    def else_(self, result_expr):
        """
        Specify

        Returns
        -------
        builder : CaseBuilder
        """
        result_expr = ir.as_value_expr(result_expr)

        # Maintain immutability
        return SimpleCaseBuilder(self.base, cases=list(self.cases),
                                 results=list(self.results),
                                 default=result_expr)

    def end(self):
        if self.default is None:
            default = ir.null()
        else:
            default = self.default

        op = SimpleCase(self.base, self.cases, self.results, default)
        return op.to_expr()


class SearchedCaseBuilder(object):

    def __init__(self, cases=None, results=None, default=None):
        self.cases = cases or []
        self.results = results or []
        self.default = default

    def when(self, case_expr, result_expr):
        """
        Add a new case-result pair.

        Parameters
        ----------
        case : Expr
          Expression to equality-compare with base expression. Must be
          comparable with the base.
        result : Expr
          Value when the case predicate evaluates to true.

        Returns
        -------
        builder : CaseBuilder
        """
        case_expr = ir.as_value_expr(case_expr)
        result_expr = ir.as_value_expr(result_expr)

        if not isinstance(case_expr, ir.BooleanValue):
            raise TypeError(case_expr)

        cases = list(self.cases)
        cases.append(case_expr)

        results = list(self.results)
        results.append(result_expr)

        # Maintain immutability
        return SearchedCaseBuilder(cases=cases, results=results,
                                   default=self.default)

    def else_(self, result_expr):
        """
        Specify

        Returns
        -------
        builder : CaseBuilder
        """
        result_expr = ir.as_value_expr(result_expr)

        # Maintain immutability
        return SearchedCaseBuilder(cases=list(self.cases),
                                   results=list(self.results),
                                   default=result_expr)

    def end(self):
        if self.default is None:
            default = ir.null()
        else:
            default = self.default

        op = SearchedCase(self.cases, self.results, default)
        return op.to_expr()


class SimpleCase(ValueOp):
    base = Arg(rlz.any)
    cases = Arg(rlz.list_of(rlz.any))
    results = Arg(rlz.list_of(rlz.any))
    default = Arg(rlz.any)

    def _validate(self):
        assert len(self.cases) == len(self.results)

    def root_tables(self):
        return distinct_roots(
            *itertools.chain(
                [self.base],
                self.cases,
                self.results,
                [] if self.default is None else [self.default]
            )
        )

    def output_type(self):
        exprs = self.results + [self.default]
        return rlz.shape_like(self.base, dtype=exprs.type())


class SearchedCase(ValueOp):
    cases = Arg(rlz.list_of(rlz.boolean))
    results = Arg(rlz.list_of(rlz.any))
    default = Arg(rlz.any)

    def _validate(self):
        assert len(self.cases) == len(self.results)

    def root_tables(self):
        cases, results, default = self.args
        return distinct_roots(
            *itertools.chain(
                cases.values,
                results.values,
                [] if default is None else [default]
            )
        )

    def output_type(self):
        exprs = self.results + [self.default]
        dtype = rlz.highest_precedence_dtype(exprs)
        return rlz.shape_like(self.cases, dtype)


class Where(ValueOp):

    """
    Ternary case expression, equivalent to

    bool_expr.case()
             .when(True, true_expr)
             .else_(false_or_null_expr)
    """
    bool_expr = Arg(rlz.boolean)
    true_expr = Arg(rlz.any)
    false_null_expr = Arg(rlz.any)

    def output_type(self):
        return rlz.shape_like(self.bool_expr, self.true_expr.type())


def _validate_join_tables(left, right):
    if not isinstance(left, ir.TableExpr):
        raise TypeError('Can only join table expressions, got {} for '
                        'left table'.format(type(left).__name__))

    if not isinstance(right, ir.TableExpr):
        raise TypeError('Can only join table expressions, got {} for '
                        'right table'.format(type(right).__name__))


def _make_distinct_join_predicates(left, right, predicates):
    # see GH #667

    # If left and right table have a common parent expression (e.g. they
    # have different filters), must add a self-reference and make the
    # appropriate substitution in the join predicates

    if left.equals(right):
        right = right.view()

    predicates = _clean_join_predicates(left, right, predicates)
    return left, right, predicates


def _clean_join_predicates(left, right, predicates):
    import ibis.expr.analysis as L

    result = []

    if not isinstance(predicates, (list, tuple)):
        predicates = [predicates]

    for pred in predicates:
        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise com.ExpressionError('Join key tuple must be '
                                          'length 2')
            lk, rk = pred
            lk = left._ensure_expr(lk)
            rk = right._ensure_expr(rk)
            pred = lk == rk
        elif isinstance(pred, six.string_types):
            pred = left[pred] == right[pred]
        elif not isinstance(pred, ir.Expr):
            raise NotImplementedError

        if not isinstance(pred, ir.BooleanColumn):
            raise com.ExpressionError('Join predicate must be comparison')

        preds = L.flatten_predicate(pred)
        result.extend(preds)

    _validate_join_predicates(left, right, result)
    return result


def _validate_join_predicates(left, right, predicates):
    from ibis.expr.analysis import fully_originate_from

    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    for predicate in predicates:
        if not fully_originate_from(predicate, [left, right]):
            raise com.RelationError('The expression {!r} does not fully '
                                    'originate from dependencies of the table '
                                    'expression.'.format(predicate))


class Join(TableNode):
    left = Arg(rlz.noop)
    right = Arg(rlz.noop)
    predicates = Arg(rlz.noop)

    def __init__(self, left, right, predicates):
        _validate_join_tables(left, right)
        left, right, predicates = _make_distinct_join_predicates(left, right,
                                                                 predicates)
        super(Join, self).__init__(left, right, predicates)

    def _get_schema(self):
        # For joins retaining both table schemas, merge them together here
        left = self.left
        right = self.right

        if not left._is_materialized():
            left = left.materialize()

        if not right._is_materialized():
            right = right.materialize()

        sleft = left.schema()
        sright = right.schema()

        overlap = set(sleft.names) & set(sright.names)
        if overlap:
            raise com.RelationError('Joined tables have overlapping names: %s'
                                    % str(list(overlap)))

        return sleft.append(sright)

    def has_schema(self):
        return False

    def root_tables(self):
        if util.all_of([self.left.op(), self.right.op()],
                       (Join, Selection)):
            # Unraveling is not possible
            return [self.left.op(), self.right.op()]
        else:
            return distinct_roots(self.left, self.right)


class InnerJoin(Join):
    pass


class LeftJoin(Join):
    pass


class RightJoin(Join):
    pass


class OuterJoin(Join):
    pass


class AnyInnerJoin(Join):
    pass


class AnyLeftJoin(Join):
    pass


class LeftSemiJoin(Join):

    def _get_schema(self):
        return self.left.schema()


class LeftAntiJoin(Join):

    def _get_schema(self):
        return self.left.schema()


class MaterializedJoin(TableNode, HasSchema):
    join = Arg(ir.TableExpr)

    def _validate(self):
        assert isinstance(self.join.op(), Join)
        # check whether the underlying schema has overlapping columns or not
        assert self.schema

    @property
    def schema(self):
        return self.join.op()._get_schema()

    def root_tables(self):
        return self.join._root_tables()

    def blocks(self):
        return True


class CrossJoin(InnerJoin):

    """
    Some databases have a CROSS JOIN operator, that may be preferential to use
    over an INNER JOIN with no predicates.
    """

    def __init__(self, *args, **kwargs):
        if 'prefixes' in kwargs:
            raise NotImplementedError

        if len(args) < 2:
            raise com.IbisInputError('Must pass at least 2 tables')

        left = args[0]
        right = args[1]
        for t in args[2:]:
            right = right.cross_join(t)
        InnerJoin.__init__(self, left, right, [])


class AsOfJoin(Join):
    left = Arg(rlz.noop)
    right = Arg(rlz.noop)
    predicates = Arg(rlz.noop)
    by = Arg(rlz.noop, default=None)
    tolerance = Arg(rlz.interval(), default=None)

    def __init__(self, left, right, predicates, by, tolerance):
        super(AsOfJoin, self).__init__(left, right, predicates)
        self.by = _clean_join_predicates(self.left, self.right, by)
        self.tolerance = tolerance


class Union(TableNode, HasSchema):
    left = Arg(rlz.noop)
    right = Arg(rlz.noop)
    distinct = Arg(rlz.validator(bool), default=False)

    def _validate(self):
        if not self.left.schema().equals(self.right.schema()):
            raise com.RelationError('Table schemas must be equal '
                                    'to form union')

    @property
    def schema(self):
        return self.left.schema()

    def blocks(self):
        return True


class Limit(TableNode):
    table = Arg(ir.TableExpr)
    n = Arg(rlz.validator(int))
    offset = Arg(rlz.validator(int))

    def blocks(self):
        return True

    @property
    def schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        return [self]


# --------------------------------------------------------------------
# Sorting

def to_sort_key(table, key):
    if isinstance(key, DeferredSortKey):
        key = key.resolve(table)

    if isinstance(key, ir.SortExpr):
        return key

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, ir.Expr):
        key = table._ensure_expr(key)
        if isinstance(key, (ir.SortExpr, DeferredSortKey)):
            return to_sort_key(table, key)

    if isinstance(sort_order, six.string_types):
        if sort_order.lower() in ('desc', 'descending'):
            sort_order = False
        elif not isinstance(sort_order, bool):
            sort_order = bool(sort_order)

    return SortKey(key, ascending=sort_order).to_expr()


class SortKey(Node):
    expr = Arg(rlz.column(rlz.any))
    ascending = Arg(rlz.validator(bool), default=True)

    def __repr__(self):
        # Temporary
        rows = ['Sort key:',
                '  ascending: {0!s}'.format(self.ascending),
                util.indent(_safe_repr(self.expr), 2)]
        return '\n'.join(rows)

    def output_type(self):
        return ir.SortExpr

    def root_tables(self):
        return self.expr._root_tables()

    def equals(self, other, cache=None):
        # TODO: might generalize this equals based on fields
        # requires a proxy class with equals for non expr values
        return (isinstance(other, SortKey) and
                self.expr.equals(other.expr, cache=cache) and
                self.ascending == other.ascending)

    def resolve_name(self):
        return self.expr.get_name()


class DeferredSortKey(object):

    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()


class SelfReference(TableNode, HasSchema):
    table = Arg(ir.TableExpr)

    @property
    def schema(self):
        return self.table.schema()

    def root_tables(self):
        # The dependencies of this operation are not walked, which makes the
        # table expression holding this relationally distinct from other
        # expressions, so things like self-joins are possible
        return [self]

    def blocks(self):
        return True


class Selection(TableNode, HasSchema):
    table = Arg(ir.TableExpr)
    selections = Arg(rlz.noop, default=None)
    predicates = Arg(rlz.noop, default=None)
    sort_keys = Arg(rlz.noop, default=None)

    def __init__(self, table, selections=None, predicates=None,
                 sort_keys=None):
        import ibis.expr.analysis as L

        # Argument cleaning
        selections = util.promote_list(
            selections if selections is not None else []
        )

        projections = []
        for selection in selections:
            if isinstance(selection, six.string_types):
                projection = table[selection]
            else:
                projection = selection
            projections.append(projection)

        sort_keys = [
            to_sort_key(table, k)
            for k in util.promote_list(
                sort_keys if sort_keys is not None else []
            )
        ]

        predicates = list(toolz.concat(map(
            L.flatten_predicate,
            predicates if predicates is not None else []
        )))

        super(Selection, self).__init__(table=table, selections=projections,
                                        predicates=predicates,
                                        sort_keys=sort_keys)

    def _validate(self):
        from ibis.expr.analysis import FilterValidator

        # Need to validate that the column expressions are compatible with the
        # input table; this means they must either be scalar expressions or
        # array expressions originating from the same root table expression
        dependent_exprs = self.selections + self.sort_keys
        self.table._assert_valid(dependent_exprs)

        # Validate predicates
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

        # Validate no overlapping columns in schema
        assert self.schema

    @property
    def schema(self):
        # Resolve schema and initialize
        if not self.selections:
            return self.table.schema()

        types = []
        names = []

        for projection in self.selections:
            if isinstance(projection, ir.ValueExpr):
                names.append(projection.get_name())
                types.append(projection.type())
            elif isinstance(projection, ir.TableExpr):
                schema = projection.schema()
                names.extend(schema.names)
                types.extend(schema.types)

        return Schema(names, types)

    def blocks(self):
        return bool(self.selections)

    def substitute_table(self, table_expr):
        return Selection(table_expr, self.selections)

    def root_tables(self):
        return [self]

    def can_add_filters(self, wrapped_expr, predicates):
        pass

    def is_ancestor(self, other):
        import ibis.expr.lineage as lin

        if isinstance(other, ir.Expr):
            other = other.op()

        if self.equals(other):
            return True

        expr = self.to_expr()
        fn = lambda e: (lin.proceed, e.op())  # noqa: E731
        for child in lin.traverse(fn, expr):
            if child.equals(other):
                return True

        return False

    # Operator combination / fusion logic

    def aggregate(self, this, metrics, by=None, having=None):
        if len(self.selections) > 0:
            return Aggregation(this, metrics, by=by, having=having)
        else:
            helper = AggregateSelection(this, metrics, by, having)
            return helper.get_result()

    def sort_by(self, expr, sort_exprs):
        sort_exprs = util.promote_list(sort_exprs)
        if not self.blocks():
            resolved_keys = _maybe_convert_sort_keys(self.table, sort_exprs)
            if resolved_keys and self.table._is_valid(resolved_keys):
                return Selection(self.table, self.selections,
                                 predicates=self.predicates,
                                 sort_keys=self.sort_keys + resolved_keys)

        return Selection(expr, [], sort_keys=sort_exprs)


class AggregateSelection(object):
    # sort keys cannot be discarded because of order-dependent
    # aggregate functions like GROUP_CONCAT

    def __init__(self, parent, metrics, by, having):
        self.parent = parent
        self.op = parent.op()
        self.metrics = metrics
        self.by = by
        self.having = having

    def get_result(self):
        if self.op.blocks():
            return self._plain_subquery()
        else:
            return self._attempt_pushdown()

    def _plain_subquery(self):
        return Aggregation(self.parent, self.metrics,
                           by=self.by, having=self.having)

    def _attempt_pushdown(self):
        metrics_valid, lowered_metrics = self._pushdown_exprs(self.metrics)
        by_valid, lowered_by = self._pushdown_exprs(self.by)
        having_valid, lowered_having = self._pushdown_exprs(
            self.having or None)

        if metrics_valid and by_valid and having_valid:
            return Aggregation(self.op.table, lowered_metrics,
                               by=lowered_by,
                               having=lowered_having,
                               predicates=self.op.predicates,
                               sort_keys=self.op.sort_keys)
        else:
            return self._plain_subquery()

    def _pushdown_exprs(self, exprs):
        import ibis.expr.analysis as L

        if exprs is None:
            return True, []

        resolved = self.op.table._resolve(exprs)
        subbed_exprs = []

        valid = False
        if resolved:
            for x in util.promote_list(resolved):
                subbed = L.sub_for(x, [(self.parent, self.op.table)])
                subbed_exprs.append(subbed)
            valid = self.op.table._is_valid(subbed_exprs)
        else:
            valid = False

        return valid, subbed_exprs


def _maybe_convert_sort_keys(table, exprs):
    try:
        return [to_sort_key(table, k) for k in util.promote_list(exprs)]
    except com.IbisError:
        return None


class Aggregation(TableNode, HasSchema):

    """
    metrics : per-group scalar aggregates
    by : group expressions
    having : post-aggregation predicate

    TODO: not putting this in the aggregate operation yet
    where : pre-aggregation predicate
    """
    table = Arg(ir.TableExpr)
    metrics = Arg(rlz.noop)
    by = Arg(rlz.noop)
    having = Arg(rlz.noop, default=None)
    predicates = Arg(rlz.noop, default=None)
    sort_keys = Arg(rlz.noop, default=None)

    def __init__(self, table, metrics, by=None, having=None,
                 predicates=None, sort_keys=None):
        # For tables, like joins, that are not materialized
        metrics = self._rewrite_exprs(table, metrics)

        by = [] if by is None else by
        by = table._resolve(by)

        having = [] if having is None else having
        predicates = [] if predicates is None else predicates

        # order by only makes sense with group by in an aggregation
        sort_keys = [] if not by or sort_keys is None else sort_keys
        sort_keys = [to_sort_key(table, k)
                     for k in util.promote_list(sort_keys)]

        by = self._rewrite_exprs(table, by)
        having = self._rewrite_exprs(table, having)
        predicates = self._rewrite_exprs(table, predicates)
        sort_keys = self._rewrite_exprs(table, sort_keys)

        super(Aggregation, self).__init__(table=table, metrics=metrics, by=by,
                                          having=having, predicates=predicates,
                                          sort_keys=sort_keys)

    def _validate(self):
        from ibis.expr.analysis import is_reduction
        from ibis.expr.analysis import FilterValidator

        # All aggregates are valid
        for expr in self.metrics:
            if not isinstance(expr, ir.ScalarExpr) or not is_reduction(expr):
                raise TypeError('Passed a non-aggregate expression: %s' %
                                _safe_repr(expr))

        for expr in self.having:
            if not isinstance(expr, ir.BooleanScalar):
                raise com.ExpressionError('Having clause must be boolean '
                                          'expression, was: {0!s}'
                                          .format(_safe_repr(expr)))

        # All non-scalar refs originate from the input table
        all_exprs = self.metrics + self.by + self.having + self.sort_keys
        self.table._assert_valid(all_exprs)

        # Validate predicates
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

        # Validate schema has no overlapping columns
        assert self.schema

    def _rewrite_exprs(self, table, what):
        from ibis.expr.analysis import substitute_parents
        what = util.promote_list(what)

        all_exprs = []
        for expr in what:
            if isinstance(expr, ir.ExprList):
                all_exprs.extend(expr.exprs())
            else:
                bound_expr = ir.bind_expr(table, expr)
                all_exprs.append(bound_expr)

        return [substitute_parents(x, past_projection=False)
                for x in all_exprs]

    def blocks(self):
        return True

    def substitute_table(self, table_expr):
        return Aggregation(table_expr, self.metrics, by=self.by,
                           having=self.having)

    @property
    def schema(self):
        names = []
        types = []

        # All exprs must be named
        for e in self.by + self.metrics:
            names.append(e.get_name())
            types.append(e.type())

        return Schema(names, types)

    def sort_by(self, expr, sort_exprs):
        sort_exprs = util.promote_list(sort_exprs)

        resolved_keys = _maybe_convert_sort_keys(self.table, sort_exprs)
        if resolved_keys and self.table._is_valid(resolved_keys):
            return Aggregation(self.table, self.metrics,
                               by=self.by, having=self.having,
                               predicates=self.predicates,
                               sort_keys=self.sort_keys + resolved_keys)

        return Selection(expr, [], sort_keys=sort_exprs)


class NumericBinaryOp(BinaryOp):
    left = Arg(rlz.numeric)
    right = Arg(rlz.numeric)


class Add(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.add)


class Multiply(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.mul)


class Power(NumericBinaryOp):

    def output_type(self):
        if util.all_of(self.args, ir.IntegerValue):
            return rlz.shape_like(self.args, dt.float64)
        else:
            return rlz.shape_like(self.args)


class Subtract(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.sub)


class Divide(NumericBinaryOp):
    output_type = rlz.shape_like('args', dt.float64)


class FloorDivide(Divide):
    output_type = rlz.shape_like('args', dt.int64)


class LogicalBinaryOp(BinaryOp):
    left = Arg(rlz.boolean)
    right = Arg(rlz.boolean)
    output_type = rlz.shape_like('args', dt.boolean)


class Not(UnaryOp):
    arg = Arg(rlz.boolean)
    output_type = rlz.shape_like('arg', dt.boolean)


class Modulus(NumericBinaryOp):
    output_type = rlz.numeric_like('args', operator.mod)


class And(LogicalBinaryOp):
    pass


class Or(LogicalBinaryOp):
    pass


class Xor(LogicalBinaryOp):
    pass


class Comparison(BinaryOp, BooleanValueOp):
    left = Arg(rlz.any)
    right = Arg(rlz.any)

    def __init__(self, left, right):
        """
        Casting rules for type promotions (for resolving the output type) may
        depend in some cases on the target backend.

        TODO: how will overflows be handled? Can we provide anything useful in
        Ibis to help the user avoid them?

        :param left:
        :param right:
        """
        super(BinaryOp, self).__init__(*self._maybe_cast_args(left, right))

    def _maybe_cast_args(self, left, right):
        # it might not be necessary?
        with compat.suppress(com.IbisTypeError):
            return left, rlz.cast(right, left)

        with compat.suppress(com.IbisTypeError):
            return rlz.cast(left, right), right

        return left, right

    def output_type(self):
        if not rlz.comparable(self.left, self.right):
            raise TypeError('Arguments with datatype {} and {} are '
                            'not comparable'.format(self.left.type(),
                                                    self.right.type()))
        return rlz.shape_like(self.args, dt.boolean)


class Equals(Comparison):
    pass


class NotEquals(Comparison):
    pass


class GreaterEqual(Comparison):
    pass


class Greater(Comparison):
    pass


class LessEqual(Comparison):
    pass


class Less(Comparison):
    pass


class IdenticalTo(Comparison):
    pass


class Between(ValueOp, BooleanValueOp):
    arg = Arg(rlz.any)
    lower_bound = Arg(rlz.any)
    upper_bound = Arg(rlz.any)

    def output_type(self):
        arg, lower, upper = self.args

        if not (rlz.comparable(arg, lower) and rlz.comparable(arg, upper)):
            raise TypeError('Arguments are not comparable')

        return rlz.shape_like(self.args, dt.boolean)


class BetweenTime(Between):
    arg = Arg(rlz.one_of([rlz.timestamp, rlz.time]))
    lower_bound = Arg(rlz.one_of([rlz.time, rlz.string]))
    upper_bound = Arg(rlz.one_of([rlz.time, rlz.string]))


class Contains(ValueOp, BooleanValueOp):
    value = Arg(rlz.any)
    options = Arg(rlz.one_of([rlz.list_of(rlz.any),
                              rlz.set_,
                              rlz.column(rlz.any),
                              rlz.array_of(rlz.any)]))

    def __init__(self, value, options):
        # it can be a single expression, like a column
        if not isinstance(options, ir.Expr):
            if util.any_of(options, ir.Expr):
                # or a list of expressions
                options = ir.sequence(options)
            else:
                # or a set of scalar values
                options = frozenset(options)
        super(Contains, self).__init__(value, options)

    def output_type(self):
        all_args = [self.value]

        if isinstance(self.options, ir.ListExpr):
            all_args += self.options
        else:
            all_args += [self.options]

        return rlz.shape_like(all_args, dt.boolean)


class NotContains(Contains):
    pass


class ReplaceValues(ValueOp):

    """
    Apply a multi-value replacement on a particular column. As an example from
    SQL, given DAYOFWEEK(timestamp_col), replace 1 through 5 to "WEEKDAY" and 6
    and 7 to "WEEKEND"
    """
    pass


class SummaryFilter(ValueOp):
    expr = Arg(rlz.noop)

    def output_type(self):
        return dt.boolean.array_type()


class TopK(ValueOp):
    arg = Arg(rlz.noop)
    k = Arg(int)
    by = Arg(rlz.noop)

    def __init__(self, arg, k, by=None):
        if by is None:
            by = arg.count()

        if not isinstance(arg, ir.ColumnExpr):
            raise TypeError(arg)

        if not isinstance(k, int) or k < 0:
            raise ValueError('k must be positive integer, was: {0}'.format(k))

        super(ValueOp, self).__init__(arg, k, by)

    def output_type(self):
        return ir.TopKExpr

    def blocks(self):
        return True


class Constant(ValueOp):
    pass


class TimestampNow(Constant):

    def output_type(self):
        return dt.timestamp.scalar_type()


class E(Constant):

    def output_type(self):
        return functools.partial(ir.FloatingScalar, dtype=dt.float64)


class Pi(Constant):
    """
    The constant pi
    """
    def output_type(self):
        return functools.partial(ir.FloatingScalar, dtype=dt.float64)


class TemporalUnaryOp(UnaryOp):
    arg = Arg(rlz.temporal)


class TimestampUnaryOp(UnaryOp):
    arg = Arg(rlz.timestamp)


_date_units = dict(
    Y='Y',
    y='Y',
    year='Y',
    YEAR='Y',
    YYYY='Y',
    SYYYY='Y',
    YYY='Y',
    YY='Y',

    Q='Q',
    q='Q',
    quarter='Q',
    QUARTER='Q',

    M='M',
    month='M',
    MONTH='M',

    w='W',
    W='W',
    week='W',
    WEEK='W',

    d='D',
    D='D',
    J='D',
    day='D',
    DAY='D'
)

_time_units = dict(
    h='h',
    H='h',
    HH24='h',
    hour='h',
    HOUR='h',

    m='m',
    MI='m',
    minute='m',
    MINUTE='m',

    s='s',
    second='s',
    SECOND='s',

    ms='ms',
    millisecond='ms',
    MILLISECOND='ms',

    us='us',
    microsecond='ms',
    MICROSECOND='ms',

    ns='ns',
    nanosecond='ns',
    NANOSECOND='ns',
)

_timestamp_units = toolz.merge(_date_units, _time_units)


class TimestampTruncate(ValueOp):
    arg = Arg(rlz.timestamp)
    unit = Arg(rlz.isin(_timestamp_units))
    output_type = rlz.shape_like('arg', dt.timestamp)


class DateTruncate(ValueOp):
    arg = Arg(rlz.date)
    unit = Arg(rlz.isin(_date_units))
    output_type = rlz.shape_like('arg', dt.date)


class TimeTruncate(ValueOp):
    arg = Arg(rlz.time)
    unit = Arg(rlz.isin(_time_units))
    output_type = rlz.shape_like('arg', dt.time)


class Strftime(ValueOp):
    arg = Arg(rlz.temporal)
    format_str = Arg(rlz.string)
    output_type = rlz.shape_like('arg', dt.string)


class StringToTimestamp(ValueOp):
    arg = Arg(rlz.string)
    format_str = Arg(rlz.string)
    timezone = Arg(rlz.string, default=None)
    output_type = rlz.shape_like('arg', dt.Timestamp(timezone='UTC'))


class ExtractTemporalField(TemporalUnaryOp):
    output_type = rlz.shape_like('arg', dt.int32)


ExtractTimestampField = ExtractTemporalField


class ExtractDateField(ExtractTemporalField):
    arg = Arg(rlz.one_of([rlz.date, rlz.timestamp]))


class ExtractTimeField(ExtractTemporalField):
    arg = Arg(rlz.one_of([rlz.time, rlz.timestamp]))


class ExtractYear(ExtractDateField):
    pass


class ExtractMonth(ExtractDateField):
    pass


class ExtractDay(ExtractDateField):
    pass


class ExtractHour(ExtractTimeField):
    pass


class ExtractMinute(ExtractTimeField):
    pass


class ExtractSecond(ExtractTimeField):
    pass


class ExtractMillisecond(ExtractTimeField):
    pass


class DayOfWeekIndex(UnaryOp):
    arg = Arg(rlz.one_of([rlz.date, rlz.timestamp]))
    output_type = rlz.shape_like('arg', dt.int16)


class DayOfWeekName(UnaryOp):
    arg = Arg(rlz.one_of([rlz.date, rlz.timestamp]))
    output_type = rlz.shape_like('arg', dt.string)


class DayOfWeekNode(Node):
    arg = Arg(rlz.one_of([rlz.date, rlz.timestamp]))

    def output_type(self):
        return ir.DayOfWeek


class Time(UnaryOp):
    output_type = rlz.shape_like('arg', dt.time)


class Date(UnaryOp):
    output_type = rlz.shape_like('arg', dt.date)


class TimestampFromUNIX(ValueOp):
    arg = Arg(rlz.any)
    unit = Arg(rlz.isin({'s', 'ms', 'us'}))
    output_type = rlz.shape_like('arg', dt.timestamp)


class DecimalUnaryOp(UnaryOp):
    arg = Arg(rlz.decimal)


class DecimalPrecision(DecimalUnaryOp):
    output_type = rlz.shape_like('arg', dt.int32)


class DecimalScale(UnaryOp):
    output_type = rlz.shape_like('arg', dt.int32)


class Hash(ValueOp):
    arg = Arg(rlz.any)
    how = Arg(rlz.isin({'fnv'}))
    output_type = rlz.shape_like('arg', dt.int64)


class DateAdd(BinaryOp):
    left = Arg(rlz.date)
    right = Arg(rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'}))
    output_type = rlz.shape_like('left')


class DateSub(BinaryOp):
    left = Arg(rlz.date)
    right = Arg(rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'}))
    output_type = rlz.shape_like('left')


class DateDiff(BinaryOp):
    left = Arg(rlz.date)
    right = Arg(rlz.date)
    output_type = rlz.shape_like('left', dt.Interval('D'))


class TimeAdd(BinaryOp):
    left = Arg(rlz.time)
    right = Arg(rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'}))
    output_type = rlz.shape_like('left')


class TimeSub(BinaryOp):
    left = Arg(rlz.time)
    right = Arg(rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'}))
    output_type = rlz.shape_like('left')


class TimeDiff(BinaryOp):
    left = Arg(rlz.time)
    right = Arg(rlz.time)
    output_type = rlz.shape_like('left', dt.Interval('s'))


class TimestampAdd(BinaryOp):
    left = Arg(rlz.timestamp)
    right = Arg(rlz.interval(units={'Y', 'Q', 'M', 'W', 'D',
                                    'h', 'm', 's', 'ms', 'us', 'ns'}))
    output_type = rlz.shape_like('left')


class TimestampSub(BinaryOp):
    left = Arg(rlz.timestamp)
    right = Arg(rlz.interval(units={'Y', 'Q', 'M', 'W', 'D',
                                    'h', 'm', 's', 'ms', 'us', 'ns'}))
    output_type = rlz.shape_like('left')


class TimestampDiff(BinaryOp):
    left = Arg(rlz.timestamp)
    right = Arg(rlz.timestamp)
    output_type = rlz.shape_like('left', dt.Interval('s'))


class IntervalBinaryOp(BinaryOp):
    def output_type(self):
        args = [
            arg.cast(arg.type().value_type)
            if isinstance(arg.type(), dt.Interval)
            else arg for arg in self.args
        ]
        expr = rlz.numeric_like(args, self.__class__.op)(self)
        left_dtype = self.left.type()
        dtype_type = type(left_dtype)
        additional_args = {
            attr: getattr(left_dtype, attr) for attr in dtype_type.__slots__
            if attr not in {'unit', 'value_type'}
        }
        dtype = dtype_type(left_dtype.unit, expr.type(), **additional_args)
        return rlz.shape_like(self.args, dtype=dtype)


class IntervalAdd(IntervalBinaryOp):
    left = Arg(rlz.interval)
    right = Arg(rlz.interval)
    op = operator.add


class IntervalSubtract(IntervalBinaryOp):
    left = Arg(rlz.interval)
    right = Arg(rlz.interval)
    op = operator.sub


class IntervalMultiply(IntervalBinaryOp):
    left = Arg(rlz.interval)
    right = Arg(rlz.numeric)
    op = operator.mul


class IntervalFloorDivide(IntervalBinaryOp):
    left = Arg(rlz.interval)
    right = Arg(rlz.numeric)
    op = operator.floordiv


class IntervalFromInteger(ValueOp):
    arg = Arg(rlz.integer)
    unit = Arg(rlz.isin({
        'Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'
    }))

    @property
    def resolution(self):
        return dt.Interval(self.unit).resolution

    def output_type(self):
        dtype = dt.Interval(self.unit, self.arg.type())
        return rlz.shape_like(self.arg, dtype=dtype)


class ArrayLength(UnaryOp):
    arg = Arg(rlz.array)
    output_type = rlz.shape_like('arg', dt.int64)


class ArraySlice(ValueOp):
    arg = Arg(rlz.array)
    start = Arg(rlz.integer)
    stop = Arg(rlz.integer, default=None)
    output_type = rlz.typeof('arg')


class ArrayIndex(ValueOp):
    arg = Arg(rlz.array)
    index = Arg(rlz.integer)

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shape_like(self.arg, value_dtype)


class ArrayConcat(ValueOp):
    left = Arg(rlz.array)
    right = Arg(rlz.array)
    output_type = rlz.shape_like('left')

    def _validate(self):
        left_dtype, right_dtype = self.left.type(), self.right.type()
        if left_dtype != right_dtype:
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left_dtype, right_dtype
                )
            )


class ArrayRepeat(ValueOp):
    arg = Arg(rlz.array)
    times = Arg(rlz.integer)
    output_type = rlz.typeof('arg')


class ArrayCollect(Reduction):
    arg = Arg(rlz.column(rlz.any))

    def output_type(self):
        dtype = dt.Array(self.arg.type())
        return dtype.scalar_type()


class MapLength(ValueOp):
    arg = Arg(rlz.mapping)
    output_type = rlz.shape_like('arg', dt.int64)


class MapValueForKey(ValueOp):
    arg = Arg(rlz.mapping)
    key = Arg(rlz.one_of([rlz.string, rlz.integer]))

    def output_type(self):
        return rlz.shape_like(tuple(self.args), self.arg.type().value_type)


class MapValueOrDefaultForKey(ValueOp):
    arg = Arg(rlz.mapping)
    key = Arg(rlz.one_of([rlz.string, rlz.integer]))
    default = Arg(rlz.any)

    def output_type(self):
        map_type = self.arg.type()
        value_type = map_type.value_type
        default_type = self.default.type()

        if default_type is not dt.null and value_type != default_type:
            raise ValueError("default type: {} must be the same "
                             "as the map value_type {}".format(
                                 default_type, value_type))
        return rlz.shape_like(tuple(self.args), map_type.value_type)


class MapKeys(ValueOp):
    arg = Arg(rlz.mapping)

    def output_type(self):
        arg = self.arg
        return rlz.shape_like(arg, dt.Array(arg.type().key_type))


class MapValues(ValueOp):
    arg = Arg(rlz.mapping)

    def output_type(self):
        arg = self.arg
        return rlz.shape_like(arg, dt.Array(arg.type().value_type))


class MapConcat(ValueOp):
    left = Arg(rlz.mapping)
    right = Arg(rlz.mapping)
    output_type = rlz.typeof('left')


class StructField(ValueOp):
    arg = Arg(rlz.struct)
    field = Arg(six.string_types)

    def output_type(self):
        struct_dtype = self.arg.type()
        value_dtype = struct_dtype[self.field]
        return rlz.shape_like(self.arg, value_dtype)


class Literal(ValueOp):
    value = Arg(rlz.noop)
    dtype = Arg(dt.dtype)

    def __repr__(self):
        return '{}({})'.format(
            type(self).__name__,
            ', '.join(map(repr, self.args))
        )

    def equals(self, other, cache=None):
        return (
            isinstance(other, Literal) and
            isinstance(other.value, type(self.value)) and
            self.value == other.value
        )

    def output_type(self):
        return self.dtype.scalar_type()

    def root_tables(self):
        return []


class NullLiteral(Literal):
    """Typeless NULL literal"""

    value = Arg(type(None), default=None)
    dtype = Arg(dt.Null, default=dt.null)


class ScalarParameter(ValueOp):
    _counter = itertools.count()

    dtype = Arg(dt.dtype)
    counter = Arg(int, default=lambda: next(ScalarParameter._counter))

    def resolve_name(self):
        return 'param_{:d}'.format(self.counter)

    def __repr__(self):
        return '{}(type={})'.format(type(self).__name__, self.dtype)

    def __hash__(self):
        return hash((self.dtype, self.counter))

    def output_type(self):
        return self.dtype.scalar_type()

    def equals(self, other, cache=None):
        return (
            isinstance(other, ScalarParameter) and
            self.counter == other.counter and
            self.dtype.equals(other.dtype, cache=cache)
        )

    def root_tables(self):
        return []


class ExpressionList(Node):
    """Data structure for a list of arbitrary expressions"""

    exprs = Arg(rlz.noop)

    def __init__(self, values):
        super(ExpressionList, self).__init__(list(map(rlz.any, values)))

    @property
    def inputs(self):
        return (tuple(self.exprs),)

    def root_tables(self):
        return distinct_roots(self.exprs)

    def output_type(self):
        return ir.ExprList


class ValueList(ValueOp):
    """Data structure for a list of value expressions"""

    values = Arg(rlz.noop)
    display_argnames = False  # disable showing argnames in repr

    def __init__(self, values):
        super(ValueList, self).__init__(tuple(map(rlz.any, values)))

    def output_type(self):
        dtype = rlz.highest_precedence_dtype(self.values)
        return functools.partial(ir.ListExpr, dtype=dtype)

    def root_tables(self):
        return distinct_roots(*self.values)
