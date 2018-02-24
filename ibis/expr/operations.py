# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator
import six
import itertools
import collections
import toolz
from functools import partial
from ibis.expr.schema import HasSchema, Schema
from ibis.expr.types import as_value_expr  # TODO move these to here

import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.util as util
import ibis.compat as compat
import ibis.expr.rlz as rlz
from collections import OrderedDict

# TODO: eliminate as_value_expr


def _safe_repr(x, memo=None):
    return x._repr(memo=memo) if isinstance(x, (ir.Expr, Node)) else repr(x)


# TODO: move to analysis
def distinct_roots(*expressions):
    roots = toolz.concat(
        expression._root_tables() for expression in expressions
    )
    return list(toolz.unique(roots, key=id))


def _pack_arguments(names, args, kwargs):
    # TODO: docstrings
    # TODO: error messages
    assert len(args) <= len(names)

    result = dict(zip(names, args))
    assert not (set(result.keys()) & set(kwargs.keys()))

    for name in names[len(result):]:
        result[name] = kwargs.get(name, None)

    return result


class Argument(object):

    __slots__ = 'name', 'validate'

    def __init__(self, name, rule):
        self.name = '_' + name
        self.validate = rule

    def __get__(self, obj, objtype):
        return getattr(obj, self.name)

    def __set__(self, obj, value):
        setattr(obj, self.name, self.validate(value))


class OperationMeta(type):

    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return OrderedDict()

    def __new__(cls, name, parents, dct):
        # TODO: cleanup

        slots, args = ['_expr_cached'], []
        for parent in parents:
            # TODO: use ordereddicts to allow overriding
            # or just disallow multiple inheritance for __slots__
            if hasattr(parent, '__slots__'):
                slots += parent.__slots__
            if hasattr(parent, '_arg_names'):
                args += parent._arg_names

        odict = OrderedDict()
        for key, value in dct.items():
            if isinstance(value, rlz.validator):
                # TODO: make it cleaner
                arg = Argument(key, value)
                args.append(key)
                slots.append(arg.name)
                odict[key] = arg
            else:
                odict[key] = value

        odict['__slots__'] = tuple(toolz.unique(slots))
        odict['_arg_names'] = tuple(toolz.unique(args))
        return super(OperationMeta, cls).__new__(cls, name, parents, odict)


class Node(six.with_metaclass(OperationMeta, object)):

    # should read the signature property of class instead of using descriptors?
    # after that call self._validate to support validating more arguments together

    def __init__(self, *args, **kwargs):
        self._expr_cached = None
        for k, v in _pack_arguments(self._arg_names, args, kwargs).items():
            # TODO: consider validating arguments direclty here instead of a
            # descriptor + properties for getters
            setattr(self, k, v)
        self._validate()

    def _validate(self):
        pass

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

        if id(self) == id(other):
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

    # _expr_cached = None

    def to_expr(self):
        # _expr_cache is set in the metaclass
        if self._expr_cached is None:
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

    @property
    def args(self):
        return tuple(getattr(self, name) for name in self._arg_names)


class ValueOp(Node):

    def root_tables(self):
        exprs = [arg for arg in self.args if isinstance(arg, ir.Expr)]
        return distinct_roots(*exprs)

    def resolve_name(self):
        raise com.ExpressionError('Expression is not named: %s' % repr(self))

    def has_resolved_name(self):
        return False


def all_equal(left, right, cache=None):
    if isinstance(left, list):
        if not isinstance(right, list):
            return False
        for a, b in zip(left, right):
            if not all_equal(a, b, cache=cache):
                return False
        return True

    if hasattr(left, 'equals'):
        return left.equals(right, cache=cache)
    return left == right


_table_names = ('t{:d}'.format(i) for i in itertools.count())


def genname():
    return next(_table_names)


class TableNode(Node):

    def get_type(self, name):
        return self.schema[name]

    def _make_expr(self):
        return ir.TableExpr(self)

    def aggregate(self, this, metrics, by=None, having=None):
        return Aggregation(this, metrics, by=by, having=having)

    def sort_by(self, expr, sort_exprs):
        return Selection(expr, [], sort_keys=sort_exprs)


class TableColumn(ValueOp):
    """Selects a column from a TableExpr"""

    name = rlz.instanceof(six.string_types + six.integer_types)
    table = rlz.instanceof(ir.TableExpr)

    def __init__(self, name, table_expr):
        schema = table_expr.schema()
        if isinstance(name, six.integer_types):
            name = schema.name_at_position(name)

        super(TableColumn, self).__init__(name, table_expr)
        if name not in schema:
            raise com.IbisTypeError(
                "'{0}' is not a field in {1}".format(name, table_expr.columns)
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
        # TODO: convert it to output_type
        ctype = self.table._get_type(self.name)
        klass = ctype.array_type()
        return klass(self, name=self.name)


# TODO: move this to analysis
def find_all_base_tables(expr, memo=None):
    # TODO: refactor this
    if memo is None:
        memo = {}

    node = expr.op()

    if isinstance(expr, ir.TableExpr) and node.blocks():
        if id(expr) not in memo:
            memo[id(node)] = expr
        return memo

    for arg in expr.op().flat_args():
        if isinstance(arg, Expr):
            find_all_base_tables(arg, memo)

    return memo


class PhysicalTable(TableNode, HasSchema):

    def blocks(self):
        return True


class UnboundTable(PhysicalTable):
    schema = rlz.schema
    name = rlz.optional(rlz.instanceof(six.string_types), default=genname)


class DatabaseTable(PhysicalTable):
    name = rlz.instanceof(six.string_types)
    schema = rlz.schema
    source = rlz.noop

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


class SQLQueryResult(TableNode, HasSchema):
    """A table sourced from the result set of a select query"""

    query = rlz.noop
    schema = rlz.schema
    source = rlz.noop

    def blocks(self):
        return True


class TableArrayView(ValueOp):

    """
    (Temporary?) Helper operation class for SQL translation (fully formed table
    subqueries to be viewed as arrays)
    """

    def __init__(self, table):
        if not isinstance(table, ir.TableExpr):
            raise com.ExpressionError('Requires table')

        schema = table.schema()
        if len(schema) > 1:
            raise com.ExpressionError('Table can only have a single column')

        # TODO
        self.table = table
        self.name = schema.names[0]

        Node.__init__(self, [table])

    def _make_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ctype.array_type()
        return klass(self, name=self.name)


class UnaryOp(ValueOp):
    arg = rlz.any


class Cast(ValueOp):
    arg = rlz.any
    to = rlz.datatype

    # see #396 for the issue preventing this
    # def resolve_name(self):
    #     return self.args[0].get_name()

    def output_type(self):
        return rlz.shapeof('arg', dtype=self.to)(self)


class TypeOf(ValueOp):
    arg = rlz.value
    output_type = rlz.shapeof('arg', dt.string)


class Negate(UnaryOp):
    arg = rlz.numeric
    output_type = rlz.typeof('arg')


class IsNull(UnaryOp):
    """Returns true if values are null

    Returns
    -------
    isnull : boolean with dimension of caller
    """
    output_type = rlz.shapeof('arg', dt.boolean)


class NotNull(UnaryOp):
    """Returns true if values are not null

    Returns
    -------
    notnull : boolean with dimension of caller
    """
    output_type = rlz.shapeof('arg', 'boolean')


class ZeroIfNull(UnaryOp):
    output_type = rlz.typeof('arg')


class IfNull(ValueOp):
    """Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """

    arg = rlz.any
    ifnull_expr = rlz.any

    def output_type(self):
        args = self.args
        highest_type = rules.highest_precedence_type(args)
        return rules.shape_like(args[0], highest_type)


class NullIf(ValueOp):
    """Set values to NULL if they equal the null_if_expr"""
    arg = rlz.any
    null_if_expr = rlz.any
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

    arg = rlz.numeric
    output_type = rlz.typeof('arg')


class IsNan(ValueOp):
    arg = rlz.floating
    output_type = rlz.shapeof('arg', dt.boolean)


class IsInf(ValueOp):
    arg = rlz.floating
    output_type = rlz.shapeof('arg', dt.boolean)


class CoalesceLike(ValueOp):

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column

    arg = rlz.listof(rlz.any)

    def output_type(self):
        # TODO: how much validation is necessary that the call is valid and can
        # succeed?
        first_value = self.arg[0]

        if isinstance(first_value, ir.IntegerValue):
            out_type = dt.int64
        elif isinstance(first_value, ir.FloatingValue):
            out_type = dt.double
        else:
            out_type = first_value.type()

        return rules.shape_like_args(self.arg, out_type)


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
    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shapeof('arg', dt.int64)(self)


class Floor(UnaryOp):

    """
    Round down to the nearest integer value less than or equal to this value

    Returns
    -------
    floored : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """
    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shapeof('arg', dt.int64)(self)


class Round(ValueOp):

    arg = rlz.numeric
    digits = rlz.optional(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None:
            return rules.shape_like(self.arg, dt.int64)
        else:
            return rules.shape_like(self.arg, 'double')


class Clip(ValueOp):
    arg = rlz.strict_numeric
    lower = rlz.optional(rlz.strict_numeric)
    upper = rlz.optional(rlz.strict_numeric)
    output_type = rlz.typeof('arg')


class BaseConvert(ValueOp):
    arg = rlz.oneof([rlz.integer, rlz.string])
    from_base = rlz.integer
    to_base = rlz.integer
    output_type = rules.shape_like_flatargs(dt.string)


class RealUnaryOp(UnaryOp):
    arg = rlz.numeric
    output_type = rlz.shapeof('arg', 'double')


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    # This is the Impala output for both integers and double/float
    output_type = rlz.shapeof('arg', 'float')


class Sqrt(RealUnaryOp):
    pass


class Logarithm(RealUnaryOp):
    arg = rlz.strict_numeric


class Log(Logarithm):
    arg = rlz.strict_numeric
    base = rlz.optional(rlz.strict_numeric)


class Ln(Logarithm):
    """Natural logarithm"""


class Log2(Logarithm):
    """Logarithm base 2"""


class Log10(Logarithm):
    """Logarithm base 10"""


class StringUnaryOp(UnaryOp):
    arg = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


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
    arg = rlz.string
    start = rlz.integer
    length = rlz.optional(rlz.integer)
    output_type = rlz.shapeof('arg', dt.string)


class StrRight(ValueOp):
    arg = rlz.string
    nchars = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class Repeat(ValueOp):
    arg = rlz.string
    times = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class StringFind(ValueOp):
    arg = rlz.string
    substr = rlz.string
    start = rlz.optional(rlz.integer)
    end = rlz.optional(rlz.integer)
    output_type = rlz.shapeof('arg', dt.int64)


class Translate(ValueOp):
    arg = rlz.string
    from_str = rlz.string
    to_str = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class LPad(ValueOp):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class RPad(ValueOp):
    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class FindInSet(ValueOp):
    needle = rlz.string
    values = rlz.listof(rlz.string, min_length=1)
    output_type = rlz.shapeof('needle', dt.int64)


class StringJoin(ValueOp):
    sep = rlz.string
    arg = rlz.listof(rlz.string, min_length=1)
    output_type = rules.shape_like_flatargs(dt.string)


class BooleanValueOp(object):
    pass


class FuzzySearch(ValueOp, BooleanValueOp):
    arg = rlz.string
    pattern = rlz.string
    output_type = rlz.shapeof('arg', dt.boolean)


class StringSQLLike(FuzzySearch):
    arg = rlz.string
    pattern = rlz.string
    escape = rlz.instanceof(six.string_types)


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    index = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class RegexReplace(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class StringReplace(ValueOp):
    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class StringSplit(ValueOp):
    arg = rlz.string
    delimiter = rlz.string
    output_type = rlz.shapeof('arg', 'array<string>')


class StringConcat(ValueOp):
    arg = rlz.listof(rlz.string)

    def output_type(self):
        return rules.shape_like_args(self.args, dt.string)


class ParseURL(ValueOp):
    arg = rlz.string
    extract = rlz.isin(['PROTOCOL', 'HOST', 'PATH',
                        'REF', 'AUTHORITY', 'FILE',
                        'USERINFO', 'QUERY'])
    key = rlz.optional(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class StringLength(UnaryOp):

    """
    Compute length of strings

    Returns
    -------
    length : int32
    """

    output_type = rlz.shapeof('arg', dt.int32)


class StringAscii(UnaryOp):

    output_type = rlz.shapeof('arg', dt.int32)


class BinaryOp(ValueOp):

    """
    A binary operation

    """
    # Casting rules for type promotions (for resolving the output type) may
    # depend in some cases on the target backend.
    #
    # TODO: how will overflows be handled? Can we provide anything useful in
    # Ibis to help the user avoid them?

    left = rlz.any
    right = rlz.any

    def __init__(self, left, right):
        super(BinaryOp, self).__init__(*self._maybe_cast_args(left, right))

    def _maybe_cast_args(self, left, right):
        return left, right

    def output_type(self):
        raise NotImplementedError


# ----------------------------------------------------------------------


class Reduction(ValueOp):
    _reduction = True


class Count(Reduction):
    arg = rlz.instanceof((ir.ColumnExpr, ir.TableExpr))
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        return partial(ir.IntegerScalar, dtype=dt.int64)


class Arbitrary(Reduction):
    arg = rlz.column(rlz.any)
    how = rlz.optional(rlz.isin({'first', 'last', 'heavy'}), default='first')
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class Sum(Reduction):
    arg = rlz.column(rlz.numeric)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest()
        return dtype.scalar_type()


class Mean(Reduction):
    arg = rlz.column(rlz.numeric)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


class Quantile(Reduction):

    arg = rlz.any
    quantile = rlz.strict_numeric
    interpolation = rlz.optional(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear'
    )

    def output_type(self):
        return dt.float64.scalar_type()


class MultiQuantile(Quantile):

    arg = rlz.any
    quantile = rlz.value(dt.Array(dt.double))
    interpolation = rlz.optional(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear'
    )

    def output_type(self):
        return dt.float64.scalar_type()


class VarianceBase(Reduction):
    arg = rlz.column(rlz.numeric)
    how = rlz.optional(rlz.isin({'sample', 'pop'}))
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


class StandardDev(VarianceBase):
    pass


class Variance(VarianceBase):
    pass


class Max(Reduction):
    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class Min(Reduction):
    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class HLLCardinality(Reduction):

    """
    Approximate number of unique values using HyperLogLog algorithm. Impala
    offers the NDV built-in function for this.
    """
    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return partial(ir.IntegerScalar, dtype=dt.int64)


class GroupConcat(Reduction):

    arg = rlz.column(rlz.any)
    sep = rlz.optional(rlz.string, default=',')
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        return dt.string.scalar_type()


class CMSMedian(Reduction):

    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """
    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


# ----------------------------------------------------------------------
# Analytic functions


class AnalyticOp(ValueOp):
    pass


class WindowOp(ValueOp):

    expr = rlz.noop
    window = rlz.noop
    output_type = rlz.arrayof('expr')

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

    arg = rlz.column(rlz.any)
    offset = rlz.optional(rlz.integer)
    default = rlz.optional(rlz.any)
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
    arg = rlz.column(rlz.any)


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
    arg = rlz.column(rlz.any)


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

    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest()
        return dtype.array_type()


class CumulativeMean(CumulativeOp):
    """Cumulative mean. Requires an order window."""

    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest()
        else:
            dtype = dt.float64
        return dtype.array_type()


class CumulativeMax(CumulativeOp):
    """Cumulative max. Requires an order window."""

    arg = rlz.column(rlz.any)
    output_type = rlz.arrayof('arg')


class CumulativeMin(CumulativeOp):
    """Cumulative min. Requires an order window."""

    arg = rlz.column(rlz.any)
    output_type = rlz.arrayof('arg')


class PercentRank(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.shapeof('arg', dt.double)


class NTile(AnalyticOp):
    arg = rlz.column(rlz.any)
    buckets = rlz.integer
    output_type = rlz.shapeof('arg', dt.int64)


class FirstValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


class LastValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


class NthValue(AnalyticOp):
    arg = rlz.column(rlz.any)
    nth = rlz.integer
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

    table = rlz.instanceof(ir.TableExpr)

    def __init__(self, table):
        super(Distinct, self).__init__(table)
        self.schema  # TODO

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

    arg = rlz.noop
    output_type = rlz.typeof('arg')

    def count(self):
        """Only valid if the distinct contains a single column"""
        return CountDistinct(self.arg)


class CountDistinct(Reduction):
    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        return dt.int64.scalar_type()


# ---------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueOp):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)
    arg = rlz.column(rlz.boolean)

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

    arg = rlz.column(rlz.boolean)
    output_type = rlz.scalarof('arg')
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
    arg = rlz.column(rlz.boolean)
    output_type = rlz.typeof('arg')


class CumulativeAll(CumulativeOp):
    arg = rlz.column(rlz.boolean)
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
        case_expr = as_value_expr(case_expr)
        result_expr = as_value_expr(result_expr)

        if not rules.comparable(self.base, case_expr):
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
        result_expr = as_value_expr(result_expr)

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
        case_expr = as_value_expr(case_expr)
        result_expr = as_value_expr(result_expr)

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
        result_expr = as_value_expr(result_expr)

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
    base = rlz.any
    cases = rlz.listof(rlz.any)
    results = rlz.listof(rlz.any)
    default = rlz.any

    def __init__(self, base, cases, results, default):
        assert len(cases) == len(results)
        super(SimpleCase, self).__init__(base, cases, results, default)

    def root_tables(self):
        all_exprs = [self.base] + self.cases + self.results + (
            [] if self.default is None else [self.default]
        )
        return distinct_roots(*all_exprs)

    def output_type(self):
        out_exprs = list(filter(
            lambda expr: expr is not None,
            self.results + [self.default]
        ))
        typename = rules.highest_precedence_type(out_exprs)
        return rules.shape_like(self.base, typename)


class SearchedCase(ValueOp):
    cases = rlz.listof(rlz.boolean)
    results = rlz.listof(rlz.any)
    default = rlz.any

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super(SearchedCase, self).__init__(cases, results, default)

    def root_tables(self):
        cases, results, default = self.args
        all_exprs = cases + results + ([] if default is None else [default])
        return distinct_roots(*all_exprs)

    def output_type(self):
        cases, results, default = self.args
        out_exprs = results + [default]
        typename = rules.highest_precedence_type(out_exprs)
        return rules.shape_like_args(cases, typename)


class Where(ValueOp):

    """
    Ternary case expression, equivalent to

    bool_expr.case()
             .when(True, true_expr)
             .else_(false_or_null_expr)
    """
    bool_expr = rlz.boolean
    true_expr = rlz.any
    false_null_expr = rlz.any

    def output_type(self):
        return rules.shape_like(self.bool_expr, self.true_expr.type())


def _validate_join_tables(left, right):
    if not isinstance(left, ir.TableExpr):
        raise TypeError('Can only join table expressions, got {} for '
                        'left table'.format(type(left).__name__))

    if not isinstance(right, ir.TableExpr):
        raise TypeError('Can only join table expressions, got {} for '
                        'right table'.format(type(right).__name__))


# TODO: move to analysis
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

    left = rlz.noop
    right = rlz.noop
    predicates = rlz.noop

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

    join = rlz.noop

    def __init__(self, join_expr):
        assert isinstance(join_expr.op(), Join)
        super(MaterializedJoin, self).__init__(join_expr)
        self.schema  # TODO cleanup, this validates there is no schema overlapping

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

    left = rlz.noop
    right = rlz.noop
    predicates = rlz.noop
    by = rlz.noop

    def __init__(self, left, right, predicates, by):
        # TODO cleanup
        super(AsOfJoin, self).__init__(left, right, predicates)
        self.by = _clean_join_predicates(self.left, self.right, by)


class Union(TableNode, HasSchema):

    left = rlz.noop
    right = rlz.noop
    distinct = rlz.optional(rlz.validator(bool), default=False)

    def __init__(self, left, right, distinct=False):
        super(Union, self).__init__(left, right, distinct=distinct)
        self._validate()

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

    table = rlz.instanceof(ir.TableExpr)
    n = rlz.validator(int)
    offset = rlz.validator(int)

    # TODO:
    # _arg_names = [None, 'n', 'offset']

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

    by = rlz.column(rlz.any)
    ascending = rlz.optional(rlz.validator(bool), default=True)

    def __init__(self, by, ascending=True):
        if not isinstance(by, ir.ColumnExpr):
            raise com.ExpressionError('Must be an array/column expression')
        super(SortKey, self).__init__(by, ascending)

    def __repr__(self):
        # Temporary  # TODO
        rows = ['Sort key:',
                '  ascending: {0!s}'.format(self.ascending),
                util.indent(_safe_repr(self.by), 2)]
        return '\n'.join(rows)

    def root_tables(self):
        return self.by._root_tables()

    def _make_expr(self):
        return ir.SortExpr(self)

    def equals(self, other, cache=None):
        # TODO: might generalize this equals based on fields
        # requires a proxy class with equals for non expr values
        return (isinstance(other, SortKey) and
                self.by.equals(other.by, cache=cache) and
                self.ascending == other.ascending)


class DeferredSortKey(object):

    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()


class SelfReference(TableNode, HasSchema):

    table = rlz.instanceof(ir.TableExpr)

    def __init__(self, table_expr):
        super(SelfReference, self).__init__(table_expr)
        # TODO
        self.schema

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

    table = rlz.table
    # schema = rlz.schema
    selections = rlz.noop
    predicates = rlz.noop
    sort_keys = rlz.noop

    # TODO: rename arguments like the following
    #_arg_names = ['table', 'selections', 'predicates', 'sort_keys']

    def __init__(self, table, selections=None, predicates=None,
                 sort_keys=None):
        import ibis.expr.analysis as L

        # Argument cleaning
        selections = util.promote_list(
            selections if selections is not None else []
        )
        clean_exprs, schema = self._get_schema(table, selections)

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

        dependent_exprs = clean_exprs + sort_keys

        table._assert_valid(dependent_exprs)
        self._validate_predicates(table, predicates)

        super(Selection, self).__init__(
            table=table, selections=clean_exprs,
            predicates=predicates, sort_keys=sort_keys)

    def blocks(self):
        return bool(self.selections)

    def _validate_predicates(self, table, predicates):
        from ibis.expr.analysis import FilterValidator
        validator = FilterValidator([table])
        validator.validate_all(predicates)

    # def _validate(self, table, exprs):
    #     # Need to validate that the column expressions are compatible with the
    #     # input table; this means they must either be scalar expressions or
    #     # array expressions originating from the same root table expression
    #     table._assert_valid(exprs)

    # TODO: cleanup / remove
    def _get_schema(self, table, projections):
        if not projections:
            return projections, table.schema()

        # Resolve schema and initialize
        types = []
        names = []
        clean_exprs = []
        for projection in projections:
            if isinstance(projection, six.string_types):
                projection = self.table[projection]

            if isinstance(projection, ir.ValueExpr):
                names.append(projection.get_name())
                types.append(projection.type())
            elif isinstance(projection, ir.TableExpr):
                schema = projection.schema()
                names.extend(schema.names)
                types.extend(schema.types)
            else:
                raise TypeError(
                    "Don't know how to clean expression of type {}".format(
                        type(projection).__name__
                    )
                )

            clean_exprs.append(projection)

        # validate uniqueness
        return clean_exprs, Schema(names, types)

    # TODO: cleanup
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

    table = rlz.table
    metrics = rlz.noop
    by = rlz.noop
    having = rlz.noop
    predicates = rlz.noop
    sort_keys = rlz.noop

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
        self._validate()
        self._validate_predicates()
        self.schema  # TODO

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

    def _validate(self):
        from ibis.expr.analysis import is_reduction
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

    def _validate_predicates(self):
        from ibis.expr.analysis import FilterValidator
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

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


class Add(BinaryOp):

    def output_type(self):
        helper = rules.BinaryPromoter(self.left, self.right, operator.add)
        return helper.get_result()


class Multiply(BinaryOp):

    def output_type(self):
        helper = rules.BinaryPromoter(self.left, self.right, operator.mul)
        return helper.get_result()


class Power(BinaryOp):

    def output_type(self):
        return rules.PowerPromoter(self.left, self.right).get_result()


class Subtract(BinaryOp):

    def output_type(self):
        helper = rules.BinaryPromoter(self.left, self.right, operator.sub)
        return helper.get_result()


class Divide(BinaryOp):
    left = rlz.numeric
    right = rlz.numeric

    def output_type(self):
        return rules.shape_like_args(self.args, 'double')


class FloorDivide(Divide):

    def output_type(self):
        return rules.shape_like_args(self.args, dt.int64)


class LogicalBinaryOp(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, ir.BooleanValue):
            raise TypeError('Only valid with boolean data')
        return rules.shape_like_args(self.args, 'boolean')


class Not(UnaryOp):
    arg = rlz.boolean
    output_type = rlz.shapeof('arg', 'boolean')


class Modulus(BinaryOp):

    def output_type(self):
        helper = rules.BinaryPromoter(self.left, self.right,
                                      operator.mod)
        return helper.get_result()


class And(LogicalBinaryOp):
    pass


class Or(LogicalBinaryOp):
    pass


class Xor(LogicalBinaryOp):
    pass


class Comparison(BinaryOp, BooleanValueOp):

    def _maybe_cast_args(self, left, right):
        # it might not be necessary?
        with compat.suppress(com.IbisTypeError):
            return left, ir.cast(right, left)

        with compat.suppress(com.IbisTypeError):
            return ir.cast(left, right), right

        return left, right

    def output_type(self):
        if not rules.comparable(self.left, self.right):
            raise TypeError('Arguments with datatype {} and {} are '
                            'not comparable'.format(self.left.type(),
                                                    self.right.type()))
        return rules.shape_like_args(self.args, 'boolean')


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
    arg = rlz.any
    lower_bound = rlz.any
    upper_bound = rlz.any

    def output_type(self):
        arg, lower, upper = self.args

        if not (rules.comparable(arg, lower) and rules.comparable(arg, upper)):
            raise TypeError('Arguments are not comparable')

        return rules.shape_like_args(self.args, 'boolean')


class BetweenTime(Between):

    arg = rlz.oneof([rlz.timestamp, rlz.time])
    lower_bound = rlz.oneof([rlz.time, rlz.string])
    upper_bound = rlz.oneof([rlz.time, rlz.string])


class Contains(ValueOp, BooleanValueOp):

    value = rlz.noop
    options = rlz.noop

    def __init__(self, value, options):
        value = as_value_expr(value)
        options = as_value_expr(options)
        super(Contains, self).__init__(value, options)

    def output_type(self):
        all_args = [self.value]

        options = self.options.op()
        if isinstance(options, ValueList):
            all_args += options.values
        elif isinstance(self.options, ir.ColumnExpr):
            all_args += [self.options]
        else:
            raise TypeError(type(options))

        return rules.shape_like_args(all_args, 'boolean')


class NotContains(Contains):
    pass


class ReplaceValues(ValueOp):

    """
    Apply a multi-value replacement on a particular column. As an example from
    SQL, given DAYOFWEEK(timestamp_col), replace 1 through 5 to "WEEKDAY" and 6
    and 7 to "WEEKEND"
    """
    pass


# TODO put to types
class TopKExpr(ir.AnalyticExpr):

    def type(self):
        return 'topk'

    def _table_getitem(self):
        return self.to_filter()

    def to_filter(self):
        return SummaryFilter(self).to_expr()

    def to_aggregation(self, metric_name=None, parent_table=None,
                       backup_metric_name=None):
        """
        Convert the TopK operation to a table aggregation
        """
        op = self.op()

        arg_table = ir.find_base_table(op.arg)

        by = op.by
        if not isinstance(by, ir.Expr):
            by = by(arg_table)
            by_table = arg_table
        else:
            by_table = ir.find_base_table(op.by)

        if metric_name is None:
            if by.get_name() == op.arg.get_name():
                by = by.name(backup_metric_name)
        else:
            by = by.name(metric_name)

        if arg_table.equals(by_table):
            agg = arg_table.aggregate(by, by=[op.arg])
        elif parent_table is not None:
            agg = parent_table.aggregate(by, by=[op.arg])
        else:
            raise com.IbisError('Cross-table TopK; must provide a parent '
                                'joined table')

        return agg.sort_by([(by.get_name(), False)]).limit(op.k)


class SummaryFilter(ValueOp):
    expr = rlz.noop

    def output_type(self):
        return ir.BooleanColumn


class TopK(ValueOp):
    arg = rlz.noop
    k = rlz.noop
    by = rlz.noop

    def blocks(self):
        return True

    # TODO: simplify
    def __init__(self, arg, k, by=None):
        if by is None:
            by = arg.count()

        if not isinstance(arg, ir.ColumnExpr):
            raise TypeError(arg)

        if not isinstance(k, int) or k < 0:
            raise ValueError('k must be positive integer, was: {0}'.format(k))

        self.arg = arg
        self.k = k
        self.by = by

        super(ValueOp, self).__init__(arg, k, by)

    def output_type(self):
        return TopKExpr


class Constant(ValueOp):
    pass


class TimestampNow(Constant):

    def output_type(self):
        return dt.timestamp.scalar_type()


class E(Constant):

    def output_type(self):
        return partial(ir.FloatingScalar, dtype=dt.float64)


class TemporalUnaryOp(UnaryOp):
    arg = rlz.temporal


class TimestampUnaryOp(UnaryOp):
    arg = rlz.timestamp


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
    arg = rlz.timestamp
    unit = rlz.isin(_timestamp_units)
    output_type = rlz.shapeof('arg', dt.timestamp)


class DateTruncate(ValueOp):
    arg = rlz.date
    unit = rlz.isin(_date_units)
    output_type = rlz.shapeof('arg', dt.date)


class TimeTruncate(ValueOp):
    arg = rlz.time
    unit = rlz.isin(_time_units)
    output_type = rlz.shapeof('arg', dt.time)


class Strftime(ValueOp):
    arg = rlz.temporal
    format_str = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class ExtractTemporalField(TemporalUnaryOp):
    output_type = rlz.shapeof('arg', dt.int32)


class ExtractTimestampField(TimestampUnaryOp):
    output_type = rlz.shapeof('arg', dt.int32)


class ExtractYear(ExtractTemporalField):
    pass


class ExtractMonth(ExtractTemporalField):
    pass


class DayOfWeekIndex(UnaryOp):
    arg = rlz.oneof([rlz.date, rlz.timestamp])
    output_type = rlz.shapeof('arg', dt.int32)


class DayOfWeekName(UnaryOp):
    arg = rlz.oneof([rlz.date, rlz.timestamp])
    output_type = rlz.shapeof('arg', dt.string)


class DayOfWeek(ir.Expr):

    def index(self):
        arg, = self.op().args
        return DayOfWeekIndex(arg).to_expr()

    def full_name(self):
        arg, = self.op().args
        return DayOfWeekName(arg).to_expr()


class DayOfWeekNode(Node):
    arg = rlz.oneof([rlz.date, rlz.timestamp])

    def output_type(self):
        return DayOfWeek


class ExtractDay(ExtractTemporalField):
    pass


class ExtractHour(ExtractTimestampField):
    pass


class ExtractMinute(ExtractTimestampField):
    pass


class ExtractSecond(ExtractTimestampField):
    pass


class ExtractMillisecond(ExtractTimestampField):
    pass


class Time(UnaryOp):

    output_type = rlz.shapeof('arg', 'time')


class Date(UnaryOp):

    output_type = rlz.shapeof('arg', 'date')


class TimestampFromUNIX(ValueOp):
    arg = rlz.any
    unit = rlz.isin(['s', 'ms', 'us'])
    output_type = rlz.shapeof('arg', 'timestamp')


class DecimalUnaryOp(UnaryOp):
    arg = rlz.decimal


class DecimalPrecision(DecimalUnaryOp):

    output_type = rlz.shapeof('arg', dt.int32)


class DecimalScale(UnaryOp):

    output_type = rlz.shapeof('arg', dt.int32)


class Hash(ValueOp):
    arg = rlz.any
    how = rlz.isin({'fnv'})
    output_type = rlz.shapeof('arg', dt.int64)


class DateAdd(Add):
    left = rlz.date
    right = rlz.interval(units=['Y', 'Q', 'M', 'W', 'D'])
    output_type = rlz.shapeof('left', 'date')


class DateSub(Subtract):
    left = rlz.date
    right = rlz.interval(units=['Y', 'Q', 'M', 'W', 'D'])
    output_type = rlz.shapeof('left', 'date')


class DateDiff(BinaryOp):
    left = rlz.date
    right = rlz.date
    output_type = rlz.shapeof('left', dt.Interval('D'))


class TimeAdd(Add):
    left = rlz.time
    right = rlz.interval(units=['h', 'm', 's'])
    output_type = rlz.shapeof('left', 'time')


class TimeSub(Subtract):
    left = rlz.time
    right = rlz.interval(units=['h', 'm', 's'])
    output_type = rlz.shapeof('left', 'time')


class TimeDiff(BinaryOp):
    left = rlz.time
    right = rlz.time
    output_type = rlz.shapeof('left', dt.Interval('s'))


class TimestampAdd(Add):
    left = rlz.timestamp
    right = rlz.interval(units=['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's'])
    output_type = rlz.shapeof('left', 'timestamp')


class TimestampSub(Subtract):
    left = rlz.timestamp
    right = rlz.interval(units=['Y', 'Q', 'M', 'W', 'D', 'h', 'm', 's'])
    output_type = rlz.shapeof('left', 'timestamp')


class TimestampDiff(BinaryOp):
    left = rlz.timestamp
    right = rlz.timestamp
    output_type = rlz.shapeof('left', dt.Interval('s'))


class IntervalAdd(Add):
    left = rlz.interval
    right = rlz.interval
    output_type = rlz.shapeof('left')


class IntervalMultiply(Multiply):

    def output_type(self):
        helper = rules.IntervalPromoter(self.left, self.right, operator.mul)
        return helper.get_result()


class IntervalFloorDivide(FloorDivide):
    left = rlz.interval
    right = rlz.numeric
    output_type = rlz.shapeof('left')


class IntervalFromInteger(ValueOp):
    arg = rlz.integer
    unit = rlz.isin(['Y', 'Q', 'M', 'W', 'D',
                     'h', 'm', 's', 'ms', 'us', 'ns'])

    @property
    def resolution(self):
        return dt.Interval(self.unit).resolution

    def output_type(self):
        dtype = dt.Interval(self.unit, self.arg.type())
        return rlz.shapeof('arg', dtype=dtype)(self)


class ArrayLength(UnaryOp):
    arg = rlz.value(dt.Array(dt.any))
    output_type = rlz.shapeof('arg', dt.int64)


class ArraySlice(ValueOp):
    arg = rlz.value(dt.Array(dt.any))
    start = rlz.integer
    stop = rlz.optional(rlz.integer)
    output_type = rlz.typeof('arg')


class ArrayIndex(ValueOp):
    arg = rlz.value(dt.Array(dt.any))
    index = rlz.integer

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shapeof('arg', value_dtype)(self)


class ArrayConcat(ValueOp):
    left = rlz.value(dt.Array(dt.any))
    right = rlz.value(dt.Array(dt.any))

    def output_type(self):
        if self.left.type() != self.right.type():
            raise com.IbisTypeError(
                'Array types must match exactly in a {} operation. '
                'Left type {} != Right type {}'.format(
                    type(self).__name__, left_type, right_type
                )
            )
        return rlz.shapeof('left')(self)


class ArrayRepeat(ValueOp):
    arg = rlz.value(dt.Array(dt.any))
    times = rlz.integer
    output_type = rlz.typeof('arg')


class ArrayCollect(Reduction):
    arg = rlz.column(rlz.any)
    output_type = rlz.scalarof('arg')


class MapLength(ValueOp):
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.shapeof('arg', dt.int64)


class MapValueForKey(ValueOp):
    arg = rlz.value(dt.Map(dt.any, dt.any))
    key = rlz.oneof([rlz.string, rlz.integer])

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shapeof('arg', value_dtype)(self)


class MapValueOrDefaultForKey(ValueOp):

    input_type = [
        rules.map(dt.any, dt.any),
        rules.one_of((dt.string, dt.int_), name='key'),
        rules.value(name='default')
    ]

    def output_type(self):
        map_type = self.args[0].type()
        value_type = map_type.value_type
        default_type = self.default.type()

        if default_type is not dt.null and value_type != default_type:
            raise ValueError("default type: {}  must be the same "
                             "as the map value_type {}".format(
                                 default_type, value_type))
        return rules.shape_like(self.args[0], map_type.value_type)


class MapKeys(ValueOp):
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('arg')


class MapValues(ValueOp):
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('arg')


class MapConcat(ValueOp):
    left = rlz.value(dt.Map(dt.any, dt.any))
    right = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('left')


class StructField(ValueOp):
    arg = rlz.instanceof(ir.StructValue)  # TODO: use datatypes instead
    field = rlz.instanceof(six.string_types)

    def output_type(self):
        struct_type = self.arg.type()
        return rules.shape_like(self.arg, struct_type[self.field])


class Literal(ValueOp):
    value = rlz.noop
    dtype = rlz.datatype

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


class NullLiteral(ValueOp):
    """Typeless NULL literal"""

    value = rlz.instanceof(type(None))

    def equals(self, other, cache=None):
        return isinstance(other, NullLiteral)

    def output_type(self):
        return dt.null.scalar_type()

    def root_tables(self):
        return []


class ScalarParameter(ValueOp):
    parameter_counter = itertools.count()

    dtype = rlz.datatype
    counter = Arg(int, default=partial(next, parameter_counter))

    def __repr__(self):
        return '{}(name={!r}, dtype={})'.format(
            type(self).__name__, self.name, self.dtype
        )

    def resolve_name(self):
        return 'param_{:d}'.format(self.counter)

    def __repr__(self):
        return '{}(type={})'.format(type(self).__name__, self.type)

    def __hash__(self):
        return hash((self.type, self.counter))

    def output_type(self):
        return self.dtype.scalar_type()

    def equals(self, other, cache=None):
        return (
            isinstance(other, ScalarParameter) and
            self.counter == other.counter and
            self.type.equals(other.type, cache=cache)
        )

    def root_tables(self):
        return []

    def resolve_name(self):
        return self.name


# TODO: this class might be not used at all
class ExpressionList(Node):

    # FIXME
    # TODO rename
    exprs = rlz.noop

    def __init__(self, exprs):
        exprs = [as_value_expr(x) for x in exprs]
        super(ExpressionList, self).__init__(exprs)

    def root_tables(self):
        return distinct_roots(self.exprs)

    def output_type(self):
        return ir.ExprList


class ValueList(ValueOp):
    """Data structure for a list of value expressions"""

    values = rlz.noop

    def __init__(self, args):
        values = list(map(as_value_expr, args))
        super(ValueList, self).__init__(values)

    def root_tables(self):
        return distinct_roots(*self.values)

    def _make_expr(self):
        dtype = rules.highest_precedence_type(self.values)
        return ir.ListExpr(self, dtype=dtype)
