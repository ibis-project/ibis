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


import six
import toolz
import operator
import itertools
import collections

from functools import partial
from ibis.expr.schema import HasSchema, Schema

import ibis.util as util
import ibis.common as com
import ibis.compat as compat
import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.datatypes as dt

from collections import OrderedDict


def _safe_repr(x, memo=None):
    return x._repr(memo=memo) if isinstance(x, (ir.Expr, Node)) else repr(x)


# TODO: move to analysis
def distinct_roots(*expressions):
    roots = toolz.concat(
        expression._root_tables() for expression in expressions
    )
    return list(toolz.unique(roots, key=id))


class TypeSignature(OrderedDict):

    __slots__ = tuple()

    def __call__(self, *args, **kwargs):
        if len(args) > len(self.keys()):
            raise TypeError('takes {} positional arguments but {} were '
                            'given'.format(len(self.keys()), len(args)))

        result = []
        for i, (name, rule) in enumerate(self.items()):
            if i < len(args):
                if name in kwargs:
                    raise TypeError('got multiple values for argument'
                                    '{}'.format(name))
                value = rule.call(args[i])
            elif name in kwargs:
                value = rule.call(kwargs[name])
            else:
                value = rule.call()

            result.append((name, value))

        return result

    def names(self):
        return self.keys()


class OperationMeta(type):

    def __new__(cls, name, parents, attrs):
        signature = TypeSignature()

        # inherit from parent signatures
        for parent in parents:
            if hasattr(parent, 'signature'):
                signature.update(parent.signature)

        argnames = attrs.get('__slots__', tuple())
        for key in argnames:
            validator = attrs.pop(key)
            assert isinstance(validator, rlz.validator)
            signature[key] = validator

        attrs['signature'] = signature
        attrs['__slots__'] = tuple(signature.names()) + ('_expr_cached',)

        return super(OperationMeta, cls).__new__(cls, name, parents, attrs)


class Node(six.with_metaclass(OperationMeta, object)):

    def __init__(self, *args, **kwargs):
        self._expr_cached = None
        for name, value in self.signature(*args, **kwargs):
            setattr(self, name, value)
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
        return tuple(getattr(self, name) for name in self.signature.names())

    @property
    def argnames(self):
        return tuple(self.signature.names())


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

    __slots__ = 'name', 'table'

    name = rlz.instanceof(six.string_types + six.integer_types)
    table = rlz.table

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
    __slots__ = 'schema', 'name'

    schema = rlz.schema
    name = rlz.optional(rlz.instanceof(six.string_types), default=genname)


class DatabaseTable(PhysicalTable):
    __slots__ = 'name', 'schema', 'source'

    name = rlz.instanceof(six.string_types)
    schema = rlz.schema
    source = rlz.noop

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


class SQLQueryResult(TableNode, HasSchema):
    """A table sourced from the result set of a select query"""

    __slots__ = 'query', 'schema', 'source'

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
    __slots__ = 'table', 'name'

    table = rlz.table
    name = rlz.instanceof(six.string_types)

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
    __slots__ = 'arg',

    arg = rlz.any


class Cast(ValueOp):
    __slots__ = 'arg', 'to'

    arg = rlz.any
    to = rlz.datatype

    # see #396 for the issue preventing this
    # def resolve_name(self):
    #     return self.args[0].get_name()

    def output_type(self):
        return rlz.shapeof(self.arg, dtype=self.to)


class TypeOf(UnaryOp):
    output_type = rlz.shapeof('arg', dt.string)


class Negate(UnaryOp):
    __slots__ = 'arg',

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
    output_type = rlz.shapeof('arg', dt.boolean)


class ZeroIfNull(UnaryOp):
    output_type = rlz.typeof('arg')


class IfNull(ValueOp):
    """Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """
    __slots__ = 'arg', 'ifnull_expr'

    arg = rlz.any
    ifnull_expr = rlz.any
    output_type = rlz.shapeof('args')


class NullIf(ValueOp):
    """Set values to NULL if they equal the null_if_expr"""
    __slots__ = 'arg', 'null_if_expr'

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
    __slots__ = 'arg',

    arg = rlz.numeric
    output_type = rlz.typeof('arg')


class IsNan(ValueOp):
    __slots__ = 'arg',

    arg = rlz.floating
    output_type = rlz.shapeof('arg', dt.boolean)


class IsInf(ValueOp):
    __slots__ = 'arg',

    arg = rlz.floating
    output_type = rlz.shapeof('arg', dt.boolean)


class CoalesceLike(ValueOp):

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column
    __slots__ = 'arg',

    arg = rlz.listof(rlz.any)

    def output_type(self):
        first = self.arg[0]
        if isinstance(first, (ir.IntegerValue, ir.FloatingValue)):
            dtype = first.type().largest()
        else:
            dtype = first.type()

        # self.arg is a list of value expressions
        return rlz.shapeof(self.arg, dtype)


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
    __slots__ = 'arg',

    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shapeof(self.arg, dt.int64)


class Floor(UnaryOp):

    """
    Round down to the nearest integer value less than or equal to this value

    Returns
    -------
    floored : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """
    __slots__ = 'arg',

    arg = rlz.numeric

    def output_type(self):
        if isinstance(self.arg.type(), dt.Decimal):
            return self.arg._factory
        return rlz.shapeof(self.arg, dt.int64)


class Round(ValueOp):
    __slots__ = 'arg', 'digits'

    arg = rlz.numeric
    digits = rlz.optional(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None:
            return rlz.shapeof(self.arg, dt.int64)
        else:
            return rlz.shapeof(self.arg, dt.double)


class Clip(ValueOp):
    __slots__ = 'arg', 'lower', 'upper'

    arg = rlz.strict_numeric
    lower = rlz.optional(rlz.strict_numeric)
    upper = rlz.optional(rlz.strict_numeric)
    output_type = rlz.typeof('arg')


class BaseConvert(ValueOp):
    __slots__ = 'arg', 'from_base', 'to_base'

    arg = rlz.oneof([rlz.integer, rlz.string])
    from_base = rlz.integer
    to_base = rlz.integer

    def output_type(self):
        return rlz.shapeof(tuple(self.flat_args()), dt.string)


class RealUnaryOp(UnaryOp):
    __slots__ = 'arg',

    arg = rlz.numeric
    output_type = rlz.shapeof('arg', dt.double)


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    # This is the Impala output for both integers and double/float
    output_type = rlz.shapeof('arg', dt.float)


class Sqrt(RealUnaryOp):
    pass


class Logarithm(RealUnaryOp):
    __slots__ = 'arg',

    arg = rlz.strict_numeric


class Log(Logarithm):
    __slots__ = 'arg', 'base'

    arg = rlz.strict_numeric
    base = rlz.optional(rlz.strict_numeric)


class Ln(Logarithm):
    """Natural logarithm"""


class Log2(Logarithm):
    """Logarithm base 2"""


class Log10(Logarithm):
    """Logarithm base 10"""


class StringUnaryOp(UnaryOp):
    __slots__ = 'arg',

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
    __slots__ = 'arg', 'start', 'length'

    arg = rlz.string
    start = rlz.integer
    length = rlz.optional(rlz.integer)
    output_type = rlz.shapeof('arg', dt.string)


class StrRight(ValueOp):
    __slots__ = 'arg', 'nchars'

    arg = rlz.string
    nchars = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class Repeat(ValueOp):
    __slots__ = 'arg', 'times'

    arg = rlz.string
    times = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class StringFind(ValueOp):
    __slots__ = 'arg', 'substr', 'start', 'end'

    arg = rlz.string
    substr = rlz.string
    start = rlz.optional(rlz.integer)
    end = rlz.optional(rlz.integer)
    output_type = rlz.shapeof('arg', dt.int64)


class Translate(ValueOp):
    __slots__ = 'arg', 'from_str', 'to_str'

    arg = rlz.string
    from_str = rlz.string
    to_str = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class LPad(ValueOp):
    __slots__ = 'arg', 'length', 'pad'

    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class RPad(ValueOp):
    __slots__ = 'arg', 'length', 'pad'

    arg = rlz.string
    length = rlz.integer
    pad = rlz.optional(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class FindInSet(ValueOp):
    __slots__ = 'needle', 'values'

    needle = rlz.string
    values = rlz.listof(rlz.string, min_length=1)
    output_type = rlz.shapeof('needle', dt.int64)


class StringJoin(ValueOp):
    __slots__ = 'sep', 'arg'

    sep = rlz.string
    arg = rlz.listof(rlz.string, min_length=1)

    def output_type(self):
        return rlz.shapeof(tuple(self.flat_args()), dt.string)


class BooleanValueOp(object):
    pass


class FuzzySearch(ValueOp, BooleanValueOp):
    __slots__ = 'arg', 'pattern'

    arg = rlz.string
    pattern = rlz.string
    output_type = rlz.shapeof('arg', dt.boolean)


class StringSQLLike(FuzzySearch):
    __slots__ = 'arg', 'pattern', 'escape'

    arg = rlz.string
    pattern = rlz.string
    escape = rlz.optional(rlz.instanceof(six.string_types))


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueOp):
    __slots__ = 'arg', 'pattern', 'index'

    arg = rlz.string
    pattern = rlz.string
    index = rlz.integer
    output_type = rlz.shapeof('arg', dt.string)


class RegexReplace(ValueOp):
    __slots__ = 'arg', 'pattern', 'replacement'

    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class StringReplace(ValueOp):
    __slots__ = 'arg', 'pattern', 'replacement'

    arg = rlz.string
    pattern = rlz.string
    replacement = rlz.string
    output_type = rlz.shapeof('arg', dt.string)


class StringSplit(ValueOp):
    __slots__ = 'arg', 'delimiter'

    arg = rlz.string
    delimiter = rlz.string
    output_type = rlz.shapeof('arg', dt.Array(dt.string))


class StringConcat(ValueOp):
    __slots__ = 'arg',

    arg = rlz.listof(rlz.string)
    output_type = rlz.shapeof('arg', dt.string)


class ParseURL(ValueOp):
    __slots__ = 'arg', 'extract', 'key'

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
    """A binary operation"""

    # Casting rules for type promotions (for resolving the output type) may
    # depend in some cases on the target backend.
    #
    # TODO: how will overflows be handled? Can we provide anything useful in
    # Ibis to help the user avoid them?

    def __init__(self, left, right):
        super(BinaryOp, self).__init__(*self._maybe_cast_args(left, right))

    def _maybe_cast_args(self, left, right):
        return left, right


# ----------------------------------------------------------------------


class Reduction(ValueOp):
    _reduction = True


class Count(Reduction):
    __slots__ = 'arg', 'where'

    arg = rlz.instanceof((ir.ColumnExpr, ir.TableExpr))
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        return partial(ir.IntegerScalar, dtype=dt.int64)


class Arbitrary(Reduction):
    __slots__ = 'arg', 'how', 'where'

    arg = rlz.column(rlz.any)
    how = rlz.optional(rlz.isin({'first', 'last', 'heavy'}), default='first')
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class Sum(Reduction):
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.numeric)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest()
        return dtype.scalar_type()


class Mean(Reduction):
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.numeric)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest()
        else:
            dtype = dt.float64
        return dtype.scalar_type()


class Quantile(Reduction):
    __slots__ = 'arg', 'quantile', 'interpolation'

    arg = rlz.any
    quantile = rlz.strict_numeric
    interpolation = rlz.optional(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear'
    )

    def output_type(self):
        return dt.float64.scalar_type()


class MultiQuantile(Quantile):
    __slots__ = 'arg', 'quantile', 'interpolation'

    arg = rlz.any
    quantile = rlz.value(dt.Array(dt.float64))
    interpolation = rlz.optional(
        rlz.isin({'linear', 'lower', 'higher', 'midpoint', 'nearest'}),
        default='linear'
    )

    def output_type(self):
        return dt.Array(dt.float64).scalar_type()


class VarianceBase(Reduction):
    __slots__ = 'arg', 'how', 'where'

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
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class Min(Reduction):
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


class HLLCardinality(Reduction):

    """
    Approximate number of unique values using HyperLogLog algorithm. Impala
    offers the NDV built-in function for this.
    """
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        # return ir.DoubleScalar
        return partial(ir.IntegerScalar, dtype=dt.int64)


class GroupConcat(Reduction):
    __slots__ = 'arg', 'sep', 'where'

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
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)
    output_type = rlz.scalarof('arg')


# ----------------------------------------------------------------------
# Analytic functions


class AnalyticOp(ValueOp):
    pass


class WindowOp(ValueOp):
    __slots__ = 'expr', 'window'

    expr = rlz.noop
    window = rlz.noop
    output_type = rlz.arrayof('expr')

    argnames = False

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
    __slots__ = 'arg', 'offset', 'default'

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
    __slots__ = 'arg',
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
    __slots__ = 'arg',
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
    __slots__ = 'arg',
    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.BooleanValue):
            dtype = dt.int64
        else:
            dtype = self.arg.type().largest()
        return dtype.array_type()


class CumulativeMean(CumulativeOp):
    """Cumulative mean. Requires an order window."""
    __slots__ = 'arg',
    arg = rlz.column(rlz.numeric)

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            dtype = self.arg.type().largest()
        else:
            dtype = dt.float64
        return dtype.array_type()


class CumulativeMax(CumulativeOp):
    """Cumulative max. Requires an order window."""
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.arrayof('arg')


class CumulativeMin(CumulativeOp):
    """Cumulative min. Requires an order window."""
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.arrayof('arg')


class PercentRank(AnalyticOp):
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.shapeof('arg', dt.double)


class NTile(AnalyticOp):
    __slots__ = 'arg', 'buckets'
    arg = rlz.column(rlz.any)
    buckets = rlz.integer
    output_type = rlz.shapeof('arg', dt.int64)


class FirstValue(AnalyticOp):
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


class LastValue(AnalyticOp):
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.typeof('arg')


class NthValue(AnalyticOp):
    __slots__ = 'arg', 'nth'
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
    __slots__ = 'table',

    table = rlz.table

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
    __slots__ = 'arg',

    arg = rlz.noop
    output_type = rlz.typeof('arg')

    def count(self):
        """Only valid if the distinct contains a single column"""
        return CountDistinct(self.arg)


class CountDistinct(Reduction):
    __slots__ = 'arg', 'where'

    arg = rlz.column(rlz.any)
    where = rlz.optional(rlz.boolean)

    def output_type(self):
        return dt.int64.scalar_type()


# ---------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueOp):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)
    __slots__ = 'arg',

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
    __slots__ = 'arg',

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
    __slots__ = 'arg',
    arg = rlz.column(rlz.boolean)
    output_type = rlz.typeof('arg')


class CumulativeAll(CumulativeOp):
    __slots__ = 'arg',
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
    __slots__ = 'base', 'cases', 'results', 'default'

    base = rlz.any
    cases = rlz.listof(rlz.any)
    results = rlz.listof(rlz.any)
    default = rlz.any

    def _validate(self):
        assert len(self.cases) == len(self.results)

    def root_tables(self):
        all_exprs = [self.base] + self.cases + self.results + (
            [] if self.default is None else [self.default]
        )
        return distinct_roots(*all_exprs)

    def output_type(self):
        exprs = self.results + [self.default]
        return rlz.shapeof(self.base, dtype=exprs.type())


class SearchedCase(ValueOp):
    __slots__ = 'cases', 'results', 'default'

    cases = rlz.listof(rlz.boolean)
    results = rlz.listof(rlz.any)
    default = rlz.any

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        super(SearchedCase, self).__init__(cases, results, default)

    def root_tables(self):
        cases, results, default = self.args
        all_exprs = cases.values + results.values + (
            [] if default is None else [default]
        )
        return distinct_roots(*all_exprs)

    def output_type(self):
        exprs = self.results + [self.default]
        dtype = rlz.highest_precedence_dtype(exprs)
        return rlz.shapeof(self.cases, dtype)


class Where(ValueOp):

    """
    Ternary case expression, equivalent to

    bool_expr.case()
             .when(True, true_expr)
             .else_(false_or_null_expr)
    """
    __slots__ = 'bool_expr', 'true_expr', 'false_null_expr'

    bool_expr = rlz.boolean
    true_expr = rlz.any
    false_null_expr = rlz.any

    def output_type(self):
        return rlz.shapeof(self.bool_expr, self.true_expr.type())


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
    __slots__ = 'left', 'right', 'predicates'

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
    __slots__ = 'join',

    join = rlz.table

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
    __slots__ = 'left', 'right', 'predicates', 'by',

    left = rlz.noop
    right = rlz.noop
    predicates = rlz.noop
    by = rlz.optional(rlz.noop)

    def __init__(self, left, right, predicates, by):
        super(AsOfJoin, self).__init__(left, right, predicates)
        self.by = _clean_join_predicates(self.left, self.right, by)


class Union(TableNode, HasSchema):
    __slots__ = 'left', 'right', 'distinct'

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
    __slots__ = 'table', 'n', 'offset'

    table = rlz.table
    n = rlz.validator(int)
    offset = rlz.validator(int)

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
    __slots__ = 'expr', 'ascending'

    expr = rlz.column(rlz.any)
    ascending = rlz.optional(rlz.validator(bool), default=True)

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


class DeferredSortKey(object):

    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = parent._ensure_expr(self.what)
        return SortKey(what, ascending=self.ascending).to_expr()


class SelfReference(TableNode, HasSchema):
    __slots__ = 'table',

    table = rlz.table

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
    __slots__ = 'table', 'selections', 'predicates', 'sort_keys'

    table = rlz.table
    selections = rlz.noop
    predicates = rlz.noop
    sort_keys = rlz.noop

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
    __slots__ = 'table', 'metrics', 'by', 'having', 'predicates', 'sort_keys'

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
    __slots__ = 'left', 'right'

    left = rlz.numeric
    right = rlz.numeric


class Add(NumericBinaryOp):
    output_type = rlz.binopof('args', operator.add)


class Multiply(NumericBinaryOp):
    output_type = rlz.binopof('args', operator.mul)


class Power(NumericBinaryOp):

    def output_type(self):
        if util.all_of(self.args, ir.IntegerValue):
            return rlz.shapeof(self.args, dt.float64)
        else:
            return rlz.shapeof(self.args)


class Subtract(NumericBinaryOp):
    output_type = rlz.binopof('args', operator.sub)


class Divide(NumericBinaryOp):
    output_type = rlz.shapeof('args', dt.float64)


class FloorDivide(Divide):
    output_type = rlz.shapeof('args', dt.int64)


class LogicalBinaryOp(BinaryOp):
    __slots__ = 'left', 'right'

    left = rlz.boolean
    right = rlz.boolean
    output_type = rlz.shapeof('args', dt.boolean)


class Not(UnaryOp):
    __slots__ = 'arg',

    arg = rlz.boolean
    output_type = rlz.shapeof('arg', dt.boolean)


class Modulus(NumericBinaryOp):
    output_type = rlz.binopof('args', operator.mod)


class And(LogicalBinaryOp):
    pass


class Or(LogicalBinaryOp):
    pass


class Xor(LogicalBinaryOp):
    pass


class Comparison(BinaryOp, BooleanValueOp):
    __slots__ = 'left', 'right'

    left = rlz.any
    right = rlz.any

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
        return rlz.shapeof(self.args, dt.boolean)


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
    __slots__ = 'arg', 'lower_bound', 'upper_bound'

    arg = rlz.any
    lower_bound = rlz.any
    upper_bound = rlz.any

    def output_type(self):
        arg, lower, upper = self.args

        if not (rlz.comparable(arg, lower) and rlz.comparable(arg, upper)):
            raise TypeError('Arguments are not comparable')

        return rlz.shapeof(self.args, dt.boolean)


class BetweenTime(Between):
    __slots__ = 'arg', 'lower_bound', 'upper_bound'

    arg = rlz.oneof([rlz.timestamp, rlz.time])
    lower_bound = rlz.oneof([rlz.time, rlz.string])
    upper_bound = rlz.oneof([rlz.time, rlz.string])


class Contains(ValueOp, BooleanValueOp):
    __slots__ = 'value', 'options'

    value = rlz.any
    options = rlz.listof(rlz.any)

    def output_type(self):
        all_args = [self.value] + self.options
        return rlz.shapeof(all_args, dt.boolean)


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
    __slots__ = 'expr',

    expr = rlz.noop

    def output_type(self):
        return dt.boolean.array_type()


class TopK(ValueOp):
    __slots__ = 'arg', 'k', 'by'

    arg = rlz.noop
    k = rlz.noop
    by = rlz.noop

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
        return partial(ir.FloatingScalar, dtype=dt.float64)


class TemporalUnaryOp(UnaryOp):
    __slots__ = 'arg',

    arg = rlz.temporal


class TimestampUnaryOp(UnaryOp):
    __slots__ = 'arg',

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
    __slots__ = 'arg', 'unit'

    arg = rlz.timestamp
    unit = rlz.isin(_timestamp_units)
    output_type = rlz.shapeof('arg', dt.timestamp)


class DateTruncate(ValueOp):
    __slots__ = 'arg', 'unit'

    arg = rlz.date
    unit = rlz.isin(_date_units)
    output_type = rlz.shapeof('arg', dt.date)


class TimeTruncate(ValueOp):
    __slots__ = 'arg', 'unit'

    arg = rlz.time
    unit = rlz.isin(_time_units)
    output_type = rlz.shapeof('arg', dt.time)


class Strftime(ValueOp):
    __slots__ = 'arg', 'format_str'

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
    __slots__ = 'arg',
    arg = rlz.oneof([rlz.date, rlz.timestamp])
    output_type = rlz.shapeof('arg', dt.int32)


class DayOfWeekName(UnaryOp):
    __slots__ = 'arg',
    arg = rlz.oneof([rlz.date, rlz.timestamp])
    output_type = rlz.shapeof('arg', dt.string)


class DayOfWeekNode(Node):
    __slots__ = 'arg',
    arg = rlz.oneof([rlz.date, rlz.timestamp])

    def output_type(self):
        return ir.DayOfWeek


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
    output_type = rlz.shapeof('arg', dt.time)


class Date(UnaryOp):
    output_type = rlz.shapeof('arg', dt.date)


class TimestampFromUNIX(ValueOp):
    __slots__ = 'arg', 'unit'
    arg = rlz.any
    unit = rlz.isin(['s', 'ms', 'us'])
    output_type = rlz.shapeof('arg', dt.timestamp)


class DecimalUnaryOp(UnaryOp):
    __slots__ = 'arg',
    arg = rlz.decimal


class DecimalPrecision(DecimalUnaryOp):

    output_type = rlz.shapeof('arg', dt.int32)


class DecimalScale(UnaryOp):

    output_type = rlz.shapeof('arg', dt.int32)


class Hash(ValueOp):
    __slots__ = 'arg', 'how'
    arg = rlz.any
    how = rlz.isin({'fnv'})
    output_type = rlz.shapeof('arg', dt.int64)


class DateAdd(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.date
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'})
    output_type = rlz.shapeof('left')


class DateSub(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.date
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D'})
    output_type = rlz.shapeof('left')


class DateDiff(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.date
    right = rlz.date
    output_type = rlz.shapeof('left', dt.Interval('D'))


class TimeAdd(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.time
    right = rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'})
    output_type = rlz.shapeof('left')


class TimeSub(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.time
    right = rlz.interval(units={'h', 'm', 's', 'ms', 'us', 'ns'})
    output_type = rlz.shapeof('left')


class TimeDiff(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.time
    right = rlz.time
    output_type = rlz.shapeof('left', dt.Interval('s'))


class TimestampAdd(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.timestamp
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D',
                                'h', 'm', 's', 'ms', 'us', 'ns'})
    output_type = rlz.shapeof('left')


class TimestampSub(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.timestamp
    right = rlz.interval(units={'Y', 'Q', 'M', 'W', 'D',
                                'h', 'm', 's', 'ms', 'us', 'ns'})
    output_type = rlz.shapeof('left')


class TimestampDiff(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.timestamp
    right = rlz.timestamp
    output_type = rlz.shapeof('left', dt.Interval('s'))


class IntervalAdd(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.interval
    right = rlz.interval

    def output_type(self):
        args = [arg.cast(arg.type().value_type) for arg in self.args]
        expr = rlz.binopof(args, operator.add)(self)
        dtype = dt.Interval(self.left.type().unit, expr.type())
        return rlz.shapeof(self.args, dtype=dtype)


class IntervalMultiply(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.interval
    right = rlz.numeric

    def output_type(self):
        args = [self.left.cast(self.left.type().value_type), self.right]
        expr = rlz.binopof(args, operator.mul)(self)
        dtype = dt.Interval(self.left.type().unit, expr.type())
        return rlz.shapeof(self.args, dtype=dtype)


class IntervalFloorDivide(BinaryOp):
    __slots__ = 'left', 'right'
    left = rlz.interval
    right = rlz.numeric
    output_type = rlz.shapeof('left')


class IntervalFromInteger(ValueOp):
    __slots__ = 'arg', 'unit'

    arg = rlz.integer
    unit = rlz.isin(['Y', 'Q', 'M', 'W', 'D',
                     'h', 'm', 's', 'ms', 'us', 'ns'])

    @property
    def resolution(self):
        return dt.Interval(self.unit).resolution

    def output_type(self):
        dtype = dt.Interval(self.unit, self.arg.type())
        return rlz.shapeof(self.arg, dtype=dtype)


class ArrayLength(UnaryOp):
    __slots__ = 'arg',
    arg = rlz.value(dt.Array(dt.any))
    output_type = rlz.shapeof('arg', dt.int64)


class ArraySlice(ValueOp):
    __slots__ = 'arg', 'start', 'stop'
    arg = rlz.value(dt.Array(dt.any))
    start = rlz.integer
    stop = rlz.optional(rlz.integer)
    output_type = rlz.typeof('arg')


class ArrayIndex(ValueOp):
    __slots__ = 'arg', 'index'
    arg = rlz.value(dt.Array(dt.any))
    index = rlz.integer

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shapeof(self.arg, value_dtype)


class ArrayConcat(ValueOp):
    __slots__ = 'left', 'right'
    left = rlz.value(dt.Array(dt.any))
    right = rlz.value(dt.Array(dt.any))
    output_type = rlz.shapeof('left')

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
    __slots__ = 'arg', 'times'
    arg = rlz.value(dt.Array(dt.any))
    times = rlz.integer
    output_type = rlz.typeof('arg')


class ArrayCollect(Reduction):
    __slots__ = 'arg',
    arg = rlz.column(rlz.any)
    output_type = rlz.scalarof('arg')


class MapLength(ValueOp):
    __slots__ = 'arg',
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.shapeof('arg', dt.int64)


class MapValueForKey(ValueOp):
    __slots__ = 'arg', 'key'
    arg = rlz.value(dt.Map(dt.any, dt.any))
    key = rlz.oneof([rlz.string, rlz.integer])

    def output_type(self):
        value_dtype = self.arg.type().value_type
        return rlz.shapeof(self.arg, value_dtype)


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
    __slots__ = 'arg',
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('arg')


class MapValues(ValueOp):
    __slots__ = 'arg',
    arg = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('arg')


class MapConcat(ValueOp):
    __slots__ = 'left', 'right'
    left = rlz.value(dt.Map(dt.any, dt.any))
    right = rlz.value(dt.Map(dt.any, dt.any))
    output_type = rlz.typeof('left')


class StructField(ValueOp):
    __slots__ = 'arg', 'field'
    arg = rlz.value(dt.Struct)
    field = rlz.instanceof(six.string_types)

    def output_type(self):
        struct_dtype = self.arg.type()
        value_dtype = struct_dtype[self.field]
        return rlz.shapeof(self.arg, value_dtype)


class Literal(ValueOp):
    __slots__ = 'value', 'dtype'
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


class NullLiteral(Literal):
    """Typeless NULL literal"""

    __slots__ = 'value', 'dtype'

    value = rlz.optional(rlz.instanceof(type(None)), default=None)
    dtype = rlz.optional(rlz.instanceof(dt.Null), default=dt.null)


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


class ExpressionList(Node):
    """Data structure for a list of arbitrary expressions"""

    __slots__ = 'exprs',

    exprs = rlz.noop

    def __init__(self, values):
        values = list(map(rlz.any, values))
        super(ExpressionList, self).__init__(values)

    def root_tables(self):
        return distinct_roots(self.exprs)

    def output_type(self):
        return ir.ExprList


class ValueList(ValueOp):
    """Data structure for a list of value expressions"""

    __slots__ = 'values',

    values = rlz.noop
    argnames = False  # disable showing argnames in repr

    def __init__(self, values):
        values = list(map(rlz.any, values))
        super(ValueList, self).__init__(values)

    def root_tables(self):
        return distinct_roots(*self.values)

    def _make_expr(self):
        dtype = rlz.highest_precedence_dtype(self.values)
        return ir.ListExpr(self, dtype=dtype)
