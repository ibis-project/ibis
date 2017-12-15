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

import toolz

from ibis.expr.types import TableColumn  # noqa

from ibis.expr.datatypes import HasSchema, Schema
from ibis.expr.rules import value, string, number, integer, boolean, list_of
from ibis.expr.types import (Node, as_value_expr, Expr,
                             ValueExpr, ColumnExpr, TableExpr,
                             ValueOp, _safe_repr, distinct_roots)
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.util as util


_table_names = ('t{:d}'.format(i) for i in itertools.count())


def genname():
    return next(_table_names)


class TableNode(Node):

    def get_type(self, name):
        return self.schema[name]

    def _make_expr(self):
        return TableExpr(self)

    def aggregate(self, this, metrics, by=None, having=None):
        return Aggregation(this, metrics, by=by, having=having)

    def sort_by(self, expr, sort_exprs):
        return Selection(expr, [], sort_keys=sort_exprs)


def find_all_base_tables(expr, memo=None):
    if memo is None:
        memo = {}

    node = expr.op()

    if isinstance(expr, TableExpr) and node.blocks():
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

    def __init__(self, schema, name=None):
        if name is None:
            name = genname()
        TableNode.__init__(self, [schema, name])
        HasSchema.__init__(self, schema, name=name)


class DatabaseTable(PhysicalTable):

    """

    """

    def __init__(self, name, schema, source):
        self.source = source

        TableNode.__init__(self, [name, schema, source])
        HasSchema.__init__(self, schema, name=name)

    def change_name(self, new_name):
        return type(self)(new_name, self.args[1], self.source)


class SQLQueryResult(TableNode, HasSchema):

    """
    A table sourced from the result set of a select query
    """

    def __init__(self, query, schema, source):
        self.query = query
        TableNode.__init__(self, [query, schema, source])
        HasSchema.__init__(self, schema)

    def blocks(self):
        return True


class TableArrayView(ValueOp):

    """
    (Temporary?) Helper operation class for SQL translation (fully formed table
    subqueries to be viewed as arrays)
    """

    def __init__(self, table):
        if not isinstance(table, TableExpr):
            raise com.ExpressionError('Requires table')

        schema = table.schema()
        if len(schema) > 1:
            raise com.ExpressionError('Table can only have a single column')

        self.table = table
        self.name = schema.names[0]

        Node.__init__(self, [table])

    def _make_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ctype.array_type()
        return klass(self, name=self.name)


class UnaryOp(ValueOp):

    input_type = [value]


class Cast(ValueOp):

    input_type = [value, rules.data_type]

    # see #396 for the issue preventing this
    # def resolve_name(self):
    #     return self.args[0].get_name()

    def output_type(self):
        # TODO: error handling for invalid casts
        return rules.shape_like(self.args[0], self.args[1])


class TypeOf(ValueOp):

    input_type = [value]
    output_type = rules.shape_like_arg(0, 'string')


class Negate(UnaryOp):

    input_type = [number]
    output_type = rules.type_of_arg(0)


class IsNull(UnaryOp):

    """
    Returns true if values are null

    Returns
    -------
    isnull : boolean with dimension of caller
    """

    output_type = rules.shape_like_arg(0, 'boolean')


class NotNull(UnaryOp):

    """
    Returns true if values are not null

    Returns
    -------
    notnull : boolean with dimension of caller
    """

    output_type = rules.shape_like_arg(0, 'boolean')


class ZeroIfNull(UnaryOp):

    output_type = rules.type_of_arg(0)


class IfNull(ValueOp):

    """
    Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """

    input_type = [value, value(name='ifnull_expr')]

    def output_type(self):
        args = self.args
        highest_type = rules.highest_precedence_type(args)
        return rules.shape_like(args[0], highest_type)


class NullIf(ValueOp):

    """
    Set values to NULL if they equal the null_if_expr
    """

    input_type = [value, value(name='null_if_expr')]
    output_type = rules.type_of_arg(0)


class NullIfZero(ValueOp):

    """
    Set values to NULL if they equal to zero. Commonly used in cases where
    divide-by-zero would produce an overflow or infinity.

    Equivalent to (value == 0).ifelse(ibis.NA, value)

    Returns
    -------
    maybe_nulled : type of caller
    """

    input_type = [number]
    output_type = rules.type_of_arg(0)


class IsNan(ValueOp):

    input_type = [rules.floating]
    output_type = rules.shape_like_arg(0, 'boolean')


class IsInf(ValueOp):

    input_type = [rules.floating]
    output_type = rules.shape_like_arg(0, 'boolean')


class CoalesceLike(ValueOp):

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column

    input_type = rules.varargs(rules.value)

    def output_type(self):
        # TODO: how much validation is necessary that the call is valid and can
        # succeed?
        first_value = self.args[0]

        if isinstance(first_value, ir.IntegerValue):
            out_type = 'int64'
        elif isinstance(first_value, ir.FloatingValue):
            out_type = 'double'
        else:
            out_type = first_value.type()

        return rules.shape_like_args(self.args, out_type)


class Coalesce(CoalesceLike):
    pass


class Greatest(CoalesceLike):
    pass


class Least(CoalesceLike):
    pass


class Abs(UnaryOp):

    """
    Absolute value
    """

    output_type = rules.type_of_arg(0)


def _ceil_floor_output(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        return arg._factory
    else:
        return rules.shape_like(arg, 'int64')


class Ceil(UnaryOp):

    """
    Round up to the nearest integer value greater than or equal to this value

    Returns
    -------
    ceiled : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """
    input_type = [number]
    output_type = _ceil_floor_output


class Floor(UnaryOp):

    """
    Round down to the nearest integer value less than or equal to this value

    Returns
    -------
    floored : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """

    input_type = [number]
    output_type = _ceil_floor_output


class Round(ValueOp):

    input_type = [value, integer(name='digits', optional=True)]

    def output_type(self):
        arg, digits = self.args
        if isinstance(arg, ir.DecimalValue):
            return arg._factory
        elif digits is None:
            return rules.shape_like(arg, 'int64')
        else:
            return rules.shape_like(arg, 'double')


class Clip(ValueOp):
    input_type = [value,
                  number(name='lower', allow_boolean=False, optional=True),
                  number(name='upper', allow_boolean=False, optional=True)]

    output_type = rules.type_of_arg(0)


class BaseConvert(ValueOp):

    input_type = [rules.one_of([integer, string]),
                  integer(name='from_base'),
                  integer(name='to_base')]
    output_type = rules.shape_like_flatargs('string')


class RealUnaryOp(UnaryOp):

    input_type = [number]
    output_type = rules.shape_like_arg(0, 'double')


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    # This is the Impala output for both integers and double/float
    output_type = rules.shape_like_arg(0, 'float')


class Sqrt(RealUnaryOp):
    pass


class Logarithm(RealUnaryOp):

    # superclass

    input_type = [number(allow_boolean=False)]


class Log(Logarithm):

    input_type = [number(allow_boolean=False),
                  number(name='base', optional=True)]


class Ln(Logarithm):

    """
    Natural logarithm
    """


class Log2(Logarithm):

    """
    Logarithm base 2
    """


class Log10(Logarithm):

    """
    Logarithm base 10
    """


class StringUnaryOp(UnaryOp):

    input_type = [string]
    output_type = rules.shape_like_arg(0, 'string')


class Uppercase(StringUnaryOp):

    """
    Convert string to all uppercase
    """
    pass


class Lowercase(StringUnaryOp):

    """
    Convert string to all lowercase
    """
    pass


class Reverse(StringUnaryOp):
    pass


class Strip(StringUnaryOp):

    """
    Remove whitespace from left and right sides of string
    """
    pass


class LStrip(StringUnaryOp):

    """
    Remove whitespace from left side of string
    """
    pass


class RStrip(StringUnaryOp):
    """
    Remove whitespace from right side of string
    """
    pass


class Capitalize(StringUnaryOp):
    pass


class Substring(ValueOp):

    input_type = [string, integer(name='start'),
                  integer(name='length', optional=True)]
    output_type = rules.shape_like_arg(0, 'string')


class StrRight(ValueOp):

    input_type = [string, integer(name='nchars')]
    output_type = rules.shape_like_arg(0, 'string')


class Repeat(ValueOp):

    input_type = [string, integer(name='times')]
    output_type = rules.shape_like_arg(0, 'string')


class StringFind(ValueOp):

    input_type = [string, string(name='substr'),
                  integer(name='start', optional=True, default=None),
                  integer(name='end', optional=True, default=None)]
    output_type = rules.shape_like_arg(0, 'int64')


class Translate(ValueOp):

    input_type = [string, string(name='from_str'), string(name='to_str')]
    output_type = rules.shape_like_arg(0, 'string')


class LPad(ValueOp):

    input_type = [string, integer(name='length'),
                  string(name='pad', optional=True)]
    output_type = rules.shape_like_arg(0, 'string')


class RPad(ValueOp):

    input_type = [string, integer(name='length'),
                  string(name='pad', optional=True)]
    output_type = rules.shape_like_arg(0, 'string')


class FindInSet(ValueOp):

    input_type = [string(name='needle'), list_of(string, min_length=1)]
    output_type = rules.shape_like_arg(0, 'int64')


class StringJoin(ValueOp):

    input_type = [string(name='sep'), list_of(string, min_length=1)]
    output_type = rules.shape_like_flatargs('string')


class BooleanValueOp(ValueOp):
    pass


class FuzzySearch(BooleanValueOp):

    input_type = [string, string(name='pattern')]
    output_type = rules.shape_like_arg(0, 'boolean')


class StringSQLLike(FuzzySearch):
    pass


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueOp):

    input_type = [string, string(name='pattern'), integer(name='index')]
    output_type = rules.shape_like_arg(0, 'string')


class RegexReplace(ValueOp):

    input_type = [string, string(name='pattern'), string(name='replacement')]
    output_type = rules.shape_like_arg(0, 'string')


class StringReplace(ValueOp):

    input_type = [string, string(name='pattern'), string(name='replacement')]
    output_type = rules.shape_like_arg(0, 'string')


class ParseURL(ValueOp):

    input_type = [string, rules.string_options(['PROTOCOL', 'HOST', 'PATH',
                                                'REF', 'AUTHORITY', 'FILE',
                                                'USERINFO', 'QUERY'],
                                               name='extract'),
                  string(name='key', optional=True)]
    output_type = rules.shape_like_arg(0, 'string')


class StringLength(UnaryOp):

    """
    Compute length of strings

    Returns
    -------
    length : int32
    """

    output_type = rules.shape_like_arg(0, 'int32')


class StringAscii(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class BinaryOp(ValueOp):

    """
    A binary operation

    """
    # Casting rules for type promotions (for resolving the output type) may
    # depend in some cases on the target backend.
    #
    # TODO: how will overflows be handled? Can we provide anything useful in
    # Ibis to help the user avoid them?

    input_type = [rules.value(name='left'), rules.value(name='right')]

    def __init__(self, left, right):
        super(BinaryOp, self).__init__(*self._maybe_cast_args(left, right))

    def _maybe_cast_args(self, left, right):
        return left, right

    def output_type(self):
        raise NotImplementedError


# ----------------------------------------------------------------------


class Reduction(ValueOp):

    input_type = [rules.column, boolean(name='where', optional=True)]
    _reduction = True


def is_reduction(expr):
    # Aggregations yield typed scalar expressions, since the result of an
    # aggregation is a single value. When creating an table expression
    # containing a GROUP BY equivalent, we need to be able to easily check
    # that we are looking at the result of an aggregation.
    #
    # As an example, the expression we are looking at might be something
    # like: foo.sum().log10() + bar.sum().log10()
    #
    # We examine the operator DAG in the expression to determine if there
    # are aggregations present.
    #
    # A bound aggregation referencing a separate table is a "false
    # aggregation" in a GROUP BY-type expression and should be treated a
    # literal, and must be computed as a separate query and stored in a
    # temporary variable (or joined, for bound aggregations with keys)
    def has_reduction(op):
        if getattr(op, '_reduction', False):
            return True

        for arg in op.args:
            if isinstance(arg, ir.ScalarExpr) and has_reduction(arg.op()):
                return True

        return False

    return has_reduction(expr.op() if isinstance(expr, ir.Expr) else expr)


class Count(Reduction):
    # TODO: count(col) takes down Impala, must always do count(*) in generated
    # SQL

    input_type = [rules.collection, boolean(name='where', optional=True)]

    # TODO: counts are actually table-level operations. Let's address
    # during the SQL generation exercise

    def output_type(self):
        return ir.Int64Scalar


def _sum_output_type(self):
    arg = self.args[0]
    if isinstance(arg, (ir.IntegerValue, ir.BooleanValue)):
        t = 'int64'
    elif isinstance(arg, ir.FloatingValue):
        t = 'double'
    elif isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    else:
        raise TypeError(arg)
    return t


def _mean_output_type(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    elif isinstance(arg, ir.NumericValue):
        t = 'double'
    else:
        raise NotImplementedError
    return t


def _array_reduced_type(self):
    return dt.Array(self.args[0].type())


class Sum(Reduction):

    output_type = rules.scalar_output(_sum_output_type)


class Mean(Reduction):

    output_type = rules.scalar_output(_mean_output_type)


def _coerce_integer_to_double_type(self):
    first_arg = self.args[0]
    first_arg_type = first_arg.type()
    if isinstance(first_arg_type, dt.Integer):
        result_type = dt.double
    else:
        result_type = first_arg_type
    return result_type


class Quantile(Reduction):

    input_type = [value,
                  number(name='quantile', allow_boolean=False),
                  rules.string_options(
                      ['linear', 'lower', 'higher',
                       'midpoint', 'nearest'],
                      name='interpolation',
                      default='linear')]

    output_type = rules.scalar_output(_coerce_integer_to_double_type)


class MultiQuantile(Quantile):

    input_type = [value,
                  rules.array(dt.double),
                  rules.string_options(
                      ['linear', 'lower', 'higher',
                       'midpoint', 'nearest'],
                      name='interpolation',
                      default='linear')]

    def output_type(self):
        return dt.Array(_coerce_integer_to_double_type(self)).scalar_type()


class VarianceBase(Reduction):

    input_type = [rules.column, boolean(name='where', optional=True),
                  rules.string_options(['sample', 'pop'],
                                       name='how', optional=True)]
    output_type = rules.scalar_output(_mean_output_type)


class StandardDev(VarianceBase):
    pass


class Variance(VarianceBase):
    pass


def _decimal_scalar_ctor(precision, scale):
    out_type = dt.Decimal(precision, scale)
    return out_type.scalar_type()


def _min_max_output_rule(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    else:
        t = arg.type()

    return t


class Max(Reduction):

    output_type = rules.scalar_output(_min_max_output_rule)


class Min(Reduction):

    output_type = rules.scalar_output(_min_max_output_rule)


class HLLCardinality(Reduction):

    """
    Approximate number of unique values using HyperLogLog algorithm. Impala
    offers the NDV built-in function for this.
    """

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        return ir.DoubleScalar


class GroupConcat(Reduction):

    input_type = [rules.column, string(name='sep', default=',')]
    # boolean(name='where', optional=True)]

    def output_type(self):
        return ir.StringScalar


class CMSMedian(Reduction):

    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    def output_type(self):
        # Scalar but type of caller
        return self.args[0].type().scalar_type()


# ----------------------------------------------------------------------
# Analytic functions


class AnalyticOp(ValueOp):
    pass


class WindowOp(ValueOp):

    def output_type(self):
        return self.args[0].type().array_type()

    def __init__(self, expr, window):
        from ibis.expr.window import propagate_down_window
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
        existing_window = self.args[1]
        new_window = existing_window.combine(window)
        return WindowOp(self.args[0], new_window)

    def root_tables(self):
        window = self.args[1]
        result = list(toolz.unique(
            itertools.chain(
                self.args[0]._root_tables(),
                distinct_roots(
                    *itertools.chain(window._order_by, window._group_by)
                )
            ),
            key=id
        ))
        return result


def is_analytic(expr, exclude_windows=False):
    def _is_analytic(op):
        if isinstance(op, (Reduction, AnalyticOp)):
            return True
        elif isinstance(op, WindowOp) and exclude_windows:
            return False

        for arg in op.args:
            if isinstance(arg, ir.Expr) and _is_analytic(arg.op()):
                return True

        return False

    return _is_analytic(expr.op())


class ShiftBase(AnalyticOp):

    input_type = [rules.column, rules.integer(name='offset', optional=True),
                  rules.value(name='default', optional=True)]
    output_type = rules.type_of_arg(0)


class Lag(ShiftBase):
    pass


class Lead(ShiftBase):
    pass


class RankBase(AnalyticOp):

    def output_type(self):
        return ir.Int64Column


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
    input_type = [rules.column]


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
    input_type = [rules.column]


class RowNumber(RankBase):

    """
    Compute row number starting from 0 after sorting by column expression

    Examples
    --------
    >>> import ibis
    >>> t = ibis.table([('values', 'int64')])
    >>> w = ibis.window(order_by=t.values)
    >>> row_num = ibis.row_number().over(w)
    >>> result = t[t.values, row_num.name('row_num')]

    Returns
    -------
    row_number : Int64Column, starting from 0
    """

    # Equivalent to SQL ROW_NUMBER()
    pass


class CumulativeOp(AnalyticOp):

    input_type = [rules.column]


class CumulativeSum(CumulativeOp):

    """
    Cumulative sum. Requires an order window.
    """

    output_type = rules.array_output(_sum_output_type)


class CumulativeMean(CumulativeOp):

    """
    Cumulative mean. Requires an order window.
    """

    output_type = rules.array_output(_mean_output_type)


class CumulativeMax(CumulativeOp):

    """
    Cumulative max. Requires an order window.
    """

    output_type = rules.array_output(_min_max_output_rule)


class CumulativeMin(CumulativeOp):

    """
    Cumulative min. Requires an order window.
    """

    output_type = rules.array_output(_min_max_output_rule)


class PercentRank(AnalyticOp):

    input_type = [rules.column]
    output_type = rules.shape_like_arg(0, 'double')


class NTile(AnalyticOp):

    input_type = [rules.column, rules.integer(name='buckets')]
    output_type = rules.shape_like_arg(0, 'int64')


class FirstValue(AnalyticOp):

    input_type = [rules.column]
    output_type = rules.type_of_arg(0)


class LastValue(AnalyticOp):

    input_type = [rules.column]
    output_type = rules.type_of_arg(0)


class NthValue(AnalyticOp):

    input_type = [rules.column, rules.integer]
    output_type = rules.type_of_arg(0)

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

    def __init__(self, table):
        self.table = table

        TableNode.__init__(self, [table])
        schema = self.table.schema()
        HasSchema.__init__(self, schema)

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

    def __init__(self, arg):
        self.arg = arg
        ValueOp.__init__(self, arg)

    def output_type(self):
        return type(self.arg)

    def count(self):
        """
        Only valid if the distinct contains a single column
        """
        return CountDistinct(self.arg)


class CountDistinct(Reduction):

    input_type = [rules.column, boolean(name='where', optional=True)]

    def output_type(self):
        return ir.Int64Scalar


# ---------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueOp):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)

    input_type = [rules.column(boolean)]

    @property
    def _reduction(self):
        roots = self.args[0]._root_tables()
        return len(roots) < 2

    def output_type(self):
        return ir.BooleanScalar if self._reduction else ir.BooleanColumn

    def negate(self):
        return NotAny(self.args[0])


class All(ValueOp):

    input_type = [rules.column(boolean)]
    _reduction = True

    def output_type(self):
        return ir.BooleanScalar

    def negate(self):
        return NotAll(self.args[0])


class NotAny(Any):

    def negate(self):
        return Any(self.args[0])


class NotAll(All):

    def negate(self):
        return All(self.args[0])


class CumulativeAny(CumulativeOp):

    """
    Cumulative any
    """

    output_type = rules.array_output(lambda self: 'boolean')


class CumulativeAll(CumulativeOp):

    """
    Cumulative all
    """

    output_type = rules.array_output(lambda self: 'boolean')


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

        if not self.base._can_compare(case_expr):
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

    input_type = [value(name='base'),
                  list_of(value, name='cases'),
                  list_of(value, name='results'),
                  value(name='default')]

    def __init__(self, base, cases, results, default):
        assert len(cases) == len(results)
        ValueOp.__init__(self, base, cases, results, default)

    def root_tables(self):
        base, cases, results, default = self.args
        all_exprs = [base] + cases + results
        if default is not None:
            all_exprs.append(default)
        return ir.distinct_roots(*all_exprs)

    def output_type(self):
        base, cases, results, default = self.args
        out_exprs = results + [default]
        typename = rules.highest_precedence_type(out_exprs)
        return rules.shape_like(base, typename)


class SearchedCase(ValueOp):

    input_type = [list_of(boolean, name='cases'),
                  list_of(value, name='results'),
                  value(name='default')]

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        ValueOp.__init__(self, cases, results, default)

    def root_tables(self):
        cases, results, default = self.args
        all_exprs = cases + results
        if default is not None:
            all_exprs.append(default)
        return ir.distinct_roots(*all_exprs)

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

    input_type = [boolean(name='bool_expr'),
                  value(name='true_expr'), value(name='false_null_expr')]

    def output_type(self):
        return rules.shape_like(self.args[0], self.args[1].type())


def _validate_join_tables(left, right):
    if not rules.is_table(left):
        raise TypeError('Can only join table expressions, got {} for '
                        'left table'.format(type(left).__name__))

    if not rules.is_table(right):
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
    # Validate join predicates. Each predicate must be valid jointly when
    # considering the roots of each input table
    from ibis.expr.analysis import CommonSubexpr
    validator = CommonSubexpr([left, right])
    validator.validate_all(predicates)


class Join(TableNode):

    _arg_names = ['left', 'right', 'predicates']

    def __init__(self, left, right, predicates):
        _validate_join_tables(left, right)
        (self.left,
         self.right,
         self.predicates) = _make_distinct_join_predicates(
            left, right, predicates)

        Node.__init__(self, [self.left, self.right, self.predicates])

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
            return ir.distinct_roots(self.left, self.right)


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

    def __init__(self, join_expr):
        assert isinstance(join_expr.op(), Join)
        self.join = join_expr

        TableNode.__init__(self, [join_expr])
        schema = self.join.op()._get_schema()
        HasSchema.__init__(self, schema)

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
    _arg_names = ['left', 'right', 'predicates', 'by']

    def __init__(self, left, right, predicates, by_predicates):
        _validate_join_tables(left, right)
        (self.left,
         self.right,
         self.predicates) = _make_distinct_join_predicates(
            left, right, predicates)
        self.by_predicates = _clean_join_predicates(
            self.left, self.right, by_predicates)

        Node.__init__(
            self, [self.left, self.right, self.predicates, self.by_predicates])


class Union(TableNode, HasSchema):

    def __init__(self, left, right, distinct=False):
        self.left = left
        self.right = right
        self.distinct = distinct

        TableNode.__init__(self, [left, right, distinct])
        self._validate()
        HasSchema.__init__(self, self.left.schema())

    def _validate(self):
        if not self.left.schema().equals(self.right.schema()):
            raise com.RelationError('Table schemas must be equal '
                                    'to form union')

    def blocks(self):
        return True


class Limit(TableNode):

    _arg_names = [None, 'n', 'offset']

    def __init__(self, table, n, offset=0):
        super(Limit, self).__init__([table, n, offset])
        self.table = table
        self.n = n
        self.offset = offset

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


class SortKey(ir.Node):

    _arg_names = ['by', 'ascending']

    def __init__(self, expr, ascending=True):
        if not rules.is_array(expr):
            raise com.ExpressionError('Must be an array/column expression')

        self.expr = expr
        self.ascending = ascending

        ir.Node.__init__(self, [self.expr, self.ascending])

    def __repr__(self):
        # Temporary
        rows = ['Sort key:',
                '  ascending: {0!s}'.format(self.ascending),
                util.indent(_safe_repr(self.expr), 2)]
        return '\n'.join(rows)

    def root_tables(self):
        return self.expr._root_tables()

    def _make_expr(self):
        return ir.SortExpr(self)

    def equals(self, other, cache=None):
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

    def __init__(self, table_expr):
        self.table = table_expr
        TableNode.__init__(self, [table_expr])
        HasSchema.__init__(self, table_expr.schema())

    def root_tables(self):
        # The dependencies of this operation are not walked, which makes the
        # table expression holding this relationally distinct from other
        # expressions, so things like self-joins are possible
        return [self]

    def blocks(self):
        return True


class Selection(TableNode, HasSchema):

    _arg_names = ['table', 'selections', 'predicates', 'sort_keys']

    def __init__(self, table_expr, proj_exprs=None, predicates=None,
                 sort_keys=None):
        import ibis.expr.analysis as L

        # Argument cleaning
        proj_exprs = util.promote_list(
            proj_exprs if proj_exprs is not None else []
        )
        clean_exprs, schema = self._get_schema(table_expr, proj_exprs)

        self.sort_keys = [
            to_sort_key(table_expr, k)
            for k in util.promote_list(
                sort_keys if sort_keys is not None else []
            )
        ]

        self.predicates = list(toolz.concat(map(
            L.flatten_predicate,
            predicates if predicates is not None else []
        )))

        dependent_exprs = clean_exprs + self.sort_keys

        table_expr._assert_valid(dependent_exprs)
        self._validate_predicates(table_expr)

        self.selections = clean_exprs
        self.table = table_expr

        HasSchema.__init__(self, schema)
        Node.__init__(self, [table_expr] + [self.selections] +
                      [self.predicates] + [self.sort_keys])

    def blocks(self):
        return bool(self.selections)

    def _validate_predicates(self, table):
        from ibis.expr.analysis import FilterValidator
        validator = FilterValidator([table])
        validator.validate_all(self.predicates)

    def _validate(self, table, exprs):
        # Need to validate that the column expressions are compatible with the
        # input table; this means they must either be scalar expressions or
        # array expressions originating from the same root table expression
        table._assert_valid(exprs)

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

            if isinstance(projection, ValueExpr):
                names.append(projection.get_name())
                types.append(projection.type())
            elif rules.is_table(projection):
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

    def substitute_table(self, table_expr):
        return Selection(table_expr, self.selections)

    def root_tables(self):
        return [self]

    def can_add_filters(self, wrapped_expr, predicates):
        pass

    def is_ancestor(self, other):
        if isinstance(other, ir.Expr):
            other = other.op()

        if self.equals(other):
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

    _arg_names = ['table', 'metrics', 'by', 'having',
                  'predicates', 'sort_keys']

    def __init__(self, table, metrics, by=None, having=None,
                 predicates=None, sort_keys=None):
        # For tables, like joins, that are not materialized
        self.table = table

        self.metrics = self._rewrite_exprs(metrics)

        by = [] if by is None else by
        self.by = self.table._resolve(by)

        self.having = [] if having is None else having

        self.predicates = [] if predicates is None else predicates

        # order by only makes sense with group by in an aggregation
        sort_keys = [] if not self.by or sort_keys is None else sort_keys
        self.sort_keys = [to_sort_key(self.table, k)
                          for k in util.promote_list(sort_keys)]

        self.by = self._rewrite_exprs(self.by)
        self.having = self._rewrite_exprs(self.having)
        self.predicates = self._rewrite_exprs(self.predicates)
        self.sort_keys = self._rewrite_exprs(self.sort_keys)

        self._validate()
        self._validate_predicates()

        TableNode.__init__(self, [table, self.metrics, self.by,
                                  self.having, self.predicates,
                                  self.sort_keys])

        schema = self._result_schema()
        HasSchema.__init__(self, schema)

    def _rewrite_exprs(self, what):
        from ibis.expr.analysis import substitute_parents
        what = util.promote_list(what)

        all_exprs = []
        for expr in what:
            if isinstance(expr, ir.ExprList):
                all_exprs.extend(expr.exprs())
            else:
                bound_expr = ir.bind_expr(self.table, expr)
                all_exprs.append(bound_expr)

        return [substitute_parents(x, past_projection=False)
                for x in all_exprs]

    def blocks(self):
        return True

    def substitute_table(self, table_expr):
        return Aggregation(table_expr, self.metrics, by=self.by,
                           having=self.having)

    def _validate(self):
        # All aggregates are valid
        for expr in self.metrics:
            if not rules.is_scalar(expr) or not is_reduction(expr):
                raise TypeError('Passed a non-aggregate expression: %s' %
                                _safe_repr(expr))

        for expr in self.having:
            if not isinstance(expr, ir.BooleanScalar):
                raise com.ExpressionError('Having clause must be boolean '
                                          'expression, was: {0!s}'
                                          .format(_safe_repr(expr)))

        # All non-scalar refs originate from the input table
        all_exprs = (self.metrics + self.by + self.having +
                     self.sort_keys)
        self.table._assert_valid(all_exprs)

    def _validate_predicates(self):
        from ibis.expr.analysis import FilterValidator
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

    def _result_schema(self):
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

    input_type = [number(name='left'), number(name='right')]

    def output_type(self):
        return rules.shape_like_args(self.args, 'double')


class FloorDivide(Divide):

    def output_type(self):
        return rules.shape_like_args(self.args, 'int64')


class LogicalBinaryOp(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, ir.BooleanValue):
            raise TypeError('Only valid with boolean data')
        return rules.shape_like_args(self.args, 'boolean')


class Not(UnaryOp):

    input_type = [rules.boolean]
    output_type = rules.shape_like_arg(0, 'boolean')


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
        if left._can_implicit_cast(right):
            return left, left._implicit_cast(right)

        if right._can_implicit_cast(left):
            return right._implicit_cast(left), right

        return left, right

    def output_type(self):
        # if not dt.comparable(self.left.type(), self.right):
        #     raise TypeError('Cannot compare argument types')
        self._assert_can_compare()
        return rules.shape_like_args(self.args, 'boolean')

    def _assert_can_compare(self):
        if not self.left._can_compare(self.right):
            raise TypeError('Cannot compare argument types')


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


class Between(BooleanValueOp):

    input_type = [
        rules.value,
        rules.value(name='lower_bound'),
        rules.value(name='upper_bound')
    ]

    def output_type(self):
        self._assert_can_compare()
        return rules.shape_like_args(self.args, 'boolean')

    def _assert_can_compare(self):
        expr, lower, upper = self.args
        if not expr._can_compare(lower) or not expr._can_compare(upper):
            raise TypeError('Arguments are not comparable')


class BetweenTime(Between):

    input_type = [
        rules.one_of((rules.timestamp, rules.time)),
        rules.one_of((rules.time, rules.string), name='lower_bound'),
        rules.one_of((rules.time, rules.string), name='upper_bound')
    ]


class Contains(BooleanValueOp):

    def __init__(self, value, options):
        self.value = as_value_expr(value)
        self.options = as_value_expr(options)
        super(Contains, self).__init__(self.value, self.options)

    def output_type(self):
        all_args = [self.value]

        options = self.options.op()
        if isinstance(options, ir.ValueList):
            all_args += options.values
        elif isinstance(self.options, ColumnExpr):
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

    def output_type(self):
        return ir.BooleanColumn


class TopK(ValueOp):

    def blocks(self):
        return True

    def __init__(self, arg, k, by=None):
        if by is None:
            by = arg.count()

        if not isinstance(arg, ColumnExpr):
            raise TypeError(arg)

        if not isinstance(k, int) or k < 0:
            raise ValueError('k must be positive integer, was: {0}'.format(k))

        self.arg = arg
        self.k = k
        self.by = by

        Node.__init__(self, [arg, k, by])

    def output_type(self):
        return TopKExpr


class Constant(ValueOp):
    pass


class TimestampNow(Constant):

    def output_type(self):
        return ir.TimestampScalar


class E(Constant):

    def output_type(self):
        return ir.DoubleScalar


class TemporalUnaryOp(UnaryOp):

    input_type = [rules.temporal]


class TimestampUnaryOp(UnaryOp):

    input_type = [rules.timestamp]


_truncate_units = [
    'Y', 'Q', 'M', 'D', 'J', 'W', 'H', 'MI'
]

_truncate_unit_aliases = {
    # year
    'YYYY': 'Y',
    'SYYYY': 'Y',
    'YEAR': 'Y',
    'YYY': 'Y',
    'YY': 'Y',

    # month
    'MONTH': 'M',
    'MON': 'M',

    # week
    'WW': 'W',

    # day of month

    # starting day of week

    # hour
    'HOUR': 'H',
    'HH24': 'H',

    # minute
    'MINUTE': 'MI',

    # second

    # millisecond

    # microsecond
}


def _truncate_unit_validate(unit):
    orig_unit = unit
    unit = unit.upper()

    # TODO: truncate autocompleter

    unit = _truncate_unit_aliases.get(unit, unit)
    valid_units = set(_truncate_units)

    if unit not in valid_units:
        raise com.IbisInputError('Passed unit {0} was not one of'
                                 ' {1}'.format(orig_unit,
                                               repr(valid_units)))

    return unit


class Truncate(ValueOp):

    input_type = [
        rules.timestamp,
        rules.string_options(_truncate_units, name='unit',
                             validator=_truncate_unit_validate)]
    output_type = rules.shape_like_arg(0, 'timestamp')


class Strftime(ValueOp):

    input_type = [rules.temporal, rules.string(name='format_str')]
    output_type = rules.shape_like_arg(0, 'string')


class ExtractTemporalField(TemporalUnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class ExtractTimestampField(TimestampUnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class ExtractYear(ExtractTemporalField):
    pass


class ExtractMonth(ExtractTemporalField):
    pass


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

    output_type = rules.shape_like_arg(0, 'time')


class Date(UnaryOp):

    output_type = rules.shape_like_arg(0, 'date')


class TimestampFromUNIX(ValueOp):

    input_type = [value, rules.string_options(['s', 'ms', 'us'], name='unit')]
    output_type = rules.shape_like_arg(0, 'timestamp')


class DecimalUnaryOp(UnaryOp):

    input_type = [rules.decimal]


class DecimalPrecision(DecimalUnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class DecimalScale(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class Hash(ValueOp):

    input_type = [value, rules.string_options(['fnv'], name='how')]
    output_type = rules.shape_like_arg(0, 'int64')


class TemporalSubtract(BinaryOp):

    def output_type(self):
        if self.args[0].type() == self.args[1].type():
            value_type = dt.Interval(self.output_unit, dt.int32)
        else:
            value_type = self.args[0].type()

        return rules.shape_like(self.args[0], value_type)


class DateSubtract(TemporalSubtract):

    input_type = [
        rules.date,
        rules.one_of([
            rules.date,
            rules.interval(units=['Y', 'Q', 'M', 'w', 'd'])
        ])
    ]
    output_unit = 'd'


class TimeSubtract(TemporalSubtract):

    input_type = [
        rules.time,
        rules.one_of([
            rules.time,
            rules.interval(units=['h', 'm', 's'])
        ])
    ]
    output_unit = 's'


class TimestampSubtract(TemporalSubtract):

    input_type = [
        rules.timestamp,
        rules.one_of([
            rules.timestamp,
            rules.interval
        ])
    ]
    output_unit = 's'


TimestampDelta = TimestampSubtract


class DateAdd(Add):

    input_type = [rules.date, rules.interval(units=['Y', 'Q', 'M', 'w', 'd'])]
    output_type = rules.shape_like_arg(0, 'date')


class TimeAdd(Add):

    input_type = [rules.time, rules.interval(units=['h', 'm', 's'])]
    output_type = rules.shape_like_arg(0, 'time')


class TimestampAdd(Add):

    input_type = [rules.timestamp, rules.interval]
    output_type = rules.shape_like_arg(0, 'timestamp')


class IntervalAdd(Add):

    input_type = [rules.interval, rules.interval]

    def output_type(self):
        left, right = self.args

        # should upconvert to the smaller unit?
        # value_type = left.to_unit(right.type().unit).type()
        return rules.shape_like(left, left.type())


class IntervalMultiply(Multiply):

    def output_type(self):
        helper = rules.IntervalPromoter(self.left, self.right, operator.mul)
        return helper.get_result()


class IntervalFloorDivide(FloorDivide):

    input_type = [rules.interval, number]

    def output_type(self):
        return rules.shape_like(self.args[0],  self.args[0].type())


class IntervalFromInteger(ValueOp):

    input_type = [
        rules.integer,
        rules.string_options([
            'Y', 'Q', 'M', 'w', 'd',
            'h', 'm', 's', 'ms', 'us', 'ns'
        ], name='unit')
    ]

    def output_type(self):
        arg, unit = self.args
        type = dt.Interval(unit, arg.type())
        return rules.shape_like(arg, type)


class ArrayLength(UnaryOp):

    input_type = [rules.array(dt.any)]
    output_type = rules.shape_like_arg(0, 'int64')


class ArraySlice(ValueOp):

    input_type = [
        rules.array(dt.any),
        rules.integer(name='start'),
        rules.integer(name='stop', optional=True)
    ]
    output_type = rules.type_of_arg(0)


class ArrayIndex(ValueOp):

    input_type = [rules.array(dt.any), rules.integer(name='index')]

    def output_type(self):
        value_type = self.args[0].type().value_type
        return rules.shape_like(self.args[0], value_type)


def _array_binop_invariant_output_type(self):
    """Check whether two arrays in an array OP array binary operation have
    the same type.
    """
    args = self.args
    left_type = args[0].type()
    right_type = args[1].type()
    if left_type != right_type:
        raise TypeError(
            'Array types must match exactly in a {} operation. '
            'Left type {} != Right type {}'.format(
                type(self).__name__, left_type, right_type
            )
        )
    return left_type


class ArrayConcat(ValueOp):

    input_type = [rules.array(dt.any), rules.array(dt.any)]

    def output_type(self):
        result_type = _array_binop_invariant_output_type(self)
        return rules.shape_like(self.args[0], result_type)


class ArrayRepeat(ValueOp):

    input_type = [rules.array(dt.any), integer(name='times')]

    def output_type(self):
        array_type = self.args[0].type()
        return rules.shape_like(self.args[0], array_type)


class ArrayCollect(Reduction):

    input_type = [rules.column]
    output_type = rules.scalar_output(_array_reduced_type)


class MapLength(ValueOp):

    input_type = [rules.map(dt.any, dt.any)]
    output_type = rules.shape_like_arg(0, 'int64')


class MapValueForKey(ValueOp):

    input_type = [
        rules.map(dt.any, dt.any),
        rules.one_of((dt.string, dt.int_), name='key')
    ]

    def output_type(self):
        map_type = self.args[0].type()
        return rules.shape_like(self.args[0], map_type.value_type)


class MapKeys(ValueOp):

    input_type = [rules.map(dt.any, dt.any)]
    output_type = rules.type_of_arg(0)


class MapValues(ValueOp):

    input_type = [rules.map(dt.any, dt.any)]
    output_type = rules.type_of_arg(0)


class MapConcat(ValueOp):

    input_type = [rules.map(dt.any, dt.any), rules.map(dt.any, dt.any)]
    output_type = rules.type_of_arg(0)


class StructField(ValueOp):

    input_type = [
        rules.struct,
        rules.instance_of(six.string_types, name='field')
    ]

    def output_type(self):
        struct_type = self.args[0].type()
        return rules.shape_like(self.args[0], struct_type[self.field])
