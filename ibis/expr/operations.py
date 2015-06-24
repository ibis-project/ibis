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

from ibis.common import RelationError, ExpressionError
from ibis.expr.types import (Node,
                             ValueExpr, ScalarExpr, ArrayExpr, TableExpr,
                             ArrayNode, TableNode, ValueNode,
                             HasSchema, _safe_repr)
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.util as util


py_string = basestring


def is_table(e):
    return isinstance(e, TableExpr)


def is_array(e):
    return isinstance(e, ArrayExpr)


def is_scalar(e):
    return isinstance(e, ScalarExpr)


def is_collection(expr):
    return isinstance(expr, (ArrayExpr, TableExpr))


def as_value_expr(val):
    if not isinstance(val, ir.Expr):
        if isinstance(val, (tuple, list)):
            val = sequence(val)
        else:
            val = literal(val)

    return val


def table(schema, name=None):
    if not isinstance(schema, ir.Schema):
        if isinstance(schema, list):
            schema = ir.Schema.from_tuples(schema)
        else:
            schema = ir.Schema.from_dict(schema)

    node = UnboundTable(schema, name=name)
    return TableExpr(node)


def literal(value):
    """
    Create a scalar expression from a Python value

    Parameters
    ----------
    value : some Python basic type

    Returns
    -------
    lit_value : value expression, type depending on input value
    """
    if value is None or value is null:
        return null()
    else:
        return ir.Literal(value).to_expr()


def timestamp(value):
    """
    Returns a timestamp literal if value is likely coercible to a timestamp
    """
    if isinstance(value, py_string):
        from pandas import Timestamp
        value = Timestamp(value)
    op = ir.Literal(value)
    return ir.TimestampScalar(op)


_NULL = None


def null():
    global _NULL
    if _NULL is None:
        _NULL = ir.NullScalar(NullLiteral())

    return _NULL


def sequence(values):
    """
    Wrap a list of Python values as an Ibis sequence type

    Parameters
    ----------
    values : list
      Should all be None or the same type

    Returns
    -------
    seq : Sequence
    """
    return ValueList(values).to_expr()


class NullLiteral(ValueNode):

    """
    Typeless NULL literal
    """

    def __init__(self):
        return

    @property
    def args(self):
        return [None]

    def equals(self, other):
        return isinstance(other, NullLiteral)

    def output_type(self):
        return ir.NullScalar

    def root_tables(self):
        return []


class ValueList(ArrayNode):

    """
    Data structure for a list of value expressions
    """

    def __init__(self, args):
        self.values = [as_value_expr(x) for x in args]
        Node.__init__(self, [self.values])

    def root_tables(self):
        return ir.distinct_roots(*self.values)

    def to_expr(self):
        return ir.ListExpr(self)


class PhysicalTable(ir.BlockingTableNode, HasSchema):

    pass


class UnboundTable(PhysicalTable):

    def __init__(self, schema, name=None):
        TableNode.__init__(self, [schema, name])
        HasSchema.__init__(self, schema, name=name)


class DatabaseTable(PhysicalTable):

    """

    """

    def __init__(self, name, schema, source):
        self.source = source

        TableNode.__init__(self, [name, schema, source])
        HasSchema.__init__(self, schema, name=name)


class SQLQueryResult(ir.BlockingTableNode, HasSchema):

    """
    A table sourced from the result set of a select query
    """

    def __init__(self, query, schema, source):
        self.query = query
        TableNode.__init__(self, [query, schema, source])
        HasSchema.__init__(self, schema)


class TableColumn(ArrayNode):

    """
    Selects a column from a TableExpr
    """

    def __init__(self, name, table_expr):
        Node.__init__(self, [name, table_expr])

        if name not in table_expr.schema():
            raise KeyError("'{0}' is not a field".format(name))

        self.name = name
        self.table = table_expr

    def parent(self):
        return self.table

    def resolve_name(self):
        return self.name

    def root_tables(self):
        return self.table._root_tables()

    def to_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ir.array_type(ctype)
        return klass(self, name=self.name)


class TableArrayView(ArrayNode):

    """
    (Temporary?) Helper operation class for SQL translation (fully formed table
    subqueries to be viewed as arrays)
    """

    def __init__(self, table):
        if not isinstance(table, TableExpr):
            raise ExpressionError('Requires table')

        schema = table.schema()
        if len(schema) > 1:
            raise ExpressionError('Table can only have a single column')

        self.table = table
        self.name = schema.names[0]

        Node.__init__(self, [table])

    def root_tables(self):
        return self.table._root_tables()

    def to_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ir.array_type(ctype)
        return klass(self, name=self.name)


class UnaryOp(ValueNode):

    def __init__(self, arg):
        self.arg = arg
        ValueNode.__init__(self, [arg])


class Cast(ValueNode):

    def __init__(self, arg, target_type):
        self._ensure_value(arg)

        self.arg = arg
        self.target_type = ir._validate_type(target_type.lower())
        ValueNode.__init__(self, [arg, self.target_type])

    def resolve_name(self):
        return self.arg.get_name()

    def output_type(self):
        # TODO: error handling for invalid casts
        return rules.shape_like(self.arg, self.target_type)


class Negate(UnaryOp):

    def output_type(self):
        return type(self.arg)


class IsNull(UnaryOp):

    """
    Returns true if values are null

    Returns
    -------
    isnull : boolean with dimension of caller
    """

    def output_type(self):
        return rules.shape_like(self.arg, 'boolean')


class NotNull(UnaryOp):

    """
    Returns true if values are not null

    Returns
    -------
    notnull : boolean with dimension of caller
    """

    def output_type(self):
        return rules.shape_like(self.arg, 'boolean')


class ZeroIfNull(UnaryOp):

    def output_type(self):
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif isinstance(self.arg, ir.FloatingValue):
            # Impala upcasts float to double in this op
            return rules.shape_like(self.arg, 'double')
        elif isinstance(self.arg, ir.IntegerValue):
            return rules.shape_like(self.arg, 'int64')
        else:
            raise NotImplementedError


class IfNull(ValueNode):

    """
    Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """

    def __init__(self, value, ifnull_expr):
        self.value = as_value_expr(value)
        self.ifnull_expr = as_value_expr(ifnull_expr)
        ValueNode.__init__(self, [self.value, self.ifnull_expr])

    def output_type(self):
        return self.value._factory


class NullIf(ValueNode):

    """
    Set values to NULL if they equal the null_if_expr
    """

    def __init__(self, value, null_if_expr):
        self.value = as_value_expr(value)
        self.null_if_expr = as_value_expr(null_if_expr)
        ValueNode.__init__(self, [self.value, self.null_if_expr])

    def output_type(self):
        return self.value._factory


def _coalesce_upcast(self):
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


class CoalesceLike(ValueNode):

    def __init__(self, args):
        if len(args) == 0:
            raise ValueError('Must provide at least one value')

        self.values = [as_value_expr(x) for x in args]
        ValueNode.__init__(self, self.values)

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column

    output_type = _coalesce_upcast

    def root_tables(self):
        return ir.distinct_roots(*self.args)


class Coalesce(CoalesceLike):
    pass


class Greatest(CoalesceLike):
    pass


class Least(CoalesceLike):
    pass


def _numeric_same_type(self):
    if not isinstance(self.arg, ir.NumericValue):
        raise TypeError('Only valid for numeric types')
    return self.arg._factory


class Abs(UnaryOp):

    """
    Absolute value
    """

    output_type = _numeric_same_type


def _ceil_floor_output(self):
    if not isinstance(self.arg, ir.NumericValue):
        raise TypeError('Only valid for numeric types')

    if isinstance(self.arg, ir.DecimalValue):
        return self.arg._factory
    else:
        return rules.shape_like(self.arg, 'int32')


class Ceil(UnaryOp):

    """
    Round up to the nearest integer value greater than or equal to this value

    Returns
    -------
    ceiled : type depending on input
      Decimal values: yield decimal
      Other numeric values: yield integer (int32)
    """

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

    output_type = _ceil_floor_output


class Round(ValueNode):

    def __init__(self, arg, digits=None):
        self.arg = arg
        self.digits = validate_int(digits)
        ValueNode.__init__(self, [self.arg, self.digits])

    def output_type(self):
        validate_numeric(self.arg)
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None or self.digits == 0:
            return rules.shape_like(self.arg, 'int64')
        else:
            return rules.shape_like(self.arg, 'double')


def validate_int(x):
    if x is not None and not isinstance(x, int):
        raise ValueError('Value must be an integer')

    return x


def validate_numeric(x):
    if not isinstance(x, ir.NumericValue):
        raise TypeError('Only implemented for numeric types')


class RealUnaryOp(UnaryOp):

    _allow_boolean = True

    def output_type(self):
        if not isinstance(self.arg, ir.NumericValue):
            raise TypeError('Only implemented for numeric types')
        elif (isinstance(self.arg, ir.BooleanValue)
              and not self._allow_boolean):
            raise TypeError('Not implemented for boolean types')

        return rules.shape_like(self.arg, 'double')


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    def output_type(self):
        return rules.shape_like(self.arg, 'int32')


class Sqrt(RealUnaryOp):
    pass


class Log(RealUnaryOp):

    _allow_boolean = False

    def __init__(self, arg, base=None):
        self.base = base
        RealUnaryOp.__init__(self, arg)


class Ln(RealUnaryOp):

    """
    Natural logarithm
    """

    _allow_boolean = False


class Log2(RealUnaryOp):

    """
    Logarithm base 2
    """

    _allow_boolean = False


class Log10(RealUnaryOp):

    """
    Logarithm base 10
    """

    _allow_boolean = False


def _string_output(self):
    if not isinstance(self.arg, ir.StringValue):
        raise TypeError('Only implemented for string types')
    return rules.shape_like(self.arg, 'string')


def _bool_output(self):
    return rules.shape_like(self.arg, 'boolean')


def _int_output(self):
    return rules.shape_like(self.arg, 'int32')


class StringUnaryOp(UnaryOp):

    output_type = _string_output


class Uppercase(StringUnaryOp):
    pass


class Lowercase(StringUnaryOp):
    pass


class Reverse(StringUnaryOp):
    pass


class Strip(StringUnaryOp):
    pass


class LStrip(StringUnaryOp):
    pass


class RStrip(StringUnaryOp):
    pass


class Substring(ValueNode):

    def __init__(self, arg, start, length=None):
        self.arg = arg
        self.start = start
        self.length = length
        ValueNode.__init__(self, [self.arg, self.start, self.length])

    output_type = _string_output


class StrRight(ValueNode):

    def __init__(self, arg, nchars):
        self.arg = arg
        self.nchars = as_value_expr(nchars)
        ValueNode.__init__(self, [self.arg, self.nchars])

    output_type = _string_output


class Repeat(ValueNode):

    def __init__(self, arg, n):
        self.arg = arg
        self.n = as_value_expr(n)
        ValueNode.__init__(self, [self.arg, self.n])

    output_type = _string_output


class StringFind(ValueNode):

    def __init__(self, arg, substr):
        self.arg = arg
        self.substr = as_value_expr(substr)
        ValueNode.__init__(self, [self.arg, self.substr])

    output_type = _int_output


class Translate(ValueNode):

    def __init__(self, arg, from_str, to_str):
        self.arg = arg
        self.from_str = as_value_expr(from_str)
        self.to_str = as_value_expr(to_str)
        ValueNode.__init__(self, [self.arg, self.from_str, self.to_str])

    output_type = _string_output


class Locate(ValueNode):

    def __init__(self, arg, substr, pos=0):
        self.arg = arg
        self.substr = as_value_expr(substr)
        self.pos = pos
        ValueNode.__init__(self, [self.arg, self.substr, self.pos])

    output_type = _int_output


class LPad(ValueNode):

    def __init__(self, arg, length, pad):
        self.arg = arg
        self.length = as_value_expr(length)
        self.pad = as_value_expr(pad)
        ValueNode.__init__(self, [self.arg, self.length, self.pad])

    output_type = _string_output


class RPad(ValueNode):

    def __init__(self, arg, length, pad):
        self.arg = arg
        self.length = as_value_expr(length)
        self.pad = as_value_expr(pad)
        ValueNode.__init__(self, [self.arg, self.length, self.pad])

    output_type = _string_output


class FindInSet(ValueNode):

    def __init__(self, arg, str_list):
        self.arg = arg
        self.str_list = [as_value_expr(x) for x in str_list]
        ValueNode.__init__(self, [self.arg, self.str_list])

    output_type = _int_output


class StringJoin(ValueNode):

    def __init__(self, arg, strings):
        self.arg = arg
        self.strings = [as_value_expr(x) for x in strings]
        ValueNode.__init__(self, [self.arg, self.strings])

    output_type = _string_output


class BooleanValueOp(ValueNode):
    pass


class FuzzySearch(BooleanValueOp):

    def __init__(self, arg, pattern):
        self.arg = arg
        self.pattern = as_value_expr(pattern)

        if not isinstance(self.pattern, ir.StringScalar):
            raise TypeError(self.pattern)

        ValueNode.__init__(self, [self.arg, self.pattern])

    output_type = _bool_output


class StringSQLLike(FuzzySearch):
    pass


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueNode):

    def __init__(self, arg, pattern, index):
        self.arg = arg
        self.pattern = as_value_expr(pattern)
        self.index = index

        if not isinstance(self.pattern, ir.StringScalar):
            raise TypeError(self.pattern)

        ValueNode.__init__(self, [self.arg, self.pattern, self.index])

    output_type = _string_output


class RegexReplace(ValueNode):

    def __init__(self, arg, pattern, replacement):
        self.arg = as_value_expr(arg)
        self.pattern = as_value_expr(pattern)
        self.replacement = as_value_expr(replacement)

        if not isinstance(self.pattern, ir.StringScalar):
            raise TypeError(self.pattern)

        ValueNode.__init__(self, [self.arg, self.pattern, self.replacement])

    output_type = _string_output


class StringLength(UnaryOp):

    output_type = _int_output


class StringAscii(UnaryOp):

    output_type = _int_output


class BinaryOp(ValueNode):

    """
    A binary operation

    """
    # Casting rules for type promotions (for resolving the output type) may
    # depend in some cases on the target backend.
    #
    # TODO: how will overflows be handled? Can we provide anything useful in
    # Ibis to help the user avoid them?

    def __init__(self, left, right):
        left, right = self._maybe_cast_args(left, right)
        self.left = left
        self.right = right
        ValueNode.__init__(self, [self.left, self.right])

    def _maybe_cast_args(self, left, right):
        return left, right

    def root_tables(self):
        return ir.distinct_roots(self.left, self.right)

    def output_type(self):
        raise NotImplementedError


# ----------------------------------------------------------------------


class Count(ir.Reduction):
    # TODO: count(col) takes down Impala, must always do count(*) in generated
    # SQL

    def __init__(self, expr, where=None):
        # TODO: counts are actually table-level operations. Let's address
        # during the SQL generation exercise
        if not is_collection(expr):
            raise TypeError

        ir.Reduction.__init__(self, expr, where)

    def output_type(self):
        return ir.Int64Scalar


class Sum(ir.Reduction):

    def output_type(self):
        _ = ir
        if isinstance(self.arg, (_.IntegerValue, _.BooleanValue)):
            return _.Int64Scalar
        elif isinstance(self.arg, _.FloatingValue):
            return _.DoubleScalar
        elif isinstance(self.arg, _.DecimalValue):
            return _decimal_scalar_ctor(self.arg._precision, 38)
        else:
            raise TypeError(self.arg)


class Mean(ir.Reduction):

    def output_type(self):
        _ = ir
        if isinstance(self.arg, _.DecimalValue):
            return _decimal_scalar_ctor(self.arg._precision, 38)
        elif isinstance(self.arg, _.NumericValue):
            return _.DoubleScalar
        else:
            raise NotImplementedError


def _decimal_scalar_ctor(precision, scale):
    _ = ir
    out_type = _.DecimalType(precision, scale)
    return _.DecimalScalar._make_constructor(out_type)


class StdDeviation(ir.Reduction):
    pass


def _min_max_output_rule(self):
    _ = ir
    if isinstance(self.arg, _.DecimalValue):
        return _decimal_scalar_ctor(self.arg._precision, 38)
    else:
        return _.scalar_type(self.arg.type())


class Max(ir.Reduction):

    output_type = _min_max_output_rule


class Min(ir.Reduction):

    output_type = _min_max_output_rule


class HLLCardinality(ir.Reduction):

    """
    Approximate number of unique values using HyperLogLog algorithm. Impala
    offers the NDV built-in function for this.
    """

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        return ir.DoubleScalar


class GroupConcat(ir.Reduction):

    def __init__(self, arg, sep=','):
        self._ensure_array(arg)
        self.arg = arg
        self.sep = as_value_expr(sep)
        ValueNode.__init__(self, [self.arg, self.sep])

    def root_tables(self):
        return self.arg._root_tables()

    def output_type(self):
        return ir.StringScalar


class CMSMedian(ir.Reduction):

    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    def output_type(self):
        # Scalar but type of caller
        return ir.scalar_type(self.arg.type())

# ----------------------------------------------------------------------
# Distinct stuff


class Distinct(ir.BlockingTableNode, ir.HasSchema):

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

        ir.BlockingTableNode.__init__(self, [table])
        schema = self.table.schema()
        HasSchema.__init__(self, schema)


class DistinctArray(ArrayNode):

    """
    COUNT(DISTINCT ...) is really just syntactic suger, but we provide a
    distinct().count() nicety for users nonetheless.

    For all intents and purposes, like Distinct, but can be distinguished later
    for evaluation if the result should be array-like versus table-like. Also
    for calling count()
    """

    def __init__(self, arg):
        self.arg = arg
        self.name = arg.get_name()
        self.table = arg.to_projection().distinct()
        ArrayNode.__init__(self, arg)

    def output_type(self):
        return type(self.arg)

    def root_tables(self):
        return [self.table]

    def count(self):
        """
        Only valid if the distinct contains a single column
        """
        return CountDistinct(self.arg)


class CountDistinct(ir.Reduction):

    def output_type(self):
        return ir.Int64Scalar


# ---------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueNode):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)

    def __init__(self, expr):
        if not isinstance(expr, ir.BooleanArray):
            raise ValueError('Expression must be a boolean array')

        self.arg = expr
        ValueNode.__init__(self, [expr])

    def output_type(self):
        _ = ir
        roots = self.arg._root_tables()
        if len(roots) > 1:
            return _.BooleanArray
        else:
            # A reduction
            return _.BooleanScalar

    def negate(self):
        return NotAny(self.arg)


class NotAny(Any):

    def negate(self):
        return Any(self.arg)


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
            default = null()
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
            default = null()
        else:
            default = self.default

        op = SearchedCase(self.cases, self.results, default)
        return op.to_expr()


class SimpleCase(ValueNode):

    def __init__(self, base_expr, case_exprs, result_exprs,
                 default_expr):
        assert len(case_exprs) == len(result_exprs)

        self.base = base_expr
        self.cases = case_exprs
        self.results = result_exprs
        self.default = default_expr
        Node.__init__(self, [self.base, self.cases, self.results,
                             self.default])

    def root_tables(self):
        all_exprs = [self.base] + self.cases + self.results
        if self.default is not None:
            all_exprs.append(self.default)
        return ir.distinct_roots(*all_exprs)

    def output_type(self):
        out_exprs = self.results + [self.default]
        typename = rules.highest_precedence_type(out_exprs)
        return rules.shape_like(self.base, typename)


class SearchedCase(ValueNode):

    def __init__(self, case_exprs, result_exprs, default_expr):
        assert len(case_exprs) == len(result_exprs)

        self.cases = case_exprs
        self.results = result_exprs
        self.default = default_expr
        ValueNode.__init__(self, [self.cases, self.results, self.default])

    def root_tables(self):
        all_exprs = self.cases + self.results
        if self.default is not None:
            all_exprs.append(self.default)
        return ir.distinct_roots(*all_exprs)

    def output_type(self):
        out_exprs = self.results + [self.default]
        typename = rules.highest_precedence_type(out_exprs)
        return rules.shape_like_args(self.cases, typename)


class Where(ValueNode):

    """
    Ternary case expression, equivalent to

    bool_expr.case()
             .when(True, true_expr)
             .else_(false_or_null_expr)
    """

    def __init__(self, bool_expr, true_expr, false_null_expr):
        self.bool_expr = as_value_expr(bool_expr)
        self.true_expr = as_value_expr(true_expr)
        self.false_null_expr = as_value_expr(false_null_expr)

        ValueNode.__init__(self, [self.bool_expr, self.true_expr,
                                  self.false_null_expr])

    def output_type(self):
        return rules.shape_like(self.bool_expr, self.true_expr.type())


class Join(TableNode):

    def __init__(self, left, right, join_predicates):
        from ibis.expr.analysis import ExprValidator

        if not is_table(left):
            raise TypeError('Can only join table expressions, got %s for '
                            'left table' % type(left))

        if not is_table(right):
            raise TypeError('Can only join table expressions, got %s for '
                            'right table' % type(left))

        if left.equals(right):
            right = right.view()

        self.left = left
        self.right = right
        self.predicates = self._clean_predicates(join_predicates)

        # Validate join predicates. Each predicate must be valid jointly when
        # considering the roots of each input table
        validator = ExprValidator([self.left, self.right])
        validator.validate_all(self.predicates)

        Node.__init__(self, [self.left, self.right, self.predicates])

    def _clean_predicates(self, predicates):
        import ibis.expr.analysis as L

        result = []

        if not isinstance(predicates, (list, tuple)):
            predicates = [predicates]

        for pred in predicates:
            if isinstance(pred, tuple):
                if len(pred) != 2:
                    raise ExpressionError('Join key tuple must be length 2')
                lk, rk = pred
                lk = self.left._ensure_expr(lk)
                rk = self.right._ensure_expr(rk)
                pred = lk == rk
            else:
                pred = L.substitute_parents(pred, past_projection=False)

            if not isinstance(pred, ir.BooleanArray):
                raise ExpressionError('Join predicate must be comparison')

            preds = L.unwrap_ands(pred)
            result.extend(preds)

        return result

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
            raise RelationError('Joined tables have overlapping names: %s'
                                % str(list(overlap)))

        return sleft.append(sright)

    def has_schema(self):
        return False

    def root_tables(self):
        if util.all_of([self.left.op(), self.right.op()],
                       (Join, Projection)):
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


class LeftSemiJoin(Join):

    """

    """

    def _get_schema(self):
        return self.left.schema()


class LeftAntiJoin(Join):

    """

    """

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


class CrossJoin(InnerJoin):

    """
    Some databases have a CROSS JOIN operator, that may be preferential to use
    over an INNER JOIN with no predicates.
    """

    def __init__(self, left, right, predicates=[]):
        InnerJoin.__init__(self, left, right, [])


class Union(ir.BlockingTableNode, HasSchema):

    def __init__(self, left, right, distinct=False):
        self.left = left
        self.right = right
        self.distinct = distinct

        TableNode.__init__(self, [left, right, distinct])
        self._validate()
        HasSchema.__init__(self, self.left.schema())

    def _validate(self):
        if not self.left.schema().equals(self.right.schema()):
            raise RelationError('Table schemas must be equal to form union')


class Filter(TableNode):

    def __init__(self, table_expr, predicates):
        self.table = table_expr
        self.predicates = predicates

        TableNode.__init__(self, [table_expr, predicates])
        self._validate()

    def _validate(self):
        from ibis.expr.analysis import FilterValidator
        validator = FilterValidator([self.table])
        validator.validate_all(self.predicates)

    def get_schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        return self.table._root_tables()


class Limit(ir.BlockingTableNode):

    _arg_names = [None, 'n', 'offset']

    def __init__(self, table, n, offset=0):
        self.table = table
        self.n = n
        self.offset = offset
        TableNode.__init__(self, [table, n, offset])

    def get_schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        return [self]


# --------------------------------------------------------------------
# Sorting


class SortBy(TableNode):

    # Q: Will SortBy always require a materialized schema?

    def __init__(self, table_expr, sort_keys):
        self.table = table_expr
        self.keys = [_to_sort_key(self.table, k)
                     for k in util.promote_list(sort_keys)]

        TableNode.__init__(self, [self.table, self.keys])

    def get_schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        tables = self.table._root_tables()
        return tables


def _to_sort_key(table, key):
    if isinstance(key, DeferredSortKey):
        key = key.resolve(table)

    if isinstance(key, SortKey):
        return key

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, ir.Expr):
        key = table._ensure_expr(key)

    if isinstance(sort_order, py_string):
        if sort_order.lower() in ('desc', 'descending'):
            sort_order = False
        elif not isinstance(sort_order, bool):
            sort_order = bool(sort_order)

    return SortKey(key, ascending=sort_order)


class SortKey(object):

    def __init__(self, expr, ascending=True):
        if not is_array(expr):
            raise ExpressionError('Must be an array/column expression')

        self.expr = expr
        self.ascending = ascending

    def __repr__(self):
        # Temporary
        rows = ['Sort key:',
                '  ascending: {0!s}'.format(self.ascending),
                util.indent(_safe_repr(self.expr), 2)]
        return '\n'.join(rows)

    def equals(self, other):
        return (isinstance(other, SortKey) and self.expr.equals(other.expr)
                and self.ascending == other.ascending)


class DeferredSortKey(object):

    def __init__(self, what, ascending=True):
        self.what = what
        self.ascending = ascending

    def resolve(self, parent):
        what = self.what
        if not isinstance(what, ir.Expr):
            what = parent.get_column(what)
        return SortKey(what, ascending=self.ascending)


def desc(expr):
    """
    Create a sort key (when used in sort_by) by the passed array expression or
    column name.

    Parameters
    ----------
    expr : array expression or string
      Can be a column name in the table being sorted

    Examples
    --------
    result = (self.table.group_by('g')
              .size('count')
              .sort_by(ibis.desc('count')))
    """
    return DeferredSortKey(expr, ascending=False)


class SelfReference(ir.BlockingTableNode, HasSchema):

    def __init__(self, table_expr):
        self.table = table_expr
        TableNode.__init__(self, [table_expr])
        HasSchema.__init__(self, table_expr.schema(),
                           name=table_expr.op().name)

    def root_tables(self):
        # The dependencies of this operation are not walked, which makes the
        # table expression holding this relationally distinct from other
        # expressions, so things like self-joins are possible
        return [self]


class Projection(ir.BlockingTableNode, HasSchema):

    def __init__(self, table_expr, proj_exprs):
        from ibis.expr.analysis import ExprValidator

        # Need to validate that the column expressions are compatible with the
        # input table; this means they must either be scalar expressions or
        # array expressions originating from the same root table expression
        validator = ExprValidator([table_expr])

        # Resolve schema and initialize
        types = []
        names = []
        clean_exprs = []
        for expr in proj_exprs:
            if isinstance(expr, py_string):
                expr = table_expr[expr]

            validator.assert_valid(expr)
            if isinstance(expr, ValueExpr):
                name = expr.get_name()
                names.append(name)
                types.append(expr.type())
            elif is_table(expr):
                schema = expr.schema()
                names.extend(schema.names)
                types.extend(schema.types)
            else:
                raise NotImplementedError

            clean_exprs.append(expr)

        # validate uniqueness
        schema = ir.Schema(names, types)

        HasSchema.__init__(self, schema)
        Node.__init__(self, [table_expr] + [clean_exprs])

        self.table = table_expr
        self.selections = clean_exprs

    def substitute_table(self, table_expr):
        return Projection(table_expr, self.selections)

    def root_tables(self):
        return [self]

    def is_ancestor(self, other):
        if isinstance(other, ir.Expr):
            other = other.op()

        if self.equals(other):
            return True

        table = self.table
        exist_layers = False
        while not isinstance(table.op(), (ir.BlockingTableNode, Join)):
            table = table.op().table
            exist_layers = True

        if exist_layers:
            reboxed = Projection(table, self.selections)
            return reboxed.is_ancestor(other)
        else:
            return False


class Aggregation(ir.BlockingTableNode, HasSchema):

    """
    agg_exprs : per-group scalar aggregates
    by : group expressions
    having : post-aggregation predicate

    TODO: not putting this in the aggregate operation yet
    where : pre-aggregation predicate
    """

    _arg_names = ['table', 'metrics', 'by', 'having']

    def __init__(self, table, agg_exprs, by=None, having=None):
        # For tables, like joins, that are not materialized
        self.table = table

        self.agg_exprs = self._rewrite_exprs(agg_exprs)

        by = by or []
        self.by = self.table._resolve(by)
        self.by = self._rewrite_exprs(self.by)

        self.having = having or []
        self.having = self._rewrite_exprs(self.having)
        self._validate()

        TableNode.__init__(self, [table, self.agg_exprs, self.by, self.having])

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
                all_exprs.append(expr)

        return [substitute_parents(x, past_projection=False)
                for x in all_exprs]

    def substitute_table(self, table_expr):
        return Aggregation(table_expr, self.agg_exprs, by=self.by,
                           having=self.having)

    def _validate(self):
        # All aggregates are valid
        for expr in self.agg_exprs:
            if not is_scalar(expr) or not expr.is_reduction():
                raise TypeError('Passed a non-aggregate expression: %s' %
                                _safe_repr(expr))

        for expr in self.having:
            if not isinstance(expr, ir.BooleanScalar):
                raise ExpressionError('Having clause must be boolean '
                                      'expression, was: {0!s}'
                                      .format(_safe_repr(expr)))

        # All non-scalar refs originate from the input table
        all_exprs = self.agg_exprs + self.by + self.having
        self.table._assert_valid(all_exprs)

    def _result_schema(self):
        names = []
        types = []

        # All exprs must be named

        for e in self.by + self.agg_exprs:
            names.append(e.get_name())
            types.append(e.type())

        return ir.Schema(names, types)


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

    def output_type(self):
        if not util.all_of(self.args, ir.NumericValue):
            raise TypeError('One argument was non-numeric')

        return rules.shape_like_args(self.args, 'double')


class LogicalBinaryOp(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, ir.BooleanValue):
            raise TypeError('Only valid with boolean data')
        return rules.shape_like_args(self.args, 'boolean')


class Modulus(BinaryOp):

    def output_type(self):
        helper = rules.BinaryPromoter(self.left, self.right, operator.add)
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
            return right, right._implicit_cast(left)

        return left, right

    def output_type(self):
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


class Between(BooleanValueOp):

    def __init__(self, expr, lower_bound, upper_bound):
        self.expr = expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        BooleanValueOp.__init__(self, [expr, lower_bound, upper_bound])

    def root_tables(self):
        return ir.distinct_roots(*self.args)

    def output_type(self):
        self._assert_can_compare()
        return rules.shape_like_args(self.args, 'boolean')

    def _assert_can_compare(self):
        if (not self.expr._can_compare(self.lower_bound) or
                not self.expr._can_compare(self.upper_bound)):
            raise TypeError('Arguments are not comparable')


class Contains(BooleanValueOp):

    def __init__(self, value, options):
        self.value = as_value_expr(value)
        self.options = as_value_expr(options)
        BooleanValueOp.__init__(self, [self.value, self.options])

    def root_tables(self):
        exprs = [self.value, self.options]
        return ir.distinct_roots(*exprs)

    def output_type(self):
        all_args = [self.value]

        options = self.options.op()
        if isinstance(options, ValueList):
            all_args += options.values
        elif isinstance(self.options, ArrayExpr):
            all_args += [self.options]
        else:
            raise TypeError(type(options))

        return rules.shape_like_args(all_args, 'boolean')


class NotContains(Contains):

    def __init__(self, value, options):
        Contains.__init__(self, value, options)


class ReplaceValues(ArrayNode):

    """
    Apply a multi-value replacement on a particular column. As an example from
    SQL, given DAYOFWEEK(timestamp_col), replace 1 through 5 to "WEEKDAY" and 6
    and 7 to "WEEKEND"
    """
    pass


class TopK(ArrayNode):

    def __init__(self, arg, k, by=None):
        if by is None:
            by = arg.count()

        if not isinstance(arg, ArrayExpr):
            raise TypeError(arg)

        if not isinstance(k, int) or k < 0:
            raise ValueError('k must be positive integer, was: {0}'.format(k))

        self.arg = arg
        self.k = k
        self.by = by

        Node.__init__(self, [arg, k, by])

    def root_tables(self):
        return self.arg._root_tables()

    def to_expr(self):
        return ir.BooleanArray(self)


class Constant(ValueNode):

    def __init__(self):
        ValueNode.__init__(self, [])

    def root_tables(self):
        return []


class TimestampNow(Constant):

    def output_type(self):
        return ir.TimestampScalar


class E(Constant):

    def output_type(self):
        return ir.DoubleScalar


class ExtractTimestampField(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, ir.TimestampValue):
            raise AssertionError
        return rules.shape_like(self.arg, 'int32')


class ExtractYear(ExtractTimestampField):
    pass


class ExtractMonth(ExtractTimestampField):
    pass


class ExtractDay(ExtractTimestampField):
    pass


class ExtractHour(ExtractTimestampField):
    pass


class ExtractMinute(ExtractTimestampField):
    pass


class ExtractSecond(ExtractTimestampField):
    pass


class ExtractMillisecond(ExtractTimestampField):
    pass


class TimestampFromUNIX(ValueNode):

    def __init__(self, arg, unit='s'):
        self.arg = as_value_expr(arg)
        self.unit = unit

        if self.unit not in set(['s', 'ms', 'us']):
            raise ValueError(self.unit)

        ValueNode.__init__(self, [self.arg, self.unit])

    def output_type(self):
        return rules.shape_like(self.arg, 'timestamp')


class DecimalPrecision(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, ir.DecimalValue):
            raise AssertionError
        return rules.shape_like(self.arg, 'int32')


class DecimalScale(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, ir.DecimalValue):
            raise AssertionError
        return rules.shape_like(self.arg, 'int32')


class Hash(ValueNode):

    def __init__(self, arg, how):
        self.arg = as_value_expr(arg)
        self.how = how
        ValueNode.__init__(self, [self.arg, self.how])

    def output_type(self):
        return rules.shape_like(self.arg, 'int64')


class TimestampDelta(ValueNode):

    def __init__(self, arg, offset):
        from ibis.expr.temporal import Timedelta

        self.arg = as_value_expr(arg)
        self.offset = offset

        if not isinstance(self.arg, ir.TimestampValue):
            raise TypeError('Must interact with a timestamp expression')

        if not isinstance(offset, Timedelta):
            raise TypeError(offset)

        ValueNode.__init__(self, [self.arg, self.offset])

    def output_type(self):
        return rules.shape_like(self.arg, 'timestamp')
