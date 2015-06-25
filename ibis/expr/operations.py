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
from ibis.compat import py_string
from ibis.expr.rules import value, string, number, integer, boolean, list_of
from ibis.expr.types import (Node, as_value_expr,
                             ValueExpr, ArrayExpr, TableExpr,
                             ArrayNode, TableNode, ValueNode,
                             HasSchema, _safe_repr)
import ibis.expr.rules as rules
import ibis.expr.types as ir
import ibis.util as util


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

    input_type = [value]


class Cast(ValueNode):

    # input_type = [value, rules.data_type]

    def __init__(self, arg, target_type):
        self.arg = as_value_expr(arg)
        self.target_type = ir._validate_type(target_type.lower())
        ValueNode.__init__(self, arg, self.target_type)

    def resolve_name(self):
        return self.arg.get_name()

    def output_type(self):
        # TODO: error handling for invalid casts
        return rules.shape_like(self.arg, self.target_type)


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

    output_type = rules.numeric_highest_promote(0)


class IfNull(ValueNode):

    """
    Equivalent to (but perhaps implemented differently):

    case().when(expr.notnull(), expr)
          .else_(null_substitute_expr)
    """

    input_type = [value, value(name='ifnull_expr')]
    output_type = rules.type_of_arg(0)


class NullIf(ValueNode):

    """
    Set values to NULL if they equal the null_if_expr
    """

    input_type = [value, value(name='null_if_expr')]
    output_type = rules.type_of_arg(0)


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
        ValueNode.__init__(self, *self.values)

    # According to Impala documentation:
    # Return type: same as the initial argument value, except that integer
    # values are promoted to BIGINT and floating-point values are promoted to
    # DOUBLE; use CAST() when inserting into a smaller numeric column

    output_type = _coalesce_upcast


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
        return rules.shape_like(arg, 'int32')


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


class Round(ValueNode):

    input_type = [value, integer(name='digits', optional=True)]

    def output_type(self):
        arg, digits = self.args
        if isinstance(arg, ir.DecimalValue):
            return arg._factory
        elif digits is None:
            return rules.shape_like(arg, 'int64')
        else:
            return rules.shape_like(arg, 'double')


class RealUnaryOp(UnaryOp):

    input_type = [number]
    output_type = rules.shape_like_arg(0, 'double')


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class Sqrt(RealUnaryOp):
    pass


class Logarithm(RealUnaryOp):

    # superclass

    input_type = [number(allow_boolean=False)]


class Log(Logarithm):

    input_type = (Logarithm.input_type +
                  [number(name='base', optional=True)])


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

    input_type = [string, integer(name='start'),
                  integer(name='length', optional=True)]
    output_type = rules.shape_like_arg(0, 'string')


class StrRight(ValueNode):

    input_type = [string, integer(name='nchars')]
    output_type = rules.shape_like_arg(0, 'string')


class Repeat(ValueNode):

    input_type = [string, integer(name='times')]
    output_type = rules.shape_like_arg(0, 'string')


class StringFind(ValueNode):

    input_type = [string, string(name='substr')]
    output_type = rules.shape_like_arg(0, 'int32')


class Translate(ValueNode):

    input_type = [string, string(name='from_str'), string(name='to_str')]
    output_type = rules.shape_like_arg(0, 'string')


class Locate(ValueNode):

    input_type = [string, string(name='substr'),
                  integer(name='pos', default=0)]
    output_type = rules.shape_like_arg(0, 'int32')


class LPad(ValueNode):

    input_type = [string, integer(name='length'), string(name='pad')]
    output_type = rules.shape_like_arg(0, 'string')


class RPad(ValueNode):

    input_type = [string, integer(name='length'), string(name='pad')]
    output_type = rules.shape_like_arg(0, 'string')


class FindInSet(ValueNode):

    input_type = [string(name='needle'), list_of(string, min_length=1)]
    output_type = rules.shape_like_arg(0, 'int32')


class StringJoin(ValueNode):

    input_type = [string(name='sep'), list_of(string, min_length=1)]
    output_type = rules.shape_like_arg(0, 'string')


class BooleanValueOp(ValueNode):
    pass


class FuzzySearch(BooleanValueOp):

    input_type = [string, string(name='pattern')]
    output_type = rules.shape_like_arg(0, 'boolean')


class StringSQLLike(FuzzySearch):
    pass


class RegexSearch(FuzzySearch):
    pass


class RegexExtract(ValueNode):

    input_type = [string, string(name='pattern'), integer(name='index')]
    output_type = rules.shape_like_arg(0, 'string')


class RegexReplace(ValueNode):

    input_type = [string, string(name='pattern'),
                  string(name='replacement')]
    output_type = rules.shape_like_arg(0, 'string')


class StringLength(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class StringAscii(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


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
        ValueNode.__init__(self, self.left, self.right)

    def _maybe_cast_args(self, left, right):
        return left, right

    def output_type(self):
        raise NotImplementedError


# ----------------------------------------------------------------------


class Reduction(ValueNode):

    input_type = [rules.array, boolean(name='where', optional=True)]


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
        if isinstance(op, Reduction):
            return True

        for arg in op.args:
            if isinstance(arg, ir.ScalarExpr) and has_reduction(arg.op()):
                return True

        return False

    return has_reduction(expr.op())


class Count(Reduction):
    # TODO: count(col) takes down Impala, must always do count(*) in generated
    # SQL

    input_type = [rules.collection, boolean(name='where', optional=True)]

    # TODO: counts are actually table-level operations. Let's address
    # during the SQL generation exercise

    def output_type(self):
        return ir.Int64Scalar


class Sum(Reduction):

    def output_type(self):
        arg = self.args[0]
        if isinstance(arg, (ir.IntegerValue, ir.BooleanValue)):
            return ir.Int64Scalar
        elif isinstance(arg, ir.FloatingValue):
            return ir.DoubleScalar
        elif isinstance(arg, ir.DecimalValue):
            return _decimal_scalar_ctor(arg._precision, 38)
        else:
            raise TypeError(arg)


class Mean(Reduction):

    def output_type(self):
        arg = self.args[0]
        if isinstance(arg, ir.DecimalValue):
            return _decimal_scalar_ctor(arg._precision, 38)
        elif isinstance(arg, ir.NumericValue):
            return ir.DoubleScalar
        else:
            raise NotImplementedError


def _decimal_scalar_ctor(precision, scale):
    out_type = ir.DecimalType(precision, scale)
    return ir.DecimalScalar._make_constructor(out_type)


class StdDeviation(Reduction):
    pass


def _min_max_output_rule(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        return _decimal_scalar_ctor(arg._precision, 38)
    else:
        return ir.scalar_type(arg.type())


class Max(Reduction):

    output_type = _min_max_output_rule


class Min(Reduction):

    output_type = _min_max_output_rule


class HLLCardinality(Reduction):

    """
    Approximate number of unique values using HyperLogLog algorithm. Impala
    offers the NDV built-in function for this.
    """

    def output_type(self):
        # Impala 2.0 and higher returns a DOUBLE
        return ir.DoubleScalar


class GroupConcat(Reduction):

    input_type = [rules.array, string(name='sep', default=','),
                  boolean(name='where', optional=True)]

    def output_type(self):
        return ir.StringScalar


class CMSMedian(Reduction):

    """
    Compute the approximate median of a set of comparable values using the
    Count-Min-Sketch algorithm. Exposed in Impala using APPX_MEDIAN.
    """

    def output_type(self):
        # Scalar but type of caller
        return ir.scalar_type(self.args[0].type())

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


class CountDistinct(Reduction):

    input_type = [rules.array]

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
        ValueNode.__init__(self, self.arg)

    def output_type(self):
        roots = self.arg._root_tables()
        if len(roots) > 1:
            return ir.BooleanArray
        else:
            # A reduction
            return ir.BooleanScalar

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


def _arg_getter(i):
    @property
    def arg_accessor(self):
        return self.args[i]
    return arg_accessor


class SimpleCase(ValueNode):

    input_type = [value(name='base'),
                  list_of(value, name='cases'),
                  list_of(value, name='results'),
                  value(name='default')]

    def __init__(self, base, cases, results, default):
        assert len(cases) == len(results)
        ValueNode.__init__(self, base, cases, results, default)

    base = _arg_getter(0)
    cases = _arg_getter(1)
    results = _arg_getter(2)
    default = _arg_getter(3)

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


class SearchedCase(ValueNode):

    input_type = [list_of(boolean, name='cases'),
                  list_of(value, name='results'),
                  value(name='default')]

    def __init__(self, cases, results, default):
        assert len(cases) == len(results)
        ValueNode.__init__(self, cases, results, default)

    cases = _arg_getter(0)
    results = _arg_getter(1)
    default = _arg_getter(2)

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


class Where(ValueNode):

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


class Join(TableNode):

    def __init__(self, left, right, join_predicates):
        from ibis.expr.analysis import ExprValidator

        if not rules.is_table(left):
            raise TypeError('Can only join table expressions, got %s for '
                            'left table' % type(left))

        if not rules.is_table(right):
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
        if not rules.is_array(expr):
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
            elif rules.is_table(expr):
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
            if not rules.is_scalar(expr) or not is_reduction(expr):
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
        BooleanValueOp.__init__(self, expr, lower_bound, upper_bound)

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
        BooleanValueOp.__init__(self, self.value, self.options)

    def output_type(self):
        all_args = [self.value]

        options = self.options.op()
        if isinstance(options, ir.ValueList):
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


class TimestampUnaryOp(UnaryOp):

    input_type = [rules.value_typed_as(ir.TimestampValue)]


class ExtractTimestampField(TimestampUnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


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

    input_type = [value, rules.string_options(['s', 'ms', 'us'], name='unit')]
    output_type = rules.shape_like_arg(0, 'timestamp')


class DecimalUnaryOp(UnaryOp):

    input_type = [rules.value_typed_as(ir.DecimalValue)]


class DecimalPrecision(DecimalUnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class DecimalScale(UnaryOp):

    output_type = rules.shape_like_arg(0, 'int32')


class Hash(ValueNode):

    input_type = [value, rules.string_options(['fnv'], name='how')]
    output_type = rules.shape_like_arg(0, 'int64')


class TimestampDelta(ValueNode):

    output_type = rules.shape_like_arg(0, 'timestamp')

    def __init__(self, arg, offset):
        from ibis.expr.temporal import Timedelta

        self.arg = as_value_expr(arg)
        self.offset = offset

        if not isinstance(self.arg, ir.TimestampValue):
            raise TypeError('Must interact with a timestamp expression')

        if not isinstance(offset, Timedelta):
            raise TypeError(offset)

        ValueNode.__init__(self, self.arg, self.offset)
