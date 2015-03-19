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
import ibis.expr.types as ir

import ibis.util as util


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
            val = value_list(val)
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
    if value is None:
        return null()
    else:
        return Literal(value).to_expr()


_NULL = None


def null():
    global _NULL
    if _NULL is None:
        _NULL = ir.NullScalar(NullLiteral())

    return _NULL


def desc(expr):
    return SortKey(expr, ascending=False)


def value_list(values):
    return ValueList(values).to_expr()


class Literal(ValueNode):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return 'Literal(%s)' % repr(self.value)

    @property
    def args(self):
        return [self.value]

    def equals(self, other):
        if not isinstance(other, Literal):
            return False
        return (type(self.value) == type(other.value)
                and self.value == other.value)

    def output_type(self):
        import ibis.expr.rules as rules
        if isinstance(self.value, bool):
            klass = ir.BooleanScalar
        elif isinstance(self.value, (int, long)):
            int_type = rules.int_literal_class(self.value)
            klass = ir.scalar_type(int_type)
        elif isinstance(self.value, float):
            klass = ir.DoubleScalar
        elif isinstance(self.value, basestring):
            klass = ir.StringScalar

        return klass

    def root_tables(self):
        return []


class NullLiteral(ValueNode):

    """
    Typeless NULL literal
    """

    def __init__(self):
        return

    @property
    def args(self):
        return [None]

    def equals(other):
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
            raise KeyError("'{}' is not a field".format(name))

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

    def resolve_name(self):
        return self.arg.get_name()


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
        return _shape_like(self.arg, self.target_type)


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
        return _shape_like(self.arg, 'boolean')


class NotNull(UnaryOp):

    """
    Returns true if values are not null

    Returns
    -------
    notnull : boolean with dimension of caller
    """

    def output_type(self):
        return _shape_like(self.arg, 'boolean')


def _shape_like(arg, out_type):
    if isinstance(arg, ir.ScalarExpr):
        return ir.scalar_type(out_type)
    else:
        return ir.array_type(out_type)


def _shape_like_args(args, out_type):
    if util.any_of(args, ArrayExpr):
        return ir.array_type(out_type)
    else:
        return ir.scalar_type(out_type)


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
        return _shape_like(self.arg, 'int32')


class Ceil(UnaryOp):

    output_type = _ceil_floor_output


class Floor(UnaryOp):

    output_type = _ceil_floor_output


class Round(ValueNode):

    def __init__(self, arg, digits=None):
        self.arg = arg
        self.digits = validate_int(digits)
        ValueNode.__init__(self, [arg, digits])

    def output_type(self):
        validate_numeric(self.arg)
        if isinstance(self.arg, ir.DecimalValue):
            return self.arg._factory
        elif self.digits is None or self.digits == 0:
            return _shape_like(self.arg, 'int64')
        else:
            return _shape_like(self.arg, 'double')


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

        return _shape_like(self.arg, 'double')


class Exp(RealUnaryOp):
    pass


class Sign(UnaryOp):

    def output_type(self):
        return _shape_like(self.arg, 'int32')


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
    return _shape_like(self.arg, 'string')


class StringUnaryOp(UnaryOp):

    output_type = _string_output


class Uppercase(StringUnaryOp):
    pass


class Lowercase(StringUnaryOp):
    pass


class Substring(ValueNode):

    def __init__(self, arg, start, length=None):
        self.arg = arg
        self.start = start
        self.length = length
        ValueNode.__init__(self, [arg, start, length])

    output_type = _string_output


class StrRight(ValueNode):

    def __init__(self, arg, nchars):
        self.arg = arg
        self.nchars = nchars
        ValueNode.__init__(self, [arg, nchars])

    output_type = _string_output


class StringLength(UnaryOp):

    def output_type(self):
        return _shape_like(self.arg, 'int32')


class BinaryOp(ValueNode):

    """
    A binary operation

    """
    # Casting rules for type promotions (for resolving the output type) may
    # depend in some cases on the target backend.
    #
    # TODO: how will overflows be handled? Can we provide anything useful in
    # Ibis to help the user avoid them?

    def __init__(self, left_expr, right_expr):
        self.left = left_expr
        self.right = right_expr
        ValueNode.__init__(self, [left_expr, right_expr])

    def root_tables(self):
        return ir.distinct_roots(self.left, self.right)

    def output_type(self):
        raise NotImplementedError


#----------------------------------------------------------------------


class Count(ir.Reduction):
    # TODO: count(col) takes down Impala, must always do count(*) in generated
    # SQL

    def __init__(self, expr):
        # TODO: counts are actually table-level operations. Let's address
        # during the SQL generation exercise
        if not is_collection(expr):
            raise TypeError
        self.arg = expr
        ValueNode.__init__(self, [expr])

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
            return _decimal_scalar_ctor(self.arg.precision, 38)
        else:
            raise TypeError(self.arg)


class Mean(ir.Reduction):

    def output_type(self):
        _ = ir
        if isinstance(self.arg, _.DecimalValue):
            return _decimal_scalar_ctor(self.arg.precision, 38)
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
        return _decimal_scalar_ctor(self.arg.precision, 38)
    else:
        return _.scalar_type(self.arg.type())


class Max(ir.Reduction):

    output_type = _min_max_output_rule


class Min(ir.Reduction):

    output_type = _min_max_output_rule


#----------------------------------------------------------------------
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


#----------------------------------------------------------------------
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

#----------------------------------------------------------------------


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
        from ibis.expr.rules import highest_precedence_type
        out_exprs = self.results + [self.default]
        typename = highest_precedence_type(out_exprs)
        return _shape_like(self.base, typename)


class SearchedCase(ValueNode):

    def __init__(self, case_exprs, result_exprs, default_expr):
        assert len(case_exprs) == len(result_exprs)

        self.cases = case_exprs
        self.results = result_exprs
        self.default = default_expr
        Node.__init__(self, [self.cases, self.results, self.default])

    def root_tables(self):
        all_exprs = self.cases + self.results
        if self.default is not None:
            all_exprs.append(self.default)
        return ir.distinct_roots(*all_exprs)

    def output_type(self):
        from ibis.expr.rules import highest_precedence_type
        out_exprs = self.results + [self.default]
        typename = highest_precedence_type(out_exprs)
        return _shape_like_args(self.cases, typename)


class Join(TableNode):

    def __init__(self, left, right, join_predicates):
        from ibis.expr.analysis import ExprValidator

        if not is_table(left):
            raise TypeError('Can only join table expressions, got %s for '
                            'left table' % type(left))

        if not is_table(right):
            raise TypeError('Can only join table expressions, got %s for '
                            'right table' % type(left))

        self.left = left
        self.right = right
        self.predicates = self._clean_predicates(join_predicates)

        # Validate join predicates. Each predicate must be valid jointly when
        # considering the roots of each input table
        validator = ExprValidator([left, right])
        validator.validate_all(self.predicates)

        Node.__init__(self, [left, right, self.predicates])

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
                pred = L.substitute_parents(pred)

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
        tables = self.table._root_tables()
        return tables


class Limit(ir.BlockingTableNode):

    def __init__(self, table, n, offset):
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

#----------------------------------------------------------------------
# Sorting


class SortBy(TableNode):

    # Q: Will SortBy always require a materialized schema?

    def __init__(self, table_expr, sort_keys):
        self.table = table_expr
        self.keys = [_to_sort_key(self.table, k)
                     for k in _promote_list(sort_keys)]

        TableNode.__init__(self, [self.table, self.keys])

    def get_schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        tables = self.table._root_tables()
        return tables


def _to_sort_key(table, key):
    if isinstance(key, SortKey):
        return key

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, ir.Expr):
        key = table._ensure_expr(key)

    if isinstance(sort_order, basestring):
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
                '  ascending: {!s}'.format(self.ascending),
                util.indent(_safe_repr(self.expr), 2)]
        return '\n'.join(rows)

    def equals(self, other):
        return (isinstance(other, SortKey) and self.expr.equals(other.expr)
                and self.ascending == other.ascending)


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
            if isinstance(expr, basestring):
                expr = table_expr[expr]

            validator.assert_valid(expr)
            if isinstance(expr, ValueExpr):
                try:
                    name = expr.get_name()
                except NotImplementedError:
                    raise ValueError("Expression is unnamed: %s" %
                                     _safe_repr(expr))
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
        tables = self.table._root_tables()
        return tables


class Aggregation(ir.BlockingTableNode, HasSchema):

    """
    agg_exprs : per-group scalar aggregates
    by : group expressions
    having : post-aggregation predicate

    TODO: not putting this in the aggregate operation yet
    where : pre-aggregation predicate
    """

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
        what = _promote_list(what)
        return [substitute_parents(x) for x in what]

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
                                      'expression, was: {!s}'
                                      .format(_safe_repr(expr)))
            if not is_scalar(expr) or not expr.is_reduction():
                raise ExpressionError('Having clause must contain a '
                                      'reduction was: {!s}'
                                      .format(_safe_repr(expr)))

        # All non-scalar refs originate from the input table
        all_exprs = self.agg_exprs + self.by + self.having
        self.table._assert_valid(all_exprs)

    def _result_schema(self):
        names = []
        types = []
        for e in self.by + self.agg_exprs:
            names.append(e.get_name())
            types.append(e.type())

        return ir.Schema(names, types)


class Add(BinaryOp):

    def output_type(self):
        from ibis.expr.rules import BinaryPromoter
        helper = BinaryPromoter(self.left, self.right, operator.add)
        return helper.get_result()


class Multiply(BinaryOp):

    def output_type(self):
        from ibis.expr.rules import BinaryPromoter
        helper = BinaryPromoter(self.left, self.right, operator.mul)
        return helper.get_result()


class Power(BinaryOp):

    def output_type(self):
        from ibis.expr.rules import PowerPromoter
        return PowerPromoter(self.left, self.right).get_result()


class Subtract(BinaryOp):

    def output_type(self):
        from ibis.expr.rules import BinaryPromoter
        helper = BinaryPromoter(self.left, self.right, operator.sub)
        return helper.get_result()


class Divide(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, ir.NumericValue):
            raise TypeError('One argument was non-numeric')

        return _shape_like_args(self.args, 'double')


class LogicalBinaryOp(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, ir.BooleanValue):
            raise TypeError('Only valid with boolean data')
        return _shape_like_args(self.args, 'boolean')


class And(LogicalBinaryOp):
    pass


class Or(LogicalBinaryOp):
    pass


class Xor(LogicalBinaryOp):
    pass


class Comparison(BinaryOp):

    def output_type(self):
        self._assert_can_compare()
        return _shape_like_args(self.args, 'boolean')

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


class Between(ValueNode):

    def __init__(self, expr, lower_bound, upper_bound):
        self.expr = expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        ValueNode.__init__(self, [expr, lower_bound, upper_bound])

    def root_tables(self):
        return ir.distinct_roots(*self.args)

    def output_type(self):
        self._assert_can_compare()
        return _shape_like_args(self.args, 'boolean')

    def _assert_can_compare(self):
        if (not self.expr._can_compare(self.lower_bound) or
                not self.expr._can_compare(self.upper_bound)):
            raise TypeError('Arguments are not comparable')


class Contains(ArrayNode):

    def __init__(self, value, options):
        self.value = as_value_expr(value)
        self.options = as_value_expr(options)
        Node.__init__(self, [self.value, self.options])

    def root_tables(self):
        exprs = [self.value, self.options]
        return util.distinct_roots(*exprs)

    def output_type(self):
        all_args = [self.value]

        options = self.options.op()
        if isinstance(options, ValueList):
            all_args += options.values
        elif isinstance(self.options, ArrayExpr):
            all_args += [self.options]
        else:
            raise TypeError(type(options))

        return _shape_like_args(all_args, 'boolean')


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
            raise ValueError('k must be positive integer, was: {}'.format(k))

        self.arg = arg
        self.k = k
        self.by = by

        Node.__init__(self, [arg, k, by])

    def root_tables(self):
        return self.arg._root_tables()

    def to_expr(self):
        return ir.BooleanArray(self)


class TimestampNow(ValueNode):

    def __init__(self):
        ValueNode.__init__(self, [])

    def output_type(self):
        return ir.TimestampScalar


class ExtractTimestampField(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, ir.TimestampValue):
            raise AssertionError
        return _shape_like(self.arg, 'int32')

    def to_expr(self):
        klass = self.output_type()
        return klass(self)


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


def _promote_list(val):
    if not isinstance(val, list):
        val = [val]
    return val
