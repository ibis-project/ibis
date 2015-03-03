# Copyright 2014 Cloudera Inc.
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

# Here we are working on the basic data structure for building Ibis expressions
# (an intermediate representation which can be compiled to a target backend,
# e.g. Impala SQL)
#
# The design and class structure here must be explicitly guided by the kind of
# user experience (i.e., highly interactive, suitable for introspection and
# console/notebook use) we want to deliver.
#
# All data structures should be treated as immutable (as much as Python objects
# are immutable; we'll behave as if they are).
#
# Expressions can be parameterized both by tables (known schema, but not bound
# to a particular table in any database), fields, and literal values. In order
# to execute an expression containing parameters, the user must perform a
# actual data. Mixing table and field parameters can lead to tricky binding
# scenarios -- essentially all unbound field parameters within a particular
# table expression must originate from the same concrete table. Internally we
# can identify the "logical tables" in the expression and present those to the
# user for the binding. Good introspection capability will be important
# here. Literal parameters are much simpler. A literal parameter is declared
# and used however many times the user wishes; binding in that case simply
# introduces the actual value to be used.
#
# In some cases, we'll want to be able to indicate that a parameter can either
# be a scalar or array expression. In this case the binding requirements may be
# somewhat more lax.

from collections import defaultdict

import operator
import re

from ibis.common import RelationError, ExpressionError
import ibis.common as com
import ibis.util as util


class Parameter(object):

    """
    Placeholder, to be implemented
    """

    pass


class DataType(object):
    pass


class DecimalType(DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale):
        self.precision = precision
        self.scale = scale

    def __repr__(self):
        return ('decimal(precision=%s, scale=%s)'
                % (self.precision, self.scale))

    def __eq__(self, other):
        if not isinstance(other, DecimalType):
            return False

        return (self.precision == other.precision and
                self.scale == other.scale)

    def array_ctor(self):
        def constructor(op, name=None):
            return DecimalArray(op, self, name=name)
        return constructor

    def scalar_ctor(self):
        def constructor(op, name=None):
            return DecimalScalar(op, self, name=name)
        return constructor


#----------------------------------------------------------------------


class Schema(object):

    """
    Holds table schema information
    """

    def __init__(self, names, types):
        if not isinstance(names, list):
            names = list(names)
        self.names = names
        self.types = [_validate_type(x) for x in types]

        self._name_locs = dict((v, i) for i, v in enumerate(self.names))

        if len(self._name_locs) < len(self.names):
            raise com.IntegrityError('Duplicate column names')

    def __repr__(self):
        return self._repr()

    def __len__(self):
        return len(self.names)

    def _repr(self):
        return "%s(%s, %s)" % (type(self).__name__, repr(self.names),
                               repr(self.types))

    def __contains__(self, name):
        return name in self._name_locs

    @classmethod
    def from_tuples(cls, values):
        if len(values):
            names, types = zip(*values)
        else:
            names, types = [], []
        return Schema(names, types)

    @classmethod
    def from_dict(cls, values):
        names = list(values.keys())
        types = values.values()
        return Schema(names, types)

    def equals(self, other):
        return ((self.names == other.names) and
                (self.types == other.types))

    def get_type(self, name):
        return self.types[self._name_locs[name]]

    def append(self, schema):
        names = self.names + schema.names
        types = self.types + schema.types
        return Schema(names, types)


def _validate_type(t):
    if isinstance(t, DataType):
        return t

    parsed_type = _parse_type(t)
    if parsed_type is not None:
        return parsed_type

    if t not in _array_types:
        raise ValueError('Invalid type: %s' % repr(t))
    return t



_DECIMAL_RE = re.compile('decimal\((\d+),[\s]*(\d+)\)')


def _parse_decimal(t):
    m = _DECIMAL_RE.match(t)
    if m:
        precision, scale = m.groups()
        return DecimalType(int(precision), int(scale))

    if t == 'decimal':
        # From the Impala documentation
        return DecimalType(9, 0)


_type_parsers = [
    _parse_decimal
]


def _parse_type(t):
    for parse_fn in _type_parsers:
        parsed = parse_fn(t)
        if parsed is not None:
            return parsed
    return None


class HasSchema(object):

    """
    Base class representing a structured dataset with a well-defined
    schema.

    Base implementation is for tables that do not reference a particular
    concrete dataset or database table.
    """

    def __init__(self, schema, name=None):
        assert isinstance(schema, Schema)
        self._schema = schema
        self._name = name

    def __repr__(self):
        return self._repr()

    def _repr(self):
        return "%s(%s)" % (type(self).__name__, repr(self.schema))

    @property
    def schema(self):
        return self._schema

    def get_schema(self):
        return self._schema

    def has_schema(self):
        return True

    @property
    def name(self):
        return self._name

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self.schema.equals(other.schema)

    def root_tables(self):
        return [self]


#----------------------------------------------------------------------


class Expr(object):

    """

    """

    def __init__(self, arg):
        # TODO: all inputs must inherit from a common table API
        self._arg = arg

    def __repr__(self):
        return self._repr()

    def _repr(self):
        from ibis.expr.format import ExprFormatter
        return ExprFormatter(self).get_result()

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, name=name)
        return factory

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self._arg.equals(other._arg)

    def op(self):
        raise NotImplementedError

    def _can_compare(self, other):
        return False

    def _root_tables(self):
        return self.op().root_tables()

    def _get_unbound_tables(self):
        # The expression graph may contain one or more tables of a particular
        # known schema
        pass



class Node(object):

    """
    Node is the base class for all relational algebra and analytical
    functionality. It transforms the input expressions into an output
    expression.

    Each node implementation is responsible for validating the inputs,
    including any type promotion and / or casting issues, and producing a
    well-typed expression

    Note that Node is deliberately not made an expression subclass: think
    of Node as merely a typed expression builder.
    """

    def __init__(self, args):
        self.args = args

    def __repr__(self):
        return self._repr()

    def _repr(self):
        # Quick and dirty to get us started
        opname = type(self).__name__
        pprint_args = [repr(x) for x in self.args]
        return '%s(%s)' % (opname, ', '.join(pprint_args))

    def flat_args(self):
        for arg in self.args:
            if isinstance(arg, (tuple, list)):
                for x in arg:
                    yield x
            else:
                yield arg

    def equals(self, other):
        if type(self) != type(other):
            return False

        if len(self.args) != len(other.args):
            return False

        def is_equal(left, right):
            if isinstance(left, list):
                if not isinstance(right, list):
                    return False
                for a, b in zip(left, right):
                    if not is_equal(a, b):
                        return False
                return True

            if hasattr(left, 'equals'):
                return left.equals(right)
            else:
                return left == right
            return True

        for left, right in zip(self.args, other.args):
            if not is_equal(left, right):
                return False
        return True

    def to_expr(self):
        """
        This function must resolve the output type of the expression and return
        the node wrapped in the appropriate ValueExpr type.
        """
        raise NotImplementedError


class ValueNode(Node):

    def to_expr(self):
        klass = self.output_type()
        return klass(self)

    def _ensure_value(self, expr):
        if not isinstance(expr, ValueExpr):
            raise TypeError('Must be a value, got: %s' % repr(expr))

    def _ensure_array(self, expr):
        if not isinstance(expr, ArrayExpr):
            raise TypeError('Must be an array, got: %s' % repr(expr))

    def _ensure_scalar(self, expr):
        if not isinstance(expr, ScalarExpr):
            raise TypeError('Must be a scalar, got: %s' % repr(expr))

    def root_tables(self):
        return self.arg._root_tables()

    def output_type(self):
        raise NotImplementedError

    def resolve_name(self):
        raise NotImplementedError


class UnnamedMarker(object):
    pass


unnamed = UnnamedMarker()


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
        if isinstance(self.value, bool):
            klass = BooleanScalar
        elif isinstance(self.value, (int, long)):
            int_type = _int_literal_class(self.value)
            klass = scalar_type(int_type)
        elif isinstance(self.value, float):
            klass = DoubleScalar
        elif isinstance(self.value, basestring):
            klass = StringScalar

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
        return NullScalar

    def root_tables(self):
        return []


class ArrayNode(ValueNode):

    def __init__(self, expr):
        self._ensure_array(expr)
        ValueNode.__init__(self, [expr])

    def output_type(self):
        return NotImplementedError

    def to_expr(self):
        klass = self.output_type()
        return klass(self)


class TableNode(Node):

    def get_type(self, name):
        return self.get_schema().get_type(name)

    def to_expr(self):
        return TableExpr(self)


class BlockingTableNode(TableNode):
    # Try to represent the fact that whatever lies here is a semantically
    # distinct table. Like projections, aggregations, and so forth
    pass


class PhysicalTable(BlockingTableNode, HasSchema):

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


class SQLQueryResult(BlockingTableNode, HasSchema):

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
        klass = array_type(ctype)
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
        klass = array_type(ctype)
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
        self.target_type = _validate_type(target_type.lower())
        ValueNode.__init__(self, [arg, self.target_type])

    def resolve_name(self):
        return self.arg.get_name()

    def output_type(self):
        # TODO: error handling for invalid casts
        return _shape_like(self.arg, self.target_type)


class Negate(UnaryOp):

    def output_type(self):
        return type(self.arg)


def _negate(expr):
    op = expr.op()
    if hasattr(op, 'negate'):
        return op.negate()
    else:
        return Negate(expr)


class IsNull(UnaryOp):

    def output_type(self):
        return _shape_like(self.arg, 'boolean')


class NotNull(UnaryOp):

    def output_type(self):
        return _shape_like(self.arg, 'boolean')


def _shape_like(arg, out_type):
    if isinstance(arg, ScalarExpr):
        return scalar_type(out_type)
    else:
        return array_type(out_type)


def _shape_like_args(args, out_type):
    if util.any_of(args, ArrayExpr):
        return array_type(out_type)
    else:
        return scalar_type(out_type)


class RealUnaryOp(UnaryOp):

    _allow_boolean = True

    def output_type(self):
        if not isinstance(self.arg, NumericValue):
            raise TypeError('Only implemented for numeric types')
        elif isinstance(self.arg, BooleanValue) and not self._allow_boolean:
            raise TypeError('Not implemented for boolean types')

        return _shape_like(self.arg, 'double')


class Exp(RealUnaryOp):
    pass


class Sqrt(RealUnaryOp):
    pass


class Log(RealUnaryOp):

    _allow_boolean = False


class Log2(RealUnaryOp):

    _allow_boolean = False


class Log10(RealUnaryOp):

    _allow_boolean = False


class StringUnaryOp(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, StringValue):
            raise TypeError('Only implemented for numeric types')
        return _shape_like(self.arg, 'string')


class Uppercase(StringUnaryOp):
    pass


class Lowercase(StringUnaryOp):
    pass


#----------------------------------------------------------------------


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
        return _distinct_roots(self.left, self.right)

    def output_type(self):
        raise NotImplementedError


def _distinct_roots(*args):
    all_roots = []
    for arg in args:
        all_roots.extend(arg._root_tables())
    return util.unique_by_key(all_roots, id)


#----------------------------------------------------------------------


class Reduction(ArrayNode):

    def __init__(self, arg):
        self.arg = arg
        ArrayNode.__init__(self, arg)

    def root_tables(self):
        return self.arg._root_tables()

    def resolve_name(self):
        return self.arg.get_name()


class Count(Reduction):
    # TODO: count(col) takes down Impala, must always do count(*) in generated
    # SQL

    def __init__(self, expr):
        # TODO: counts are actually table-level operations. Let's address
        # during the SQL generation exercise
        if not isinstance(expr, (ArrayExpr, TableExpr)):
            raise TypeError
        self.arg = expr
        ValueNode.__init__(self, [expr])

    def output_type(self):
        return Int64Scalar


class Sum(Reduction):

    def output_type(self):
        if isinstance(self.arg, (IntegerValue, BooleanValue)):
            return Int64Scalar
        elif isinstance(self.arg, FloatingValue):
            return DoubleScalar
        elif isinstance(self.arg, DecimalValue):
            return _decimal_scalar_ctor(self.arg.precision, 38)
        else:
            raise TypeError(self.arg)


class Mean(Reduction):

    def output_type(self):
        if isinstance(self.arg, DecimalValue):
            return _decimal_scalar_ctor(self.arg.precision, 38)
        elif isinstance(self.arg, NumericValue):
            return DoubleScalar
        else:
            raise NotImplementedError


def _decimal_scalar_ctor(precision, scale):
    out_type = DecimalType(precision, scale)
    return DecimalScalar._make_constructor(out_type)


class StdDeviation(Reduction):
    pass


class Max(Reduction):

    def output_type(self):
        if isinstance(self.arg, DecimalValue):
            return _decimal_scalar_ctor(self.arg.precision, 38)
        else:
            return scalar_type(self.arg.type())


class Min(Reduction):

    def output_type(self):
        if isinstance(self.arg, DecimalValue):
            return _decimal_scalar_ctor(self.arg.precision, 38)
        else:
            return scalar_type(self.arg.type())

#----------------------------------------------------------------------
# Distinct stuff

class Distinct(BlockingTableNode, HasSchema):

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

        BlockingTableNode.__init__(self, [table])
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
        op = CountDistinct(self.arg)
        return op.to_expr()


class CountDistinct(Reduction):

    def output_type(self):
        return Int64Scalar


def _count(expr):
    op = expr.op()
    if isinstance(op, DistinctArray):
        return op.count()
    else:
        return Count(expr).to_expr()


#----------------------------------------------------------------------
# Boolean reductions and semi/anti join support

class Any(ValueNode):

    # Depending on the kind of input boolean array, the result might either be
    # array-like (an existence-type predicate) or scalar (a reduction)

    def __init__(self, expr):
        if not isinstance(expr, BooleanArray):
            raise ValueError('Expression must be a boolean array')

        self.arg = expr
        ValueNode.__init__(self, [expr])

    def output_type(self):
        roots = self.arg._root_tables()
        if len(roots) > 1:
            return BooleanArray
        else:
            # A reduction
            return BooleanScalar

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

        if not isinstance(case_expr, BooleanValue):
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
        return _distinct_roots(*all_exprs)

    def output_type(self):
        out_exprs = self.results + [self.default]
        typename = _highest_precedence_type(out_exprs)
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
        return _distinct_roots(*all_exprs)

    def output_type(self):
        out_exprs = self.results + [self.default]
        typename = _highest_precedence_type(out_exprs)
        return _shape_like_args(self.cases, typename)


class Join(TableNode):

    def __init__(self, left, right, join_predicates):
        from ibis.expr.analysis import ExprValidator

        if not isinstance(left, TableExpr):
            raise TypeError('Can only join table expressions, got %s for '
                            'left table' % type(left))

        if not isinstance(right, TableExpr):
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

            if not isinstance(pred, BooleanArray):
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
            return _distinct_roots(self.left, self.right)



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


class Union(BlockingTableNode, HasSchema):

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


class Limit(BlockingTableNode):

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


class SortBy(TableNode):

    # Q: Will SortBy always require a materialized schema?

    def __init__(self, table_expr, sort_keys):
        self.table = table_expr
        self.keys = [_to_sort_key(self.table, k) for k in sort_keys]

        TableNode.__init__(self, [self.table, self.keys])

    def get_schema(self):
        return self.table.schema()

    def has_schema(self):
        return self.table.op().has_schema()

    def root_tables(self):
        tables = self.table._root_tables()
        return tables


class SortKey(object):

    def __init__(self, expr, ascending=True):
        if not isinstance(expr, ArrayExpr):
            raise ExpressionError('Must be an array/column expression')

        self.expr = expr
        self.ascending = ascending

    def __repr__(self):
        # Temporary
        rows = ['Sort key:',
                '  ascending: {!s}'.format(self.ascending),
                util.indent(repr(self.expr), 2)]
        return '\n'.join(rows)

    def equals(self, other):
        return (isinstance(other, SortKey) and self.expr.equals(other.expr)
                and self.ascending == other.ascending)


def desc(expr):
    return SortKey(expr, ascending=False)


def _to_sort_key(table, key):
    if isinstance(key, SortKey):
        return key

    if isinstance(key, (tuple, list)):
        key, sort_order = key
    else:
        sort_order = True

    if not isinstance(key, Expr):
        key = table._ensure_expr(key)

    if isinstance(sort_order, basestring):
        if sort_order.lower() in ('desc', 'descending'):
            sort_order = False
        elif not isinstance(sort_order, bool):
            sort_order = bool(sort_order)

    return SortKey(key, ascending=sort_order)




class SelfReference(BlockingTableNode, HasSchema):

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


class Projection(BlockingTableNode, HasSchema):

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
                expr = table[expr]

            validator.assert_valid(expr)
            if isinstance(expr, ValueExpr):
                try:
                    name = expr.get_name()
                except NotImplementedError:
                    raise ValueError("Expression is unnamed: %s" % repr(expr))
                names.append(name)
                types.append(expr.type())
            elif isinstance(expr, TableExpr):
                schema = expr.schema()
                names.extend(schema.names)
                types.extend(schema.types)
            else:
                raise NotImplementedError

            clean_exprs.append(expr)

        # validate uniqueness
        schema = Schema(names, types)

        HasSchema.__init__(self, schema)
        Node.__init__(self, [table_expr] + [clean_exprs])

        self.table = table_expr
        self.selections = clean_exprs

    def substitute_table(self, table_expr):
        return Projection(table_expr, self.selections)

    def root_tables(self):
        tables = self.table._root_tables()
        return tables


class Aggregation(BlockingTableNode, HasSchema):

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

        TableNode.__init__(self, [table, agg_exprs, self.by, self.having])

        schema = self._result_schema()
        HasSchema.__init__(self, schema)

    def _rewrite_exprs(self, what):
        from ibis.expr.analysis import substitute_parents

        if isinstance(what, (list, tuple)):
            return [substitute_parents(x) for x in what]
        else:
            return substitute_parents(what)

    def substitute_table(self, table_expr):
        return Aggregation(table_expr, self.agg_exprs, by=self.by,
                           having=self.having)

    def _validate(self):
        # All aggregates are valid
        for expr in self.agg_exprs:
            if not isinstance(expr, ScalarExpr) or not expr.is_reduction():
                raise TypeError('Passed a non-aggregate expression: %s' %
                                repr(expr))

        for expr in self.having:
            if not isinstance(expr, BooleanScalar):
                raise ExpressionError('Having clause must be boolean '
                                      'expression, was: {!r}'
                                      .format(expr))
            if not isinstance(expr, ScalarExpr) or not expr.is_reduction():
                raise ExpressionError('Having clause must contain a '
                                      'reduction was: {!r}'
                                      .format(expr))

        # All non-scalar refs originate from the input table
        all_exprs = self.agg_exprs + self.by + self.having
        self.table._assert_valid(all_exprs)

    def _result_schema(self):
        names = []
        types = []
        for e in self.by + self.agg_exprs:
            names.append(e.get_name())
            types.append(e.type())

        return Schema(names, types)


class Add(BinaryOp):

    def output_type(self):
        helper = BinaryPromoter(self.left, self.right, operator.add)
        return helper.get_result()


class Multiply(BinaryOp):

    def output_type(self):
        helper = BinaryPromoter(self.left, self.right, operator.mul)
        return helper.get_result()


class Power(BinaryOp):

    def output_type(self):
        return PowerPromoter(self.left, self.right).get_result()


class Subtract(BinaryOp):

    def output_type(self):
        helper = BinaryPromoter(self.left, self.right, operator.sub)
        return helper.get_result()


class Divide(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, NumericValue):
            raise TypeError('One argument was non-numeric')

        return _shape_like_args(self.args, 'double')


class LogicalBinaryOp(BinaryOp):

    def output_type(self):
        if not util.all_of(self.args, BooleanValue):
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
        return _distinct_roots(*self.args)

    def output_type(self):
        self._assert_can_compare()
        return _shape_like_args(self.args, 'boolean')

    def _assert_can_compare(self):
        if (not self.expr._can_compare(self.lower_bound) or
            not self.expr._can_compare(self.upper_bound)):
            raise TypeError('Arguments are not comparable')


class BinaryPromoter(object):
    # placeholder for type promotions for basic binary arithmetic

    def __init__(self, left, right, op):
        self.args = [left, right]
        self.left = left
        self.right = right
        self.op = op

        self._check_compatibility()

    def get_result(self):
        promoted_type = self._get_type()
        return _shape_like_args(self.args, promoted_type)

    def _get_type(self):
        if util.any_of(self.args, FloatingValue):
            if util.any_of(self.args, DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.all_of(self.args, IntegerValue):
            return self._get_int_type()
        else:
            raise NotImplementedError

    def _get_int_type(self):
        deps = [x.op() for x in self.args]

        if util.all_of(deps, Literal):
            return _smallest_int_containing(
                [self.op(deps[0].value, deps[1].value)])
        elif util.any_of(deps, Literal):
            if isinstance(deps[0], Literal):
                val = deps[0].value
                atype = self.args[1].type()
            else:
                val = deps[1].value
                atype = self.args[0].type()
            return _int_one_literal_promotion(atype, val, self.op)
        else:
            return _int_bounds_promotion(self.left.type(),
                                         self.right.type(), self.op)

    def _check_compatibility(self):
        if (util.any_of(self.args, StringValue) and
                not util.all_of(self.args, StringValue)):
            raise TypeError('String and non-string incompatible')


class PowerPromoter(BinaryPromoter):

    def __init__(self, left, right):
        super(PowerPromoter, self).__init__(left, right, operator.pow)

    def _get_type(self):
        rval = self.args[1].op()

        if util.any_of(self.args, FloatingValue):
            if util.any_of(self.args, DoubleValue):
                return 'double'
            else:
                return 'float'
        elif isinstance(rval, Literal) and rval.value < 0:
            return 'double'
        elif util.all_of(self.args, IntegerValue):
            return self._get_int_type()
        else:
            raise NotImplementedError


def _int_bounds_promotion(ltype, rtype, op):
    lmin, lmax = _int_bounds[ltype]
    rmin, rmax = _int_bounds[rtype]

    values = [op(lmin, rmin), op(lmin, rmax),
              op(lmax, rmin), op(lmax, rmax)]

    return _smallest_int_containing(values, allow_overflow=True)


def _int_one_literal_promotion(atype, lit_val, op):
    amin, amax = _int_bounds[atype]
    bound_type = _smallest_int_containing([op(amin, lit_val),
                                           op(amax, lit_val)],
                                          allow_overflow=True)
    # In some cases, the bounding type might be int8, even though neither of
    # the types are that small. We want to ensure the containing type is _at
    # least_ as large as the smallest type in the expression
    return _largest_int([bound_type, atype])


def _smallest_int_containing(values, allow_overflow=False):
    containing_types = [_int_literal_class(x, allow_overflow=allow_overflow)
                        for x in values]
    return _largest_int(containing_types)


class Contains(ArrayNode):

    def __init__(self, value, options):
        self.value = as_value_expr(value)
        self.options = as_value_expr(options)
        Node.__init__(self, [self.value, self.options])

    def root_tables(self):
        exprs = [self.value, self.options]
        return _distinct_roots(*exprs)

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
        return BooleanArray(self)


def _binop_expr(name, klass):
    def f(self, other):
        other = as_value_expr(other)
        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary ops, like radd, etc.
    def f(self, other):
        other = as_value_expr(other)
        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _unary_op(name, klass):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    return f


class ValueExpr(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    _implicit_casts = set()

    def __init__(self, arg, name=None):
        Expr.__init__(self, arg)
        self._name = name

    def type(self):
        return self._typename

    def _base_type(self):
        # Parametric types like "decimal"
        return self.type()

    def _can_cast_implicit(self, typename):
        return typename in self._implicit_casts or typename == self.type()

    def op(self):
        return self._arg

    def get_name(self):
        if self._name is not None:
            # This value has been explicitly named
            return self._name

        # In some but not all cases we can get a name from the node that
        # produces the value
        return self.op().resolve_name()

    def name(self, name):
        return type(self)(self._arg, name=name)

    def cast(self, target_type):
        """
        Cast value(s) to indicated data type. Values that cannot be
        successfully casted

        Parameters
        ----------
        target_type : data type name

        Returns
        -------
        cast_expr : ValueExpr
        """
        # validate
        op = Cast(self, target_type)

        if op.target_type == self.type():
            # noop case if passed type is the same
            return self
        else:
            return op.to_expr()

    def between(self, lower, upper):
        """
        Check if the input expr falls between the lower/upper bounds
        passed. Bounds are inclusive. All arguments must be comparable.

        Returns
        -------
        is_between : BooleanValue
        """
        lower = as_value_expr(lower)
        upper = as_value_expr(upper)
        op = Between(self, lower, upper)
        return op.to_expr()

    def isin(self, values):
        """
        Check whether the value expression is contained within the indicated
        list of values.

        Parameters
        ----------
        values : list, tuple, or array expression
          The values can be scalar or array-like. Each of them must be
          comparable with the calling expression, or None (NULL).

        Examples
        --------
        expr = table.strings.isin(['foo', 'bar', 'baz'])

        expr2 = table.strings.isin(table2.other_string_col)

        Returns
        -------
        contains : BooleanValue
        """
        op = Contains(self, values)
        return op.to_expr()

    def notin(self, values):
        """
        Like isin, but checks whether this expression's value(s) are not
        contained in the passed values. See isin docs for full usage.
        """
        op = NotContains(self, values)
        return op.to_expr()

    isnull = _unary_op('isnull', IsNull)
    notnull = _unary_op('notnull', NotNull)

    def ifnull(self, sub_expr):
        pass

    __add__ = _binop_expr('__add__', Add)
    __sub__ = _binop_expr('__sub__', Subtract)
    __mul__ = _binop_expr('__mul__', Multiply)
    __div__ = _binop_expr('__div__', Divide)
    __pow__ = _binop_expr('__pow__', Power)

    __radd__ = _rbinop_expr('__radd__', Add)
    __rsub__ = _rbinop_expr('__rsub__', Subtract)
    __rmul__ = _rbinop_expr('__rmul__', Multiply)
    __rdiv__ = _rbinop_expr('__rdiv__', Divide)
    __rpow__ = _binop_expr('__rpow__', Power)

    __eq__ = _binop_expr('__eq__', Equals)
    __ne__ = _binop_expr('__ne__', NotEquals)
    __ge__ = _binop_expr('__ge__', GreaterEqual)
    __gt__ = _binop_expr('__gt__', Greater)
    __le__ = _binop_expr('__le__', LessEqual)
    __lt__ = _binop_expr('__lt__', Less)


def as_value_expr(val):
    if not isinstance(val, Expr):
        if isinstance(val, (tuple, list)):
            val = value_list(val)
        else:
            val = literal(val)

    return val


class ValueList(ArrayNode):

    """
    Data structure for a list of value expressions
    """

    def __init__(self, args):
        self.values = [as_value_expr(x) for x in args]
        Node.__init__(self, [self.values])

    def root_tables(self):
        return _distinct_roots(*self.values)

    def to_expr(self):
        return ListExpr(self)



class ScalarExpr(ValueExpr):

    def is_reduction(self):
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
                if isinstance(arg, ScalarExpr) and has_reduction(arg.op()):
                    return True

            return False

        return has_reduction(self.op())


class ArrayExpr(ValueExpr):

    def parent(self):
        return self._arg

    def distinct(self):
        """
        Compute set of unique values occurring in this array. Can not be used
        in conjunction with other array expressions from the same context
        (because it's a cardinality-modifying pseudo-reduction).
        """
        op = DistinctArray(self)
        return op.to_expr()

    def nunique(self):
        """
        Shorthand for foo.distinct().count(); computing the number of unique
        values in an array.
        """
        return CountDistinct(self).to_expr()

    def topk(self, k, by=None):
        """
        Produces
        """
        op = TopK(self, k, by=by)
        return op.to_expr()

    def count(self):
        return _count(self)

    def bottomk(self, k, by=None):
        raise NotImplementedError

    def case(self):
        """
        Create a new SimpleCaseBuilder to chain multiple if-else
        statements. Add new search expressions with the .when method. These
        must be comparable with this array expression. Conclude by calling
        .end()

        Examples
        --------
        case_expr = (expr.case()
                     .when(case1, output1)
                     .when(case2, output2)
                     .default(default_output)
                     .end())

        Returns
        -------
        builder : CaseBuilder
        """
        return SimpleCaseBuilder(self)

    def cases(self, case_result_pairs, default=None):
        """
        Create a case expression in one shot.

        Returns
        -------
        case_expr : SimpleCase
        """
        builder = self.case()
        for case, result in case_result_pairs:
            builder = builder.when(case, result)
        if default is not None:
            builder = builder.else_(default)
        return builder.end()

    def to_projection(self):
        """
        Promote this column expression to a table projection
        """
        roots = self._root_tables()
        if len(roots) > 1:
            raise RelationError('Cannot convert array expression involving '
                                'multiple base table references to a '
                                'projection')

        table = TableExpr(roots[0])
        return table.projection([self])


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)
    """
    return SearchedCaseBuilder()


class TableExpr(Expr):

    def op(self):
        return self._arg

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator
        ExprValidator([self]).validate_all(exprs)

    def __getitem__(self, what):
        if isinstance(what, basestring):
            return self.get_column(what)
        elif isinstance(what, (list, tuple)):
            # Projection case
            return self.projection(what)
        elif isinstance(what, BooleanArray):
            # Boolean predicate
            return self.filter([what])
        else:
            raise NotImplementedError

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if not self._is_materialized() or key not in self.schema():
                raise

            return self.get_column(key)

    def __dir__(self):
        attrs = dir(type(self))
        if self._is_materialized():
            attrs = list(sorted(set(attrs + self.schema().names)))
        return attrs

    def _resolve(self, exprs):
        # Stash this helper method here for now
        out_exprs = []
        for expr in exprs:
            expr = self._ensure_expr(expr)
            out_exprs.append(expr)
        return out_exprs

    def _get_type(self, name):
        return self._arg.get_type(name)

    def materialize(self):
        if self._is_materialized():
            return self
        else:
            op = MaterializedJoin(self)
            return TableExpr(op)

    def get_columns(self, iterable):
        return [self.get_column(x) for x in iterable]

    def get_column(self, name):
        ref = TableColumn(name, self)
        return ref.to_expr()

    def schema(self):
        if not self._is_materialized():
            raise Exception('Table operation is not yet materialized')
        return self.op().get_schema()

    def to_array(self):
        """
        Single column tables can be viewed as arrays.
        """
        op = TableArrayView(self)
        return op.to_expr()

    def _is_materialized(self):
        # The operation produces a known schema
        op = self.op()
        return isinstance(op, HasSchema) or op.has_schema()

    def view(self):
        """
        Create a new table expression that is semantically equivalent to the
        current one, but is considered a distinct relation for evaluation
        purposes (e.g. in SQL).

        For doing any self-referencing operations, like a self-join, you will
        use this operation to create a reference to the current table
        expression.

        Returns
        -------
        expr : TableExpr
        """
        return TableExpr(SelfReference(self))

    def add_column(self, expr, name=None):
        if not isinstance(expr, ArrayExpr):
            raise TypeError('Must pass array expression')

        if name is not None:
            expr = expr.name(name)

        # New column originates from this table expression if at all
        self._assert_valid([expr])
        return self.projection([self, expr])

    def add_columns(self, what):
        raise NotImplementedError

    def count(self):
        return Count(self).to_expr()

    def distinct(self):
        """
        Compute set of unique rows/tuples occurring in this table
        """
        op = Distinct(self)
        return op.to_expr()

    def cross_join(self, other, prefixes=None):
        """

        """
        op = CrossJoin(self, other)
        return TableExpr(op)

    def inner_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = InnerJoin(self, other, predicates)
        return TableExpr(op)

    def left_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = LeftJoin(self, other, predicates)
        return TableExpr(op)

    def outer_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = OuterJoin(self, other, predicates)
        return TableExpr(op)

    def semi_join(self, other, predicates, prefixes=None):
        """

        """
        op = LeftSemiJoin(self, other, predicates)
        return TableExpr(op)

    def anti_join(self, other, predicates, prefixes=None):
        """

        """
        op = LeftAntiJoin(self, other, predicates)
        return TableExpr(op)

    def projection(self, exprs):
        """

        """
        import ibis.expr.analysis as L

        clean_exprs = []

        validator = L.ExprValidator([self])

        for expr in exprs:
            expr = self._ensure_expr(expr)

            # Perform substitution only if we share common roots
            if validator.shares_some_roots(expr):
                expr = L.substitute_parents(expr)
            clean_exprs.append(expr)

        op = L._maybe_fuse_projection(self, clean_exprs)
        return TableExpr(op)

    def _ensure_expr(self, expr):
        if isinstance(expr, basestring):
            expr = self[expr]

        return expr

    def filter(self, predicates):
        """

        Parameters
        ----------

        Returns
        -------
        filtered_expr : TableExpr
        """
        import ibis.expr.analysis as L
        op = L.apply_filter(self, predicates)
        return TableExpr(op)

    def group_by(self, by):
        """
        Create an intermediate grouped table expression, pending some group
        operation to be applied with it.

        Examples
        --------
        x.group_by([b1, b2]).aggregate(metrics)

        Returns
        -------
        grouped_expr : GroupedTableExpr
        """
        return GroupedTableExpr(self, by)

    def aggregate(self, agg_exprs, by=None, having=None):
        """
        Parameters
        ----------

        Returns
        -------
        agg_expr : TableExpr
        """
        op = Aggregation(self, agg_exprs, by=by, having=having)
        return TableExpr(op)

    def limit(self, n, offset=None):
        """

        Parameters
        ----------

        Returns
        -------
        limited : TableExpr
        """
        op = Limit(self, n, offset=offset)
        return TableExpr(op)

    def sort_by(self, what):
        if not isinstance(what, list):
            what = [what]

        op = SortBy(self, what)
        return TableExpr(op)

    def union(self, other, distinct=False):
        """
        Form the table set union of two table expressions having identical
        schemas.

        Parameters
        ----------
        other : TableExpr
        distinct : boolean, default False
            Only union distinct rows not occurring in the calling table (this
            can be very expensive, be careful)

        Returns
        -------
        union : TableExpr
        """
        op = Union(self, other, distinct=distinct)
        return TableExpr(op)


class GroupedTableExpr(object):
    """
    Helper intermediate construct
    """
    def __init__(self, table, by):
        if not isinstance(by, (list, tuple)):
            if not isinstance(by, Expr):
                by = table._resolve([by])
            else:
                by = [by]

        self.table = table
        self.by = by

    def aggregate(self, metrics, having=None):
        return self.table.aggregate(metrics, by=self.by, having=having)


#------------------------------------------------------------------------------
# Declare all typed ValueExprs. This is what the user will actually interact
# with: an instance of each is well-typed and includes all valid methods
# defined for each type.


def _boolean_binary_op(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _boolean_binary_rop(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _agg_function(name, klass):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    return f


class AnyValue(ValueExpr):

    _typename = 'any'


class NullValue(AnyValue):

    _typename = 'null'

    def _can_cast_implicit(self, typename):
        return True


class NumericValue(AnyValue):

    __neg__ = _unary_op('__neg__', _negate)

    exp = _unary_op('exp', Exp)
    sqrt = _unary_op('sqrt', Sqrt)

    log = _unary_op('log', Log)
    log2 = _unary_op('log2', Log2)
    log10 = _unary_op('log10', Log10)

    def _can_compare(self, other):
        return isinstance(other, NumericValue)


class IntegerValue(NumericValue):
    pass


class BooleanValue(NumericValue):

    _typename = 'boolean'

    # TODO: logical binary operators for BooleanValue
    __and__ = _boolean_binary_op('__and__', And)
    __or__ = _boolean_binary_op('__or__', Or)
    __xor__ = _boolean_binary_op('__xor__', Xor)

    __rand__ = _boolean_binary_rop('__rand__', And)
    __ror__ = _boolean_binary_rop('__ror__', Or)
    __rxor__ = _boolean_binary_rop('__rxor__', Xor)

    def ifelse(self, true_expr, false_expr):
        """
        Shorthand for implementing ternary expressions

        bool_expr.ifelse(0, 1)
        e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END
        """
        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        return case().when(self, true_expr).else_(false_expr).end()


class Int8Value(IntegerValue):

    _typename = 'int8'
    _implicit_casts = {'int16', 'int32', 'int64', 'float', 'double'}


class Int16Value(IntegerValue):

    _typename = 'int16'
    _implicit_casts = {'int32', 'int64', 'float', 'double'}

class Int32Value(IntegerValue):

    _typename = 'int32'
    _implicit_casts = {'int64', 'float', 'double'}


class Int64Value(IntegerValue):

    _typename = 'int64'
    _implicit_casts = {'float', 'double'}




class FloatingValue(NumericValue):
    pass


class FloatValue(FloatingValue):

    _typename = 'float'
    _implicit_casts = {'double'}


class DoubleValue(FloatingValue):

    _typename = 'double'


class StringValue(AnyValue):

    _typename = 'string'

    def _can_compare(self, other):
        return isinstance(other, StringValue)

    lower = _unary_op('lower', Lowercase)
    upper = _unary_op('upper', Uppercase)


class DecimalValue(NumericValue):

    _typename = 'decimal'
    _implicit_casts = {'float', 'double'}

    def __init__(self, meta):
        self.meta = meta
        self.precision = meta.precision
        self.scale = meta.scale

    @classmethod
    def _make_constructor(cls, meta):
        def constructor(arg, name=None):
            return cls(arg, meta, name=name)
        return constructor



class ExtractTimestampField(UnaryOp):

    def output_type(self):
        if not isinstance(self.arg, TimestampValue):
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


def _extract_field(name, klass):
    def f(self):
        op = klass(self)
        return op.to_expr()
    f.__name__ = name
    return f


class TimestampValue(AnyValue):

    _typename = 'timestamp'

    year = _extract_field('year', ExtractYear)
    month = _extract_field('month', ExtractMonth)
    day = _extract_field('day', ExtractDay)
    hour = _extract_field('hour', ExtractHour)
    minute = _extract_field('minute', ExtractMinute)
    second = _extract_field('second', ExtractSecond)
    millisecond = _extract_field('millisecond', ExtractMillisecond)


class NumericArray(ArrayExpr, NumericValue):

    def count(self):
        # TODO: should actually get the parent table expression here
        return Count(self).to_expr()

    sum = _agg_function('sum', Sum)
    mean = _agg_function('mean', Mean)
    min = _agg_function('min', Min)
    max = _agg_function('max', Max)


class NullScalar(NullValue, ScalarExpr):
    pass


class ListExpr(ArrayExpr, AnyValue):
    pass


class BooleanScalar(ScalarExpr, BooleanValue):
    pass


class BooleanArray(NumericArray, BooleanValue):

    def any(self):
        op = Any(self)
        return op.to_expr()

    def none(self):
        pass

    def all(self):
        raise NotImplementedError


class Int8Scalar(ScalarExpr, Int8Value):
    pass


class Int8Array(NumericArray, Int8Value):
    pass


class Int16Scalar(ScalarExpr, Int16Value):
    pass


class Int16Array(NumericArray, Int16Value):
    pass


class Int32Scalar(ScalarExpr, Int32Value):
    pass


class Int32Array(NumericArray, Int32Value):
    pass


class Int64Scalar(ScalarExpr, Int64Value):
    pass


class Int64Array(NumericArray, Int64Value):
    pass


class FloatScalar(ScalarExpr, FloatValue):
    pass


class FloatArray(NumericArray, FloatValue):
    pass


class DoubleScalar(ScalarExpr, DoubleValue):
    pass


class DoubleArray(NumericArray, DoubleValue):
    pass


class StringScalar(ScalarExpr, StringValue):
    pass


class StringArray(ArrayExpr, StringValue):
    pass


class TimestampScalar(ScalarExpr, TimestampValue):
    pass


class TimestampArray(ArrayExpr, TimestampValue):
    pass


class DecimalScalar(DecimalValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return DecimalScalar(arg, self.meta, name=name)
        return factory


class DecimalArray(DecimalValue, NumericArray):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        ArrayExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return DecimalArray(arg, self.meta, name=name)
        return factory


_scalar_types = {
    'boolean': BooleanScalar,
    'int8': Int8Scalar,
    'int16': Int16Scalar,
    'int32': Int32Scalar,
    'int64': Int64Scalar,
    'float': FloatScalar,
    'double': DoubleScalar,
    'string': StringScalar,
    'timestamp': TimestampScalar
}

_nbytes = {
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8
}


_int_bounds = {
    'int8': (-128, 127),
    'int16': (-32768, 32767),
    'int32': (-2147483648, 2147483647),
    'int64': (-9223372036854775808, 9223372036854775807)
}

_array_types = {
    'boolean': BooleanArray,
    'int8': Int8Array,
    'int16': Int16Array,
    'int32': Int32Array,
    'int64': Int64Array,
    'float': FloatArray,
    'double': DoubleArray,
    'string': StringArray,
    'timestamp': TimestampArray
}



def _highest_precedence_type(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    selector = _TypePrecedence(exprs)
    return selector.get_result()


class _TypePrecedence(object):
    # Impala type precedence (more complex in other database implementations)
    # timestamp
    # double
    # float
    # decimal
    # bigint
    # int
    # smallint
    # tinyint
    # boolean
    # string

    _precedence = ['double', 'float', 'decimal',
                   'int64', 'int32', 'int16', 'int8',
                   'boolean', 'string']

    def __init__(self, exprs):
        self.exprs = exprs

        if len(exprs) == 0:
            raise ValueError('Must pass at least one expression')

        self.type_counts = defaultdict(lambda: 0)
        self._count_types()

    def get_result(self):
        highest_type = self._get_highest_type()
        self._check_casts(highest_type)
        return highest_type

    def _count_types(self):
        for expr in self.exprs:
            self.type_counts[expr._base_type()] += 1

    def _get_highest_type(self):
        for typename in self._precedence:
            if self.type_counts[typename] > 0:
                return typename

    def _check_casts(self, typename):
        for expr in self.exprs:
            if not expr._can_cast_implicit(typename):
                raise ValueError('Expression with type {} cannot be '
                                 'implicitly casted to {}'
                                 .format(expr.type(), typename))


def scalar_type(t):
    if isinstance(t, DataType):
        return t.scalar_ctor()
    else:
        return _scalar_types[t]


def array_type(t):
    if isinstance(t, DataType):
        return t.array_ctor()
    else:
        return _array_types[t]


def literal(value):
    if value is None:
        return null()
    else:
        return Literal(value).to_expr()


def value_list(values):
    return ValueList(values).to_expr()


def _int_literal_class(value, allow_overflow=False):
    if -128 <= value <= 127:
        scalar_type = 'int8'
    elif -32768 <= value <= 32767:
        scalar_type = 'int16'
    elif -2147483648 <= value <= 2147483647:
        scalar_type = 'int32'
    else:
        if value < -9223372036854775808 or value > 9223372036854775807:
            if not allow_overflow:
                raise OverflowError(value)
        scalar_type = 'int64'
    return scalar_type


def _largest_int(int_types):
    nbytes = max(_nbytes[t] for t in int_types)
    return 'int%d' % (8 * nbytes)


_NULL = NullScalar(NullLiteral())


def null():
    return _NULL


def table(schema, name=None):
    if not isinstance(schema, Schema):
        if isinstance(schema, list):
            schema = Schema.from_tuples(schema)
        else:
            schema = Schema.from_dict(schema)

    node = UnboundTable(schema, name=name)
    return TableExpr(node)
