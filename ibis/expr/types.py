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


import re

from ibis.common import RelationError
import ibis.common as com
import ibis.config as config
import ibis.util as util


def _ops():
    import ibis.expr.operations as mod
    return mod


class Parameter(object):

    """
    Placeholder, to be implemented
    """

    pass


#----------------------------------------------------------------------


class Schema(object):

    """
    Holds table schema information
    """

    def __init__(self, names, types):
        from ibis.expr.types import _validate_type
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


class DataType(object):
    pass


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
        if config.options.interactive:
            limit = config.options.sql.default_limit

            try:
                result = self.execute(default_limit=limit)
                return repr(result)
            except com.TranslationError:
                output = ('Translation to backend failed, repr follows:\n%s'
                          % self._repr())
                return output
        else:
            return self._repr()

    def _repr(self):
        from ibis.expr.format import ExprFormatter
        return ExprFormatter(self).get_result()

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, name=name)
        return factory

    def execute(self, default_limit=None):
        """
        If this expression is based on physical tables in a database backend,
        execute it against that backend.

        Returns
        -------
        result : expression-dependent
          Result of compiling expression and executing in backend
        """
        import ibis.expr.analysis as L
        backend = L.find_backend(self)
        return backend.execute(self, default_limit=default_limit)

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


def _safe_repr(x):
    return x._repr() if isinstance(x, Expr) else repr(x)


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
        pprint_args = [x._repr() if isinstance(x, Expr) else repr(x)
                       for x in self.args]
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

    def is_ancestor(self, other):
        if isinstance(other, Expr):
            other = other.op()

        return self.equals(other)

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
            raise TypeError('Must be a value, got: %s' % _safe_repr(expr))

    def _ensure_array(self, expr):
        if not isinstance(expr, ArrayExpr):
            raise TypeError('Must be an array, got: %s' % _safe_repr(expr))

    def _ensure_scalar(self, expr):
        if not isinstance(expr, ScalarExpr):
            raise TypeError('Must be a scalar, got: %s' % _safe_repr(expr))

    def root_tables(self):
        return self.arg._root_tables()

    def output_type(self):
        raise NotImplementedError

    def resolve_name(self):
        raise com.ExpressionError('Expression is not named: %s' % repr(self))


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


class Reduction(ArrayNode):

    def __init__(self, arg):
        self.arg = arg
        ArrayNode.__init__(self, arg)

    def root_tables(self):
        return self.arg._root_tables()


class BlockingTableNode(TableNode):
    # Try to represent the fact that whatever lies here is a semantically
    # distinct table. Like projections, aggregations, and so forth
    pass


def distinct_roots(*args):
    all_roots = []
    for arg in args:
        all_roots.extend(arg._root_tables())
    return util.unique_by_key(all_roots, id)


#----------------------------------------------------------------------
# Helper / factory functions


class ValueExpr(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    _implicit_casts = set()

    def __init__(self, arg, name=None):
        Expr.__init__(self, arg)
        self._name = name

    def equals(self, other):
        if not isinstance(other, ValueExpr):
            return False

        if self._name != other._name:
            return False

        return Expr.equals(self, other)

    def type(self):
        return self._typename

    def _base_type(self):
        # Parametric types like "decimal"
        return self.type()

    def _can_cast_implicit(self, typename):
        from ibis.expr.rules import ImplicitCast
        rule = ImplicitCast(self.type(), self._implicit_casts)
        return rule.can_cast(typename)

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
        return self._factory(self._arg, name=name)

    def between(self, lower, upper):
        """
        Check if the input expr falls between the lower/upper bounds
        passed. Bounds are inclusive. All arguments must be comparable.

        Returns
        -------
        is_between : BooleanValue
        """
        from ibis.expr.operations import as_value_expr, Between
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
        op = _ops().Contains(self, values)
        return op.to_expr()

    def notin(self, values):
        """
        Like isin, but checks whether this expression's value(s) are not
        contained in the passed values. See isin docs for full usage.
        """
        op = _ops().NotContains(self, values)
        return op.to_expr()


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
        op = _ops().DistinctArray(self)
        return op.to_expr()

    def nunique(self):
        """
        Shorthand for foo.distinct().count(); computing the number of unique
        values in an array.
        """
        return _ops().CountDistinct(self).to_expr()

    def topk(self, k, by=None):
        """
        Produces

        Returns
        -------
        topk : TopK filter expression
        """
        op = _ops().TopK(self, k, by=by)
        return op.to_expr()

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
        return _ops().SimpleCaseBuilder(self)

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


class TableExpr(Expr):

    @property
    def _factory(self):
        def factory(arg):
            return TableExpr(arg)
        return factory

    def op(self):
        return self._arg

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator
        ExprValidator([self]).validate_all(exprs)

    def __getitem__(self, what):
        if isinstance(what, basestring):
            return self.get_column(what)
        if isinstance(what, slice):
            step = what.step
            if step is not None and step != 1:
                raise ValueError('Slice step can only be 1')
            start = what.start or 0
            stop = what.stop

            if stop is None or stop < 0:
                raise ValueError('End index must be a positive number')

            if start < 0:
                raise ValueError('Start index must be a positive number')

            return self.limit(stop - start, offset=start)

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
        exprs = util.promote_list(exprs)

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
            op = _ops().MaterializedJoin(self)
            return TableExpr(op)

    def get_columns(self, iterable):
        """
        Get multiple columns from the table

        Examples
        --------
        a, b, c = table.get_columns(['a', 'b', 'c'])

        Returns
        -------
        columns : list of column/array expressions
        """
        return [self.get_column(x) for x in iterable]

    def get_column(self, name):
        """
        Get a reference to a single column from the table

        Returns
        -------
        column : array expression
        """
        ref = _ops().TableColumn(name, self)
        return ref.to_expr()

    @property
    def columns(self):
        return self.schema().names

    def schema(self):
        """
        Get the schema for this table (if one is known)

        Returns
        -------
        schema : Schema
        """
        if not self._is_materialized():
            raise Exception('Table operation is not yet materialized')
        return self.op().get_schema()

    def to_array(self):
        """
        Single column tables can be viewed as arrays.
        """
        op = _ops().TableArrayView(self)
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
        return TableExpr(_ops().SelfReference(self))

    def add_column(self, expr, name=None):
        """
        Add indicated column expression to table, producing a new table. Note:
        this is a shortcut for performing a projection having the same effect.

        Returns
        -------
        modified_table : TableExpr
        """
        if not isinstance(expr, ArrayExpr):
            raise TypeError('Must pass array expression')

        if name is not None:
            expr = expr.name(name)

        return self.projection([self, expr])

    def add_columns(self, what):
        """

        """
        raise NotImplementedError

    def count(self):
        """
        Returns the computed number of rows in the table expression

        Returns
        -------
        count : Int64Scalar
        """
        return _ops().Count(self).to_expr()

    def distinct(self):
        """
        Compute set of unique rows/tuples occurring in this table
        """
        op = _ops().Distinct(self)
        return op.to_expr()

    def projection(self, exprs):
        """
        Compute new table expression with the indicated column expressions from
        this table.

        Parameters
        ----------
        exprs : column expression, or list of column expressions, or strings
          If strings passed, must be columns in the table already

        Returns
        -------
        projection : TableExpr
        """
        import ibis.expr.analysis as L

        if isinstance(exprs, Expr):
            exprs = [exprs]

        exprs = [self._ensure_expr(e) for e in exprs]
        op = L.Projector(self, exprs).get_result()
        return TableExpr(op)

    select = projection

    def _ensure_expr(self, expr):
        if isinstance(expr, basestring):
            expr = self[expr]

        return expr

    def filter(self, predicates):
        """
        Select rows from table based on boolean expressions

        Parameters
        ----------
        predicates : boolean array expressions, or list thereof

        Returns
        -------
        filtered_expr : TableExpr
        """
        import ibis.expr.analysis as L

        if isinstance(predicates, Expr):
            predicates = L.unwrap_ands(predicates)

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
        from ibis.expr.groupby import GroupedTableExpr
        return GroupedTableExpr(self, by)

    def aggregate(self, agg_exprs, by=None, having=None):
        """
        Parameters
        ----------

        Returns
        -------
        agg_expr : TableExpr
        """
        op = _ops().Aggregation(self, agg_exprs, by=by, having=having)
        return TableExpr(op)

    def limit(self, n, offset=0):
        """


        Parameters
        ----------
        n :
        offset :

        Returns
        -------
        limited : TableExpr
        """
        op = _ops().Limit(self, n, offset=offset)
        return TableExpr(op)

    def sort_by(self, sort_exprs):
        """
        Sort table by the indicated column expressions and sort orders
        (ascending/descending)

        Parameters
        ----------
        sort_exprs : sorting expressions
          Must be one of:
            - Column name or expression
            - Sort key, e.g. desc(col)
            - (column name, True (ascending) / False (descending))

        Examples
        --------
        sorted = table.sort_by([('a', True), ('b', False)])

        Returns
        -------
        sorted : TableExpr
        """
        op = _ops().SortBy(self, sort_exprs)
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
        op = _ops().Union(self, other, distinct=distinct)
        return TableExpr(op)


#------------------------------------------------------------------------------
# Declare all typed ValueExprs. This is what the user will actually interact
# with: an instance of each is well-typed and includes all valid methods
# defined for each type.


class AnyValue(ValueExpr):

    _typename = 'any'


class NullValue(AnyValue):

    _typename = 'null'

    def _can_cast_implicit(self, typename):
        return True


class NumericValue(AnyValue):

    def _can_compare(self, other):
        return isinstance(other, NumericValue)


class IntegerValue(NumericValue):
    pass


class BooleanValue(NumericValue):

    _typename = 'boolean'

    def ifelse(self, true_expr, false_expr):
        """
        Shorthand for implementing ternary expressions

        bool_expr.ifelse(0, 1)
        e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END
        """
        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        case = _ops().SearchedCaseBuilder()
        return case.when(self, true_expr).else_(false_expr).end()


class Int8Value(IntegerValue):

    _typename = 'int8'
    _implicit_casts = {'int16', 'int32', 'int64', 'float', 'double',
                       'decimal'}


class Int16Value(IntegerValue):

    _typename = 'int16'
    _implicit_casts = {'int32', 'int64', 'float', 'double', 'decimal'}


class Int32Value(IntegerValue):

    _typename = 'int32'
    _implicit_casts = {'int64', 'float', 'double', 'decimal'}


class Int64Value(IntegerValue):

    _typename = 'int64'
    _implicit_casts = {'float', 'double', 'decimal'}


class FloatingValue(NumericValue):
    pass


class FloatValue(FloatingValue):

    _typename = 'float'
    _implicit_casts = {'double', 'decimal'}


class DoubleValue(FloatingValue):

    _typename = 'double'
    _implicit_casts = {'decimal'}


class StringValue(AnyValue):

    _typename = 'string'

    def _can_compare(self, other):
        return isinstance(other, StringValue)


class DecimalType(DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale):
        self.precision = precision
        self.scale = scale

    def _base_type(self):
        return 'decimal'

    def __repr__(self):
        return ('decimal(precision=%s, scale=%s)'
                % (self.precision, self.scale))

    def __hash__(self):
        return hash((self.precision, self.scale))

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


class DecimalValue(NumericValue):

    _typename = 'decimal'
    _implicit_casts = {'float', 'double'}

    def __init__(self, meta):
        self.meta = meta
        self._precision = meta.precision
        self._scale = meta.scale

    def type(self):
        return DecimalType(self._precision, self._scale)

    def _base_type(self):
        return 'decimal'

    @classmethod
    def _make_constructor(cls, meta):
        def constructor(arg, name=None):
            return cls(arg, meta, name=name)
        return constructor


class TimestampValue(AnyValue):

    _typename = 'timestamp'


class NumericArray(ArrayExpr, NumericValue):
    pass


class NullScalar(NullValue, ScalarExpr):
    pass


class ListExpr(ArrayExpr, AnyValue):
    pass


class BooleanScalar(ScalarExpr, BooleanValue):
    pass


class BooleanArray(NumericArray, BooleanValue):

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


class CategoryType(DataType):

    def __init__(self, cardinality):
        self.cardinality = cardinality

    def _base_type(self):
        return 'category'

    def __repr__(self):
        card = (self.cardinality if self.cardinality is not None
                else 'unknown')
        return ('category(K=%s)' % card)

    def __hash__(self):
        return hash((self.cardinality))

    def __eq__(self, other):
        if not isinstance(other, CategoryType):
            return False

        return self.cardinality == other.cardinality

    def array_ctor(self):
        def constructor(op, name=None):
            return CategoryArray(op, self, name=name)
        return constructor

    def scalar_ctor(self):
        def constructor(op, name=None):
            return CategoryScalar(op, self, name=name)
        return constructor


class CategoryValue(AnyValue):

    """
    Represents some ordered data categorization; tracked as an int32 value
    until explicitly
    """

    _typename = 'category'
    _implicit_casts = Int16Value._implicit_casts

    def __init__(self, meta):
        self.meta = meta

    def type(self):
        return self.meta

    def _base_type(self):
        return 'category'


class CategoryScalar(CategoryValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        CategoryValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return CategoryScalar(arg, self.meta, name=name)
        return factory


class CategoryArray(CategoryValue, ArrayExpr):

    def __init__(self, arg, meta, name=None):
        CategoryValue.__init__(self, meta)
        ArrayExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return CategoryArray(arg, self.meta, name=name)
        return factory


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


_scalar_types = {
    'boolean': BooleanScalar,
    'int8': Int8Scalar,
    'int16': Int16Scalar,
    'int32': Int32Scalar,
    'int64': Int64Scalar,
    'float': FloatScalar,
    'double': DoubleScalar,
    'string': StringScalar,
    'timestamp': TimestampScalar,
    'category': CategoryScalar
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
    'timestamp': TimestampArray,
    'category': CategoryArray
}

#----------------------------------------------------------------------


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


class UnnamedMarker(object):
    pass


unnamed = UnnamedMarker()
