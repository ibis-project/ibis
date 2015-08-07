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

import six

import datetime
import re

from ibis.common import IbisError, RelationError
import ibis.common as com
import ibis.config as config
import ibis.util as util

if six.PY3:
    from io import StringIO
else:
    from io import BytesIO as StringIO


def _ops():
    import ibis.expr.operations as mod
    return mod


class Parameter(object):

    """
    Placeholder, to be implemented
    """

    pass


# ---------------------------------------------------------------------


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
        buf = StringIO()
        space = 2 + max(len(x) for x in self.names)
        for name, tipo in zip(self.names, self.types):
            buf.write('\n{0}{1}'.format(name.ljust(space), str(tipo)))

        return "ibis.Schema {{{0}\n}}".format(util.indent(buf.getvalue(), 2))

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

    def __eq__(self, other):
        return self.equals(other)

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


# ---------------------------------------------------------------------


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
                result = self.execute(limit=limit)
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

    def pipe(self, f, *args, **kwargs):
        """
        Generic composition function to enable expression pipelining

        >>> (expr
             .pipe(f, *args, **kwargs)
             .pipe(g, *args2, **kwargs2))

        is equivalent to

        >>> g(f(expr, *args, **kwargs), *args2, **kwargs2)

        Parameters
        ----------
        f : function or (function, arg_name) tuple
          If the expression needs to be passed as anything other than the first
          argument to the function, pass a tuple with the argument name. For
          example, (f, 'data') if the function f expects a 'data' keyword
        args : positional arguments
        kwargs : keyword arguments

        Examples
        --------
        >>> def foo(data, a=None, b=None):
                pass
        >>> def bar(a, b, data=None):
                pass
        >>> expr.pipe(foo, a=5, b=10)
        >>> expr.pipe((bar, 'data'), 1, 2)

        Returns
        -------
        result : result type of passed function
        """
        if isinstance(f, tuple):
            f, data_keyword = f
            kwargs = kwargs.copy()
            kwargs[data_keyword] = self
            return f(*args, **kwargs)
        else:
            return f(self, *args, **kwargs)

    __call__ = pipe

    def op(self):
        return self._arg

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, name=name)
        return factory

    def _can_implicit_cast(self, arg):
        return False

    def execute(self, limit=None):
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
        return backend.execute(self, limit=limit)

    def equals(self, other):
        if type(self) != type(other):
            return False
        return self._arg.equals(other._arg)

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
        pprint_args = []

        _pp = _safe_repr

        for x in self.args:
            if isinstance(x, (tuple, list)):
                pp = repr([_pp(y) for y in x])
            else:
                pp = _pp(x)
            pprint_args.append(pp)

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

        for left, right in zip(self.args, other.args):
            if not all_equal(left, right):
                return False
        return True

    def is_ancestor(self, other):
        if isinstance(other, Expr):
            other = other.op()

        return self.equals(other)

    def to_expr(self):
        klass = self.output_type()
        return klass(self)

    def output_type(self):
        """
        This function must resolve the output type of the expression and return
        the node wrapped in the appropriate ValueExpr type.
        """
        raise NotImplementedError


def all_equal(left, right):
    if isinstance(left, list):
        if not isinstance(right, list):
            return False
        for a, b in zip(left, right):
            if not all_equal(a, b):
                return False
        return True

    if hasattr(left, 'equals'):
        return left.equals(right)
    else:
        return left == right
    return True


class ValueNode(Node):

    def __init__(self, *args):
        args = self._validate_args(args)
        Node.__init__(self, args)

    def _validate_args(self, args):
        if not hasattr(self, 'input_type'):
            return args

        return self.input_type.validate(args)

    def root_tables(self):
        exprs = [arg for arg in self.args if isinstance(arg, Expr)]
        return distinct_roots(*exprs)

    def resolve_name(self):
        raise com.ExpressionError('Expression is not named: %s' % repr(self))


class ExpressionList(Node):

    def __init__(self, exprs):
        exprs = [as_value_expr(x) for x in exprs]
        Node.__init__(self, exprs)

    def root_tables(self):
        return distinct_roots(*self.args)

    def output_type(self):
        return ExprList


class ExprList(Expr):

    def exprs(self):
        return self.op().args

    def names(self):
        return [x.get_name() for x in self.exprs()]

    def rename(self, f):
        new_exprs = [x.name(f(x.get_name())) for x in self.exprs()]
        return ExpressionList(new_exprs).to_expr()

    def prefix(self, value):
        return self.rename(lambda x: value + x)

    def suffix(self, value):
        return self.rename(lambda x: x + value)

    def concat(self, *others):
        """
        Concatenate expression lists

        Returns
        -------
        combined : ExprList
        """
        exprs = list(self.exprs())
        for o in others:
            if not isinstance(o, ExprList):
                raise TypeError(o)
            exprs.extend(o.exprs())
        return ExpressionList(exprs).to_expr()


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
        return (type(self.value) == type(other.value) and
                self.value == other.value)

    def output_type(self):
        import ibis.expr.rules as rules
        if isinstance(self.value, bool):
            klass = BooleanScalar
        elif isinstance(self.value, (int, long)):
            int_type = rules.int_literal_class(self.value)
            klass = scalar_type(int_type)
        elif isinstance(self.value, float):
            klass = DoubleScalar
        elif isinstance(self.value, basestring):
            klass = StringScalar
        elif isinstance(self.value, datetime.datetime):
            klass = TimestampScalar
        else:
            raise com.InputTypeError(self.value)

        return klass

    def root_tables(self):
        return []


class TableNode(Node):

    def get_type(self, name):
        return self.get_schema().get_type(name)

    def to_expr(self):
        return TableExpr(self)


class BlockingTableNode(TableNode):
    # Try to represent the fact that whatever lies here is a semantically
    # distinct table. Like projections, aggregations, and so forth
    pass


def distinct_roots(*args):
    all_roots = []
    for arg in args:
        all_roots.extend(arg._root_tables())
    return util.unique_by_key(all_roots, id)


# ---------------------------------------------------------------------
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

    def get_name(self):
        if self._name is not None:
            # This value has been explicitly named
            return self._name

        # In some but not all cases we can get a name from the node that
        # produces the value
        return self.op().resolve_name()

    def name(self, name):
        return self._factory(self._arg, name=name)


class ScalarExpr(ValueExpr):

    pass


class ArrayExpr(ValueExpr):

    def parent(self):
        return self._arg

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


class AnalyticExpr(Expr):

    @property
    def _factory(self):
        def factory(arg):
            return type(self)(arg)
        return factory

    def type(self):
        return 'analytic'


class TableExpr(Expr):

    @property
    def _factory(self):
        def factory(arg):
            return TableExpr(arg)
        return factory

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator
        ExprValidator([self]).validate_all(exprs)

    def __contains__(self, name):
        return name in self.schema()

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

        what = bind_expr(self, what)

        if isinstance(what, AnalyticExpr):
            what = what._table_getitem()

        if isinstance(what, (list, tuple)):
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

    def _ensure_expr(self, expr):
        if isinstance(expr, basestring):
            return self[expr]
        elif not isinstance(expr, Expr):
            return expr(self)
        else:
            return expr

    def _get_type(self, name):
        return self._arg.get_type(name)

    def materialize(self):
        """
        Force schema resolution for a joined table, selecting all fields from
        all tables.
        """
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
            raise IbisError('Table operation is not yet materialized')
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
        expr = self._ensure_expr(expr)

        if not isinstance(expr, ArrayExpr):
            raise com.InputTypeError('Must pass array expression')

        if name is not None:
            expr = expr.name(name)

        return self.projection([self, expr])

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
        exprs : column expression, or string, or list of column expressions and
          strings. If strings passed, must be columns in the table already

        Returns
        -------
        projection : TableExpr
        """
        import ibis.expr.analysis as L

        if isinstance(exprs, (Expr, basestring)):
            exprs = [exprs]

        exprs = [self._ensure_expr(e) for e in exprs]
        op = L.Projector(self, exprs).get_result()
        return TableExpr(op)

    select = projection

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
        Aggregate a table with a given set of reductions, with grouping
        expressions, and post-aggregation filters.

        Parameters
        ----------
        agg_exprs : expression or expression list
        by : optional, default None
          Grouping expressions
        having : optional, default None
          Post-aggregation filters

        Returns
        -------
        agg_expr : TableExpr
        """
        op = _ops().Aggregation(self, agg_exprs, by=by, having=having)
        return TableExpr(op)

    def limit(self, n, offset=0):
        """
        Select the first n rows at beginning of table (may not be deterministic
        depending on implementatino and presence of a sorting).

        Parameters
        ----------
        n : int
          Rows to include
        offset : int, default 0
          Number of rows to skip first

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


# -----------------------------------------------------------------------------
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


class Int8Value(IntegerValue):

    _typename = 'int8'
    _implicit_casts = set(['int16', 'int32', 'int64', 'float', 'double',
                           'decimal'])


class Int16Value(IntegerValue):

    _typename = 'int16'
    _implicit_casts = set(['int32', 'int64', 'float', 'double', 'decimal'])


class Int32Value(IntegerValue):

    _typename = 'int32'
    _implicit_casts = set(['int64', 'float', 'double', 'decimal'])


class Int64Value(IntegerValue):

    _typename = 'int64'
    _implicit_casts = set(['float', 'double', 'decimal'])


class FloatingValue(NumericValue):
    pass


class FloatValue(FloatingValue):

    _typename = 'float'
    _implicit_casts = set(['double', 'decimal'])


class DoubleValue(FloatingValue):

    _typename = 'double'
    _implicit_casts = set(['decimal'])


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

    def __ne__(self, other):
        return not self.__eq__(other)

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
    _implicit_casts = set(['float', 'double'])

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

    def _can_implicit_cast(self, arg):
        op = arg.op()
        if isinstance(op, Literal):
            try:
                import pandas as pd
                pd.Timestamp(op.value)
                return True
            except ValueError:
                return False
        return False

    def _can_compare(self, other):
        return isinstance(other, TimestampValue)

    def _implicit_cast(self, arg):
        # assume we've checked this is OK at this point...
        op = arg.op()
        return TimestampScalar(op)


class NumericArray(ArrayExpr, NumericValue):
    pass


class NullScalar(NullValue, ScalarExpr):
    """
    A scalar value expression representing NULL
    """
    pass


class BooleanScalar(ScalarExpr, BooleanValue):
    pass


class BooleanArray(NumericArray, BooleanValue):
    pass


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

    def __init__(self, cardinality=None):
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

    def to_integer_type(self):
        if self.cardinality is None:
            return 'int64'
        elif self.cardinality < (2 ** 7 - 1):
            return 'int8'
        elif self.cardinality < (2 ** 15 - 1):
            return 'int16'
        elif self.cardinality < (2 ** 31 - 1):
            return 'int32'
        else:
            return 'int64'

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

    def _can_compare(self, other):
        return isinstance(other, IntegerValue)


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

# ---------------------------------------------------------------------


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


def as_value_expr(val):
    import pandas as pd
    if not isinstance(val, Expr):
        if isinstance(val, (tuple, list)):
            val = sequence(val)
        elif isinstance(val, pd.Series):
            val = sequence(list(val))
        else:
            val = literal(val)

    return val


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
        return Literal(value).to_expr()


_NULL = None


def null():
    """
    Create a NULL/NA scalar
    """
    global _NULL
    if _NULL is None:
        _NULL = NullScalar(NullLiteral())

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
        pass

    @property
    def args(self):
        return [None]

    def equals(self, other):
        return isinstance(other, NullLiteral)

    def output_type(self):
        return NullScalar

    def root_tables(self):
        return []


class ListExpr(ArrayExpr, AnyValue):
    pass


class ValueList(ValueNode):

    """
    Data structure for a list of value expressions
    """

    def __init__(self, args):
        self.values = [as_value_expr(x) for x in args]
        ValueNode.__init__(self, [self.values])

    def root_tables(self):
        return distinct_roots(*self.values)

    def to_expr(self):
        return ListExpr(self)


def bind_expr(table, expr):
    if isinstance(expr, (list, tuple)):
        return [bind_expr(table, x) for x in expr]

    return table._ensure_expr(expr)


def find_base_table(expr):
    if isinstance(expr, TableExpr):
        return expr

    for arg in expr.op().flat_args():
        if isinstance(arg, Expr):
            r = find_base_table(arg)
            if isinstance(r, TableExpr):
                return r
