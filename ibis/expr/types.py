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


import os
import sys
import six
import toolz
import itertools
import webbrowser

from ibis.common import IbisError, RelationError
import ibis.common as com
import ibis.config as config
import ibis.util as util
import ibis.expr.datatypes as dt


class Expr(object):

    def _type_display(self):
        return type(self).__name__

    def __init__(self, arg):
        # TODO: all inputs must inherit from a common table API
        self._arg = arg

    def __repr__(self):
        if config.options.interactive:
            try:
                result = self.execute()
                return repr(result)
            except com.TranslationError as e:
                output = ('Translation to backend failed\n'
                          'Error message: {0}\n'
                          'Expression repr follows:\n{1}'
                          .format(e.args[0], self._repr()))
                return output
        else:
            return self._repr()

    def __bool__(self):
        raise ValueError("The truth value of an Ibis expression is not "
                         "defined")

    def __nonzero__(self):
        return self.__bool__()

    def _repr(self, memo=None):
        from ibis.expr.format import ExprFormatter
        return ExprFormatter(self, memo=memo).get_result()

    def _repr_png_(self):
        try:
            import ibis.expr.visualize as viz
        except ImportError:
            return None
        else:
            try:
                return viz.to_graph(self).pipe(format='png')
            except Exception:
                # Something may go wrong, and we can't error in the notebook
                # so fallback to the default text representation.
                return None

    def visualize(self, format='png'):
        """Visualize an expression in the browser as a PNG image.

        Parameters
        ----------
        format : str, optional
            Defaults to ``'png'``. Some additional formats are
            ``'jpeg'`` and ``'svg'``. These are specified by the ``graphviz``
            Python library.

        Notes
        -----
        This method opens a web browser tab showing the image of the expression
        graph created by the code in :module:`ibis.expr.visualize`.

        Raises
        ------
        ImportError
            If ``graphviz`` is not installed.
        """
        import ibis.expr.visualize as viz
        path = viz.draw(viz.to_graph(self), format=format)
        webbrowser.open('file://{}'.format(os.path.abspath(path)))

    def pipe(self, f, *args, **kwargs):
        """Generic composition function to enable expression pipelining.

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
        >>> import ibis
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')], name='t')
        >>> f = lambda a: (a + 1).name('a')
        >>> g = lambda a: (a * 2).name('a')
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1  # doctest: +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: t
          schema:
            a : int64
            b : string
        a = Multiply[int64*]
          left:
            a = Add[int64*]
              left:
                a = Column[int64*] 'a' from table
                  ref_0
              right:
                Literal[int8]
                  1
          right:
            Literal[int8]
              2
        >>> result2 = g(f(t.a))  # equivalent to the above
        >>> result1.equals(result2)
        True

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

    def execute(self, limit='default', async=False, params=None):
        """
        If this expression is based on physical tables in a database backend,
        execute it against that backend.

        Parameters
        ----------
        limit : integer or None, default 'default'
          Pass an integer to effect a specific row limit. limit=None means "no
          limit". The default is whatever is in ibis.options.

        Returns
        -------
        result : expression-dependent
          Result of compiling expression and executing in backend
        """
        from ibis.client import execute
        return execute(self, limit=limit, async=async, params=params)

    def compile(self, limit=None, params=None):
        """
        Compile expression to whatever execution target, to verify

        Returns
        -------
        compiled : value or list
           query representation or list thereof
        """
        from ibis.client import compile
        return compile(self, limit=limit, params=params)

    def verify(self):
        """
        Returns True if expression can be compiled to its attached client
        """
        try:
            self.compile()
        except Exception:
            return False
        else:
            return True

    def equals(self, other, cache=None):
        if type(self) != type(other):
            return False
        return self._arg.equals(other._arg, cache=cache)

    def _root_tables(self):
        return self.op().root_tables()

    def _get_unbound_tables(self):
        # The expression graph may contain one or more tables of a particular
        # known schema
        pass


if sys.version_info.major == 2:
    # Python 2.7 doesn't return NotImplemented unless the other operand has
    # an attribute called "timetuple". This is a bug that's fixed in Python 3
    Expr.timetuple = None


def _safe_repr(x, memo=None):
    return x._repr(memo=memo) if isinstance(x, (Expr, Node)) else repr(x)


class OperationMeta(type):

    def __new__(cls, name, parents, dct):
        if 'input_type' in dct:
            from ibis.expr.rules import TypeSignature, signature
            sig = dct['input_type']
            if not isinstance(sig, TypeSignature):
                dct['input_type'] = sig = signature(sig)

                for i, t in enumerate(sig.types):
                    if t.name is None:
                        continue

                    if t.name not in dct:
                        dct[t.name] = _arg_getter(i, doc=t.doc)

        return super(OperationMeta, cls).__new__(cls, name, parents, dct)


class Node(six.with_metaclass(OperationMeta, object)):

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

    def __init__(self, args=None):
        args = args or []
        self.args = self._validate_args(args)

    def _validate_args(self, args):
        if not hasattr(self, 'input_type'):
            return args

        return self.input_type.validate(args)

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
            if isinstance(arg, (tuple, list)):
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
        if isinstance(other, Expr):
            other = other.op()

        return self.equals(other)

    _expr_cached = None

    def to_expr(self):
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
    def _arg_names(self):
        try:
            input_type = self.__class__.input_type
        except AttributeError:
            return []
        else:
            return [t.name for t in getattr(input_type, 'types', [])]


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


def _arg_getter(i, doc=None):
    def arg_accessor(self):
        return self.args[i]
    return property(arg_accessor, doc=doc)


class ValueOp(Node):

    def __init__(self, *args):
        super(ValueOp, self).__init__(args)

    def root_tables(self):
        exprs = [arg for arg in self.args if isinstance(arg, Expr)]
        return distinct_roots(*exprs)

    def resolve_name(self):
        raise com.ExpressionError('Expression is not named: %s' % repr(self))

    def has_resolved_name(self):
        return False


class TableColumn(ValueOp):

    """
    Selects a column from a TableExpr
    """

    def __init__(self, name, table_expr):
        schema = table_expr.schema()
        if isinstance(name, six.integer_types):
            name = schema.name_at_position(name)

        super(TableColumn, self).__init__(name, table_expr)
        if name not in schema:
            raise com.IbisTypeError(
                "'{0}' is not a field in {1}".format(name, table_expr.columns)
            )

        self.name = name
        self.table = table_expr

    def parent(self):
        return self.table

    def resolve_name(self):
        return self.name

    def has_resolved_name(self):
        return True

    def root_tables(self):
        return self.table._root_tables()

    def _make_expr(self):
        ctype = self.table._get_type(self.name)
        klass = ctype.array_type()
        return klass(self, name=self.name)


class ExpressionList(Node):

    def __init__(self, exprs):
        exprs = [as_value_expr(x) for x in exprs]
        Node.__init__(self, exprs)

    def root_tables(self):
        return distinct_roots(*self.args)

    def output_type(self):
        return ExprList


class ExprList(Expr):

    def _type_display(self):
        list_args = [arg._type_display()
                     for arg in self.op().args]
        return ', '.join(list_args)

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


def infer_literal_type(value):
    # TODO: depricate?
    if value is null:
        return dt.null

    return dt.infer(value)


class Literal(ValueOp):

    def __init__(self, value, type=None):
        super(Literal, self).__init__(value, type)
        self.value = value
        self._output_type = type.scalar_type()

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
        return self._output_type

    def root_tables(self):
        return []


_parameter_counter = itertools.count()


def _parameter_name():
    return 'param[{:d}]'.format(next(_parameter_counter))


class ScalarParameter(ValueOp):

    def __init__(self, type, name=None):
        super(ScalarParameter, self).__init__(type)
        self.name = name
        self.output_type = type.scalar_type

    def __repr__(self):
        return '{}(name={!r}, type={})'.format(
            type(self).__name__, self.name, self.type
        )

    @property
    def type(self):
        return self.args[0]

    def equals(self, other, cache=None):
        return (
            isinstance(other, ScalarParameter) and
            self.name == other.name and
            self.type.equals(other.type, cache=cache)
        )

    def root_tables(self):
        return []

    def resolve_name(self):
        return self.name


def param(type, name=None):
    """Create a parameter of a particular type to be defined just before
    execution.

    Parameters
    ----------
    type : dt.DataType
        The type of the unbound parameter, e.g., double, int64, date, etc.
    name : str, optional
        The name of the parameter

    Returns
    -------
    ScalarExpr

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.datatypes as dt
    >>> start = ibis.param(dt.date)
    >>> end = ibis.param(dt.date)
    >>> schema = [('timestamp_col', 'timestamp'), ('value', 'double')]
    >>> t = ibis.table(schema)
    >>> predicates = [t.timestamp_col >= start, t.timestamp_col <= end]
    >>> expr = t.filter(predicates).value.sum()
    """
    if name is None:
        name = _parameter_name()
    expr = ScalarParameter(dt.validate_type(type), name=name).to_expr()
    return expr.name(name)


def distinct_roots(*expressions):
    roots = toolz.concat(
        expression._root_tables() for expression in expressions
    )
    return list(toolz.unique(roots, key=id))


# ---------------------------------------------------------------------
# Helper / factory functions


class ValueExpr(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    def __init__(self, arg, name=None):
        super(ValueExpr, self).__init__(arg)
        self._name = name

    def equals(self, other, cache=None):
        return (
            isinstance(other, ValueExpr) and
            self._name == other._name and
            super(ValueExpr, self).equals(other, cache=cache)
        )

    def type(self):
        raise NotImplementedError(
            'Expressions of type {0} must implement a type method'.format(
                type(self).__name__
            )
        )

    def has_name(self):
        if self._name is not None:
            return True
        return self.op().has_resolved_name()

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

    def _type_display(self):
        return str(self.type())


class ColumnExpr(ValueExpr):

    def _type_display(self):
        return '{}*'.format(self.type())

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

    def _type_display(self):
        return str(self.type())

    def type(self):
        return 'analytic'


class TableExpr(Expr):

    @property
    def _factory(self):
        def factory(arg):
            return TableExpr(arg)
        return factory

    def _type_display(self):
        return 'table'

    def _is_valid(self, exprs):
        try:
            self._assert_valid(util.promote_list(exprs))
        except com.RelationError:
            return False
        else:
            return True

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator
        ExprValidator([self]).validate_all(exprs)

    def __contains__(self, name):
        return name in self.schema()

    def __getitem__(self, what):
        if isinstance(what, six.string_types + six.integer_types):
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

        if isinstance(what, (list, tuple, TableExpr)):
            # Projection case
            return self.projection(what)
        elif isinstance(what, BooleanColumn):
            # Boolean predicate
            return self.filter([what])
        elif isinstance(what, ColumnExpr):
            # Projection convenience
            return self.projection(what)
        else:
            raise NotImplementedError(
                'Selection rows or columns with {} objects is not '
                'supported'.format(type(what).__name__)
            )

    def __len__(self):
        raise com.ExpressionError('Use .count() instead')

    def __setstate__(self, instance_dictionary):
        self.__dict__ = instance_dictionary

    def __getattr__(self, key):
        try:
            schema = self.schema()
        except com.IbisError:
            raise AttributeError(key)

        if key not in schema:
            raise AttributeError(key)

        try:
            return self.get_column(key)
        except com.IbisTypeError:
            raise AttributeError(key)

    def __dir__(self):
        attrs = dir(type(self))
        if self._is_materialized():
            attrs = frozenset(attrs + self.schema().names)
        return sorted(attrs)

    def _resolve(self, exprs):
        exprs = util.promote_list(exprs)

        # Stash this helper method here for now
        out_exprs = []
        for expr in exprs:
            expr = self._ensure_expr(expr)
            out_exprs.append(expr)
        return out_exprs

    def _ensure_expr(self, expr):
        if isinstance(expr, six.string_types):
            return self[expr]
        elif isinstance(expr, six.integer_types):
            return self[self.schema().name_at_position(expr)]
        elif not isinstance(expr, Expr):
            return expr(self)
        else:
            return expr

    def _get_type(self, name):
        return self._arg.get_type(name)

    def get_columns(self, iterable):
        """
        Get multiple columns from the table

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(
        ...    [
        ...        ('a', 'int64'),
        ...        ('b', 'string'),
        ...        ('c', 'timestamp'),
        ...        ('d', 'float'),
        ...    ],
        ...    name='t'
        ... )
        >>> a, b, c = table.get_columns(['a', 'b', 'c'])

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
        ref = TableColumn(name, self)
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
        return self.op().schema

    def _is_materialized(self):
        # The operation produces a known schema
        return self.op().has_schema()

    def add_column(self, expr, name=None):
        """
        Add indicated column expression to table, producing a new table. Note:
        this is a shortcut for performing a projection having the same effect.

        Returns
        -------
        modified_table : TableExpr
        """
        expr = self._ensure_expr(expr)

        if not isinstance(expr, ColumnExpr):
            raise com.InputTypeError('Must pass array expression')

        if name is not None:
            expr = expr.name(name)

        return self.projection([self, expr])

    def group_by(self, by=None, **additional_grouping_expressions):
        """
        Create an intermediate grouped table expression, pending some group
        operation to be applied with it.

        Examples
        --------
        >>> import ibis
        >>> pairs = [('a', 'int32'), ('b', 'timestamp'), ('c', 'double')]
        >>> t = ibis.table(pairs)
        >>> b1, b2 = t.a, t.b
        >>> result = t.group_by([b1, b2]).aggregate(sum_of_c=t.c.sum())

        Notes
        -----
        group_by and groupby are equivalent, with `groupby` being provided for
        ease-of-use for pandas users.

        Returns
        -------
        grouped_expr : GroupedTableExpr
        """
        from ibis.expr.groupby import GroupedTableExpr
        return GroupedTableExpr(self, by, **additional_grouping_expressions)

    groupby = group_by


# -----------------------------------------------------------------------------
# Declare all typed ValueExprs. This is what the user will actually interact
# with: an instance of each is well-typed and includes all valid methods
# defined for each type.


class AnyValue(ValueExpr):

    def type(self):
        return dt.any


class NullValue(AnyValue):

    def type(self):
        return dt.null


class NumericValue(AnyValue):
    pass


class IntegerValue(NumericValue):
    pass


class BooleanValue(NumericValue):

    def type(self):
        return dt.boolean


class Int8Value(IntegerValue):

    def type(self):
        return dt.int8


class Int16Value(IntegerValue):

    def type(self):
        return dt.int16


class Int32Value(IntegerValue):

    def type(self):
        return dt.int32


class Int64Value(IntegerValue):

    def type(self):
        return dt.int64


class UInt8Value(IntegerValue):

    def type(self):
        return dt.uint8


class UInt16Value(IntegerValue):

    def type(self):
        return dt.uint16


class UInt32Value(IntegerValue):

    def type(self):
        return dt.uint32


class UInt64Value(IntegerValue):

    def type(self):
        return dt.uint64


class HalffloatValue(NumericValue):

    def type(self):
        return dt.halffloat


class FloatingValue(NumericValue):
    pass


class FloatValue(FloatingValue):

    def type(self):
        return dt.float


class DoubleValue(FloatingValue):

    def type(self):
        return dt.double


class StringValue(AnyValue):

    def type(self):
        return dt.string


class BinaryValue(AnyValue):

    def type(self):
        return dt.binary


class ParameterizedValue(AnyValue):

    def __init__(self, meta, name=None):
        super(ParameterizedValue, self).__init__(meta, name=name)
        self.meta = meta

    def type(self):
        return self.meta

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, self.meta, name=name)
        return factory


class DecimalValue(ParameterizedValue, NumericValue):
    pass


class TemporalValue(AnyValue):
    pass


class DateValue(TemporalValue):

    def type(self):
        return dt.date


class TimeValue(TemporalValue):

    def type(self):
        return dt.time


class TimestampValue(TemporalValue):

    def __init__(self, meta=None):
        self.meta = meta
        self._timezone = getattr(meta, 'timezone', None)

    def type(self):
        return dt.Timestamp(timezone=self._timezone)


class IntervalValue(ParameterizedValue):

    def __init__(self, meta=None):
        self.meta = meta

    @property
    def unit(self):
        return self.meta.unit

    @unit.setter
    def unit(self, unit):
        self.meta.unit = unit

    @property
    def resolution(self):
        """Unit's name"""
        return self.meta.resolution


class ArrayValue(ParameterizedValue):
    pass


class MapValue(ParameterizedValue):
    pass


class StructValue(ParameterizedValue):

    def __dir__(self):
        return sorted(frozenset(
            itertools.chain(dir(type(self)), self.type().names)
        ))


class NumericColumn(ColumnExpr, NumericValue):
    pass


class NullScalar(NullValue, ScalarExpr):
    """
    A scalar value expression representing NULL
    """
    pass


class NullColumn(ColumnExpr, NullValue):
    pass


class BooleanScalar(ScalarExpr, BooleanValue):
    pass


class BooleanColumn(NumericColumn, BooleanValue):
    pass


class Int8Scalar(ScalarExpr, Int8Value):
    pass


class Int8Column(NumericColumn, Int8Value):
    pass


class Int16Scalar(ScalarExpr, Int16Value):
    pass


class Int16Column(NumericColumn, Int16Value):
    pass


class Int32Scalar(ScalarExpr, Int32Value):
    pass


class Int32Column(NumericColumn, Int32Value):
    pass


class Int64Scalar(ScalarExpr, Int64Value):
    pass


class Int64Column(NumericColumn, Int64Value):
    pass


class UInt8Scalar(ScalarExpr, UInt8Value):
    pass


class UInt8Column(NumericColumn, UInt8Value):
    pass


class UInt16Scalar(ScalarExpr, UInt16Value):
    pass


class UInt16Column(NumericColumn, UInt16Value):
    pass


class UInt32Scalar(ScalarExpr, UInt32Value):
    pass


class UInt32Column(NumericColumn, UInt32Value):
    pass


class UInt64Scalar(ScalarExpr, UInt64Value):
    pass


class UInt64Column(NumericColumn, UInt64Value):
    pass


class HalffloatScalar(ScalarExpr, HalffloatValue):
    pass


class HalffloatColumn(NumericColumn, HalffloatValue):
    pass


class FloatScalar(ScalarExpr, FloatValue):
    pass


class FloatColumn(NumericColumn, FloatValue):
    pass


class DoubleScalar(ScalarExpr, DoubleValue):
    pass


class DoubleColumn(NumericColumn, DoubleValue):
    pass


class StringScalar(ScalarExpr, StringValue):
    pass


class StringColumn(ColumnExpr, StringValue):
    pass


class BinaryScalar(ScalarExpr, BinaryValue):
    pass


class BinaryColumn(ColumnExpr, BinaryValue):
    pass


class DateScalar(ScalarExpr, DateValue):
    pass


class DateColumn(ColumnExpr, DateValue):
    pass


class TimeScalar(ScalarExpr, TimeValue):
    pass


class TimeColumn(ColumnExpr, TimeValue):
    pass


class TimestampScalar(ScalarExpr, TimestampValue):

    def __init__(self, arg, meta=None, name=None):
        ScalarExpr.__init__(self, arg, name=name)
        TimestampValue.__init__(self, meta=meta)


class TimestampColumn(ColumnExpr, TimestampValue):

    def __init__(self, arg, meta=None, name=None):
        ColumnExpr.__init__(self, arg, name=name)
        TimestampValue.__init__(self, meta=meta)


class IntervalScalar(ScalarExpr, IntervalValue):

    def __init__(self, arg, meta=None, name=None):
        ScalarExpr.__init__(self, arg, name=name)
        IntervalValue.__init__(self, meta=meta)


class IntervalColumn(ColumnExpr, IntervalValue):

    def __init__(self, arg, meta=None, name=None):
        ColumnExpr.__init__(self, arg, name=name)
        IntervalValue.__init__(self, meta=meta)


class DecimalScalar(DecimalValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)


class DecimalColumn(DecimalValue, NumericColumn):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        NumericColumn.__init__(self, arg, name=name)


class CategoryValue(ParameterizedValue):
    """
    Represents some ordered data categorization; tracked as an int32 value
    until explicitly
    """


class CategoryScalar(CategoryValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        CategoryValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)


class CategoryColumn(CategoryValue, ColumnExpr):

    def __init__(self, arg, meta, name=None):
        CategoryValue.__init__(self, meta)
        ColumnExpr.__init__(self, arg, name=name)


class ArrayScalar(ArrayValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        ArrayValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)


class ArrayColumn(ArrayValue, ColumnExpr):

    def __init__(self, arg, meta, name=None):
        ArrayValue.__init__(self, meta)
        ColumnExpr.__init__(self, arg, name=name)


class MapScalar(MapValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        MapValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)


class MapColumn(MapValue, ColumnExpr):

    def __init__(self, arg, meta, name=None):
        MapValue.__init__(self, meta)
        ColumnExpr.__init__(self, arg, name=name)


class StructScalar(StructValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        StructValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)


class StructColumn(StructValue, ColumnExpr):

    def __init__(self, arg, meta, name=None):
        StructValue.__init__(self, meta)
        ColumnExpr.__init__(self, arg, name=name)


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


def castable(source, target):
    """Return whether source ir type is castable to target

    Based on the underlying datatypes and the value in case of Literals
    """
    op = source.op()
    value = op.value if isinstance(op, Literal) else None
    return dt.castable(source.type(), target.type(), value=value)


def cast(source, target):
    """Currently Literal to *Scalar implicit casts are allowed"""

    if not castable(source, target):
        raise com.IbisTypeError('Source is not castable to target type!')

    # currently it prevents column -> scalar implicit castings
    # however the datatypes are matching
    op = source.op()
    if not isinstance(op, Literal):
        raise com.IbisTypeError('Only able to implicitly cast literals!')

    out_type = target.type().scalar_type()
    return out_type(op)


def literal(value, type=None):
    """Create a scalar expression from a Python value.

    Parameters
    ----------
    value : some Python basic type
        A Python value
    type : ibis type or string, optional
        An instance of :class:`ibis.expr.datatypes.DataType` or a string
        indicating the ibis type of `value`. This parameter should only be used
        in cases where ibis's type inference isn't sufficient for discovering
        the type of `value`.

    Returns
    -------
    literal_value : Literal
        An expression representing a literal value

    Examples
    --------
    >>> import ibis
    >>> x = ibis.literal(42)
    >>> x.type()
    int8
    >>> y = ibis.literal(42, type='double')
    >>> y.type()
    double
    >>> ibis.literal('foobar', type='int64')  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: Value 'foobar' cannot be safely coerced to int64
    """
    if hasattr(value, 'op') and isinstance(value.op(), Literal):
        return value

    dtype = infer_literal_type(value)

    if type is not None:
        try:
            # check that dtype is implicitly castable to explicitly given dtype
            dtype = dtype.cast(type)
        except com.IbisTypeError:
            raise TypeError('Value {!r} cannot be safely coerced '
                            'to {}'.format(value, type))

    if value is None or value is _NULL or value is null:
        return null().cast(dtype)
    else:
        return Literal(value, type=dtype).to_expr()


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


class NullLiteral(ValueOp):

    """
    Typeless NULL literal
    """

    def __init__(self):
        self.value = None

    @property
    def args(self):
        return [self.value]

    def equals(self, other, cache=None):
        return isinstance(other, NullLiteral)

    def output_type(self):
        return NullScalar

    def root_tables(self):
        return []


class ListExpr(ColumnExpr, AnyValue):

    @property
    def values(self):
        return self.op().values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __add__(self, other):
        other_values = getattr(other, 'values', other)
        return type(self.op())(self.values + other_values).to_expr()

    def __radd__(self, other):
        other_values = getattr(other, 'values', other)
        return type(self.op())(other_values + self.values).to_expr()

    def __bool__(self):
        return bool(self.values)

    def __len__(self):
        return len(self.values)


class SortExpr(Expr):

    def _type_display(self):
        return 'array-sort'


class ValueList(ValueOp):

    """
    Data structure for a list of value expressions
    """

    def __init__(self, args):
        self.values = list(map(as_value_expr, args))
        super(ValueList, self).__init__(self.values)

    def root_tables(self):
        return distinct_roots(*self.values)

    def _make_expr(self):
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
