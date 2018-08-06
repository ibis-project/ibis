import os
import six
import sys
import itertools
import webbrowser

import ibis
import ibis.util as util
import ibis.common as com
import ibis.config as config


# TODO move methods containing ops import to api.py

class Expr(object):
    """Base expression class"""

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
        if not ibis.options.graphviz_repr:
            return None
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

    def visualize(self, format='svg'):
        """Visualize an expression in the browser as an SVG image.

        Parameters
        ----------
        format : str, optional
            Defaults to ``'svg'``. Some additional formats are
            ``'jpeg'`` and ``'png'``. These are specified by the ``graphviz``
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

    def execute(self, limit='default', params=None, **kwargs):
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
        return execute(self, limit=limit, params=params, **kwargs)

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


if sys.version_info.major == 2:
    # Python 2.7 doesn't return NotImplemented unless the other operand has
    # an attribute called "timetuple". This is a bug that's fixed in Python 3
    Expr.timetuple = None


class ExprList(Expr):

    def _type_display(self):
        return ', '.join(expr._type_display() for expr in self.exprs())

    def exprs(self):
        return self.op().exprs

    def names(self):
        return [x.get_name() for x in self.exprs()]

    def types(self):
        return [x.type() for x in self.exprs()]

    def schema(self):
        import ibis.expr.schema as sch
        return sch.Schema(self.names(), self.types())

    def rename(self, f):
        import ibis.expr.operations as ops
        new_exprs = [x.name(f(x.get_name())) for x in self.exprs()]
        return ops.ExpressionList(new_exprs).to_expr()

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
        import ibis.expr.operations as ops
        exprs = list(self.exprs())
        for o in others:
            if not isinstance(o, ExprList):
                raise TypeError(o)
            exprs.extend(o.exprs())
        return ops.ExpressionList(exprs).to_expr()


# ---------------------------------------------------------------------
# Helper / factory functions


class ValueExpr(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    def __init__(self, arg, dtype, name=None):
        super(ValueExpr, self).__init__(arg)
        self._name = name
        self._dtype = dtype

    def equals(self, other, cache=None):
        return (
            isinstance(other, ValueExpr) and
            self._name == other._name and
            self._dtype == other._dtype and
            super(ValueExpr, self).equals(other, cache=cache)
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

    def type(self):
        return self._dtype

    @property
    def _factory(self):
        def factory(arg, name=None):
            return type(self)(arg, dtype=self.type(), name=name)
        return factory


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
            raise com.RelationError('Cannot convert array expression '
                                    'involving multiple base table references '
                                    'to a projection')

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
        import ibis.expr.operations as ops
        ref = ops.TableColumn(name, self)
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
            raise com.IbisError('Table operation is not yet materialized')
        return self.op().schema

    def _is_materialized(self):
        # The operation produces a known schema
        return self.op().has_schema()

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


class AnyValue(ValueExpr): pass  # noqa: E701,E302
class AnyScalar(ScalarExpr, AnyValue): pass  # noqa: E701,E302
class AnyColumn(ColumnExpr, AnyValue): pass  # noqa: E701,E302


class NullValue(AnyValue): pass  # noqa: E701,E302
class NullScalar(AnyScalar, NullValue): pass  # noqa: E701,E302
class NullColumn(AnyColumn, NullValue): pass  # noqa: E701,E302


class NumericValue(AnyValue): pass  # noqa: E701,E302
class NumericScalar(AnyScalar, NumericValue): pass  # noqa: E701,E302
class NumericColumn(AnyColumn, NumericValue): pass  # noqa: E701,E302


class BooleanValue(NumericValue): pass  # noqa: E701,E302
class BooleanScalar(NumericScalar, BooleanValue): pass  # noqa: E701,E302
class BooleanColumn(NumericColumn, BooleanValue): pass  # noqa: E701,E302


class IntegerValue(NumericValue): pass  # noqa: E701,E302
class IntegerScalar(NumericScalar, IntegerValue): pass  # noqa: E701,E302
class IntegerColumn(NumericColumn, IntegerValue): pass  # noqa: E701,E302


class FloatingValue(NumericValue): pass  # noqa: E701,E302
class FloatingScalar(NumericScalar, FloatingValue): pass  # noqa: E701,E302
class FloatingColumn(NumericColumn, FloatingValue): pass  # noqa: E701,E302


class DecimalValue(NumericValue): pass  # noqa: E701,E302
class DecimalScalar(NumericScalar, DecimalValue): pass  # noqa: E701,E302
class DecimalColumn(NumericColumn, DecimalValue): pass  # noqa: E701,E302


class StringValue(AnyValue): pass  # noqa: E701,E302
class StringScalar(AnyScalar, StringValue): pass  # noqa: E701,E302
class StringColumn(AnyColumn, StringValue): pass  # noqa: E701,E302


class BinaryValue(AnyValue): pass  # noqa: E701,E302
class BinaryScalar(AnyScalar, BinaryValue): pass  # noqa: E701,E302
class BinaryColumn(AnyColumn, BinaryValue): pass  # noqa: E701,E302


class TimeValue(AnyValue): pass  # noqa: E701,E302
class TimeScalar(AnyScalar, TimeValue): pass  # noqa: E701,E302
class TimeColumn(AnyColumn, TimeValue): pass  # noqa: E701,E302


class DateValue(AnyValue): pass  # noqa: E701,E302
class DateScalar(AnyScalar, DateValue): pass  # noqa: E701,E302
class DateColumn(AnyColumn, DateValue): pass  # noqa: E701,E302


class TimestampValue(AnyValue): pass  # noqa: E701,E302
class TimestampScalar(AnyScalar, TimestampValue): pass  # noqa: E701,E302
class TimestampColumn(AnyColumn, TimestampValue): pass  # noqa: E701,E302


class CategoryValue(AnyValue): pass  # noqa: E701,E302
class CategoryScalar(AnyScalar, CategoryValue): pass  # noqa: E701,E302
class CategoryColumn(AnyColumn, CategoryValue): pass  # noqa: E701,E302


class EnumValue(AnyValue): pass  # noqa: E701,E302
class EnumScalar(AnyScalar, EnumValue): pass  # noqa: E701,E302
class EnumColumn(AnyColumn, EnumValue): pass  # noqa: E701,E302


class ArrayValue(AnyValue): pass  # noqa: E701,E302
class ArrayScalar(AnyScalar, ArrayValue): pass  # noqa: E701,E302
class ArrayColumn(AnyColumn, ArrayValue): pass  # noqa: E701,E302


class SetValue(AnyValue): pass  # noqa: E701,E302
class SetScalar(AnyScalar, SetValue): pass  # noqa: E701,E302
class SetColumn(AnyColumn, SetValue): pass  # noqa: E701,E302


class MapValue(AnyValue): pass  # noqa: E701,E302
class MapScalar(AnyScalar, MapValue): pass  # noqa: E701,E302
class MapColumn(AnyColumn, MapValue): pass  # noqa: E701,E302


class StructValue(AnyValue):

    def __dir__(self):
        return sorted(frozenset(
            itertools.chain(dir(type(self)), self.type().names)
        ))

class StructScalar(AnyScalar, StructValue): pass  # noqa: E701,E302
class StructColumn(AnyColumn, StructValue): pass  # noqa: E701,E302


class IntervalValue(AnyValue): pass  # noqa: E701,E302
class IntervalScalar(AnyScalar, IntervalValue): pass  # noqa: E701,E302
class IntervalColumn(AnyColumn, IntervalValue): pass  # noqa: E701,E302


class ListExpr(ColumnExpr, AnyValue):

    @property
    def values(self):
        return self.op().values

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __add__(self, other):
        other_values = tuple(getattr(other, 'values', other))
        return type(self.op())(self.values + other_values).to_expr()

    def __radd__(self, other):
        other_values = tuple(getattr(other, 'values', other))
        return type(self.op())(other_values + self.values).to_expr()

    def __bool__(self):
        return bool(self.values)

    def __len__(self):
        return len(self.values)


class TopKExpr(AnalyticExpr):

    def type(self):
        return 'topk'

    def _table_getitem(self):
        return self.to_filter()

    def to_filter(self):
        # TODO: move to api.py
        import ibis.expr.operations as ops
        return ops.SummaryFilter(self).to_expr()

    def to_aggregation(self, metric_name=None, parent_table=None,
                       backup_metric_name=None):
        """
        Convert the TopK operation to a table aggregation
        """
        op = self.op()

        arg_table = find_base_table(op.arg)

        by = op.by
        if not isinstance(by, Expr):
            by = by(arg_table)
            by_table = arg_table
        else:
            by_table = find_base_table(op.by)

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


class SortExpr(Expr):

    def _type_display(self):
        return 'array-sort'

    def get_name(self):
        return self.op().resolve_name()


class DayOfWeek(Expr):
    def index(self):
        """Get the index of the day of the week.

        Returns
        -------
        IntegerValue
            The index of the day of the week. Ibis follows pandas conventions,
            where **Monday = 0 and Sunday = 6**.
        """
        import ibis.expr.operations as ops
        return ops.DayOfWeekIndex(self.op().arg).to_expr()

    def full_name(self):
        """Get the name of the day of the week.

        Returns
        -------
        StringValue
            The name of the day of the week
        """
        import ibis.expr.operations as ops
        return ops.DayOfWeekName(self.op().arg).to_expr()


def bind_expr(table, expr):
    if isinstance(expr, (list, tuple)):
        return [bind_expr(table, x) for x in expr]

    return table._ensure_expr(expr)


# TODO: move to analysis
def find_base_table(expr):
    if isinstance(expr, TableExpr):
        return expr

    for arg in expr.op().flat_args():
        if isinstance(arg, Expr):
            r = find_base_table(arg)
            if isinstance(r, TableExpr):
                return r


_NULL = None


def null():
    """Create a NULL/NA scalar"""
    import ibis.expr.operations as ops

    global _NULL
    if _NULL is None:
        _NULL = ops.NullLiteral().to_expr()

    return _NULL


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
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops

    if hasattr(value, 'op') and isinstance(value.op(), ops.Literal):
        return value

    if value is null:
        dtype = dt.null
    else:
        dtype = dt.infer(value)

    if type is not None:
        try:
            # check that dtype is implicitly castable to explicitly given dtype
            dtype = dtype.cast(type, value=value)
        except com.IbisTypeError:
            raise TypeError('Value {!r} cannot be safely coerced '
                            'to {}'.format(value, type))

    if dtype is dt.null:
        return null().cast(dtype)
    else:
        return ops.Literal(value, dtype=dtype).to_expr()


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
    import ibis.expr.operations as ops

    return ops.ValueList(values).to_expr()


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


def param(type):
    """Create a parameter of a particular type to be defined just before
    execution.

    Parameters
    ----------
    type : dt.DataType
        The type of the unbound parameter, e.g., double, int64, date, etc.

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
    import ibis.expr.datatypes as dt
    import ibis.expr.operations as ops
    return ops.ScalarParameter(dt.dtype(type)).to_expr()


class UnnamedMarker(object):
    pass


unnamed = UnnamedMarker()
