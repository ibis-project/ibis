"""Implements an object to describe in what context a window aggregation is
occurring.

For any particular aggregation such as ``sum``, ``mean``, etc we need to decide
based on the presence or absence of other expressions like ``group_by`` and
``order_by`` whether we should call a different method of aggregation.

Here are the different aggregation contexts and the conditions under which they
are used.

Note that in the pandas backend, only trailing and cumulative windows are
supported right now.

No ``group_by`` or ``order_by``: ``context.Summarize()``
--------------------------------------------------------
This is an aggregation on a column, repeated for every row in the table.

SQL

::

    SELECT SUM(value) OVER () AS sum_value FROM t

Pandas

::
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'key': list('aabc'),
    ...     'value': np.random.randn(4),
    ...     'time': pd.date_range(periods=4, start='now')
    ... })
    >>> s = pd.Series(df.value.sum(), index=df.index, name='sum_value')
    >>> s  # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = [
    ...    ('time', 'timestamp'), ('key', 'string'), ('value', 'double')
    ... ]
    >>> t = ibis.table(schema, name='t')
    >>> t[t, t.value.sum().name('sum_value')].sum_value  # doctest: +SKIP


``group_by``, no ``order_by``: ``context.Transform()``
------------------------------------------------------

This performs an aggregation per group and repeats it across every row in the
group.

SQL

::

    SELECT SUM(value) OVER (PARTITION BY key) AS sum_value
    FROM t

Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'key': list('aabc'),
    ...     'value': np.random.randn(4),
    ...     'time': pd.date_range(periods=4, start='now')
    ... })
    >>> df.groupby('key').value.transform('sum')  # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = [
    ...     ('time', 'timestamp'), ('key', 'string'), ('value', 'double')
    ... ]
    >>> t = ibis.table(schema, name='t')
    >>> t.value.sum().over(ibis.window(group_by=t.key))  # doctest: +SKIP

``order_by``, no ``group_by``: ``context.Cumulative()``/``context.Rolling()``
-----------------------------------------------------------------------------

Cumulative and trailing window operations.

Cumulative
~~~~~~~~~~

Also called expanding.

SQL

::

    SELECT SUM(value) OVER (
        ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'key': list('aabc'),
    ...     'value': np.random.randn(4),
    ...     'time': pd.date_range(periods=4, start='now')
    ... })
    >>> df.sort_values('time').value.cumsum()  # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = [
    ...     ('time', 'timestamp'), ('key', 'string'), ('value', 'double')
    ... ]
    >>> t = ibis.table(schema, name='t')
    >>> window = ibis.cumulative_window(order_by=t.time)
    >>> t.value.sum().over(window)  # doctest: +SKIP

Moving
~~~~~~

Also called referred to as "rolling" in other libraries such as pandas.

SQL

::

    SELECT SUM(value) OVER (
        ORDER BY time ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'key': list('aabc'),
    ...     'value': np.random.randn(4),
    ...     'time': pd.date_range(periods=4, start='now')
    ... })
    >>> df.sort_values('time').value.rolling(3).sum()  # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = [
    ...     ('time', 'timestamp'), ('key', 'string'), ('value', 'double')
    ... ]
    >>> t = ibis.table(schema, name='t')
    >>> window = ibis.trailing_window(3, order_by=t.time)
    >>> t.value.sum().over(window)  # doctest: +SKIP


``group_by`` and ``order_by``: ``context.Cumulative()``/``context.Rolling()``
-----------------------------------------------------------------------------

This performs a cumulative or rolling operation within a group.

SQL

::

    SELECT SUM(value) OVER (
        PARTITION BY key ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'key': list('aabc'),
    ...     'value': np.random.randn(4),
    ...     'time': pd.date_range(periods=4, start='now')
    ... })
    >>> sorter = lambda df: df.sort_values('time')
    >>> gb = df.groupby('key').apply(sorter).groupby('key')
    >>> gb.groupby('key').value.rolling(3).sum()  # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = [
    ...     ('time', 'timestamp'), ('key', 'string'), ('value', 'double')
    ... ]
    >>> t = ibis.table(schema, name='t')
    >>> window = ibis.trailing_window(3, order_by=t.time, group_by=t.key)
    >>> t.value.sum().over(window)  # doctest: +SKIP
"""

import abc
import operator

import six

from multipledispatch import Dispatcher

import pandas as pd

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.types as ir


@six.add_metaclass(abc.ABCMeta)
class AggregationContext(object):
    __slots__ = 'parent', 'group_by', 'order_by', 'dtype'

    def __init__(self, parent=None, group_by=None, order_by=None, dtype=None):
        self.parent = parent
        self.group_by = group_by
        self.order_by = order_by
        self.dtype = dtype

    @abc.abstractmethod
    def agg(self, grouped_data, function, *args, **kwargs):
        pass


def _apply(function, args, kwargs):
    assert callable(function), 'function {} is not callable'.format(function)
    return lambda data, function=function, args=args, kwargs=kwargs: (
        function(data, *args, **kwargs)
    )


class Summarize(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if isinstance(function, six.string_types):
            return getattr(grouped_data, function)(*args, **kwargs)

        if not callable(function):
            raise TypeError(
                'Object {} is not callable or a string'.format(function)
            )

        return grouped_data.apply(_apply(function, args, kwargs))


class Transform(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        return grouped_data.transform(function, *args, **kwargs)


compute_window_spec = Dispatcher('compute_window_spec')


@compute_window_spec.register(ir.Expr, dt.Interval)
def compute_window_spec_interval(expr, dtype):
    value = ibis.pandas.execute(expr)
    return pd.tseries.frequencies.to_offset(value)


@compute_window_spec.register(ir.Expr, dt.DataType)
def compute_window_spec_expr(expr, _):
    return ibis.pandas.execute(expr)


@compute_window_spec.register(object, type(None))
def compute_window_spec_default(obj, _):
    return obj


class Window(AggregationContext):
    __slots__ = 'construct_window',

    def __init__(self, kind, *args, **kwargs):
        super(Window, self).__init__(
            parent=kwargs.pop('parent', None),
            group_by=kwargs.pop('group_by', None),
            order_by=kwargs.pop('order_by', None),
            dtype=kwargs.pop('dtype'),
        )
        self.construct_window = operator.methodcaller(
            kind, *args, **kwargs)

    def agg(self, grouped_data, function, *args, **kwargs):
        group_by = self.group_by
        order_by = self.order_by

        # if we don't have a grouping key, just call into pandas
        if not group_by:
            # the result of calling .rolling(...) in pandas
            windowed = self.construct_window(grouped_data)

            # if we're a UD(A)F or a function that isn't a string (like the
            # collect implementation) then call apply
            if callable(function):
                return windowed.apply(_apply(function, args, kwargs))
            else:
                # otherwise we're a string and we're probably faster
                assert isinstance(function, six.string_types)
                method = getattr(windowed, function)
                result = method(*args, **kwargs)
                return result
        else:
            # do mostly the same thing as if we did NOT have a grouping key,
            # but don't call the callable just yet. See below where we call it
            if callable(function):
                method = operator.methodcaller(
                    'apply',
                    _apply(function, args, kwargs)
                )
            else:
                assert isinstance(function, six.string_types)
                method = operator.methodcaller(function, *args, **kwargs)

        # get the DataFrame from which the operand originated (passed in when
        # constructing this context object in execute_node(ops.WindowOp))
        frame = self.parent.obj
        name = getattr(grouped_data, 'obj', grouped_data).name

        # set the index to our order_by keys and append it to the existing
        # index
        # TODO: see if we can do this in the caller, when the context
        # is constructed rather than pulling out the data
        columns = group_by + order_by + [name]
        indexed_by_ordering = frame.loc[:, columns].set_index(order_by)

        # regroup
        grouped = indexed_by_ordering.groupby(group_by)[name]

        # perform the per-group rolling operation
        windowed = self.construct_window(grouped)
        result = method(windowed)
        index = result.index
        result.index = pd.MultiIndex.from_arrays(
            [frame.index] + list(map(
                index.get_level_values, range(index.nlevels)
            )),
            names=[frame.index.name] + index.names,
        )
        return result


class Cumulative(Window):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(Cumulative, self).__init__('expanding', *args, **kwargs)


class Moving(Window):
    __slots__ = ()

    def __init__(self, preceding, *args, **kwargs):
        from ibis.pandas.core import timedelta_types
        dtype = getattr(preceding, 'type', lambda: None)()
        preceding = compute_window_spec(preceding, dtype)
        closed = None if not isinstance(
            preceding, timedelta_types + (pd.offsets.DateOffset,)) else 'both'
        super(Moving, self).__init__(
            'rolling', preceding, *args, closed=closed, **kwargs
        )

    def short_circuit_method(self, grouped_data, function):
        raise AttributeError('No short circuit method for rolling operations')
