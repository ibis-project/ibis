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

Trailing
~~~~~~~~

Also called rolling.

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


@six.add_metaclass(abc.ABCMeta)
class AggregationContext(object):

    __slots__ = ()

    @abc.abstractmethod
    def agg(self, grouped_data, function, *args, **kwargs):
        pass


def _apply(function, args, kwargs):
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


class Window(AggregationContext):

    __slots__ = 'construct_window',

    def __init__(self, kind, *args, **kwargs):
        self.construct_window = operator.methodcaller(kind, *args, **kwargs)

    def agg(self, grouped_data, function, *args, **kwargs):
        if callable(function):
            return self.construct_window(grouped_data).apply(
                _apply(function, args, kwargs)
            )

        if not isinstance(function, six.string_types):
            raise TypeError(
                '{} aggregation function must be a string or callable'.format(
                    type(self).__name__
                )
            )

        try:
            method = self.short_circuit_method(grouped_data, function)
        except AttributeError:
            method = getattr(self.construct_window(grouped_data), function)

        return method(*args, **kwargs)


class Cumulative(Window):

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(Cumulative, self).__init__('expanding', *args, **kwargs)

    def short_circuit_method(self, grouped_data, function):
        return getattr(grouped_data, 'cum{}'.format(function))


class Trailing(Window):

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(Trailing, self).__init__('rolling', *args, **kwargs)

    def short_circuit_method(self, grouped_data, function):
        raise AttributeError(
            'No short circuit method for rolling operations'
        )
