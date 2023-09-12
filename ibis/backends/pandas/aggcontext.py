"""Implements an object to describe the context of a window aggregation.

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
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> s = pd.Series(df.value.sum(), index=df.index, name="sum_value")
    >>> s  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> t[
    ...     t, t.value.sum().name("sum_value")
    ... ].sum_value  # quartodoc: +SKIP # doctest: +SKIP


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
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.groupby("key").value.transform("sum")  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> t.value.sum().over(
    ...     ibis.window(group_by=t.key)
    ... )  # quartodoc: +SKIP # doctest: +SKIP

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
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.sort_values("time").value.cumsum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.cumulative_window(order_by=t.time)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP

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
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.sort_values("time").value.rolling(
    ...     3
    ... ).sum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.trailing_window(3, order_by=t.time)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP


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
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> sorter = lambda df: df.sort_values("time")
    >>> gb = (
    ...     df.groupby("key", group_keys=False)
    ...     .apply(sorter)
    ...     .reset_index(drop=True)
    ...     .groupby("key")
    ... )
    >>> rolling = gb.value.rolling(2)
    >>> rolling.sum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.trailing_window(2, order_by=t.time, group_by=t.key)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP
"""

from __future__ import annotations

import abc
import functools
import itertools
import operator
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from pandas.core.groupby import SeriesGroupBy

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.base.df.timecontext import (
    construct_time_context_aware_series,
    get_time_col,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np


class AggregationContext(abc.ABC):
    __slots__ = (
        "parent",
        "group_by",
        "order_by",
        "dtype",
        "max_lookback",
        "output_type",
    )

    def __init__(
        self,
        parent=None,
        group_by=None,
        order_by=None,
        max_lookback=None,
        output_type=None,
    ):
        self.parent = parent
        self.group_by = group_by
        self.order_by = order_by
        self.dtype = None if output_type is None else output_type.to_pandas()
        self.output_type = output_type
        self.max_lookback = max_lookback

    @abc.abstractmethod
    def agg(self, grouped_data, function, *args, **kwargs):
        pass


def wrap_for_apply(
    function: Callable,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> Callable:
    """Wrap a function for use with Pandas `apply`.

    Parameters
    ----------
    function : Callable
        A function to be used with Pandas `apply`.
    args : Optional[Tuple[Any, ...]]
        args to be passed to function when it is called by Pandas `apply`
    kwargs : Optional[Dict[str, Any]]
        kwargs to be passed to function when it is called by Pandas `apply`
    """
    assert callable(function), f"function {function} is not callable"

    new_args: tuple[Any, ...] = ()
    if args is not None:
        new_args = args

    new_kwargs: dict[str, Any] = {}
    if kwargs is not None:
        new_kwargs = kwargs

    @functools.wraps(function)
    def wrapped_func(
        data: Any,
        function: Callable = function,
        args: tuple[Any, ...] = new_args,
        kwargs: dict[str, Any] = new_kwargs,
    ) -> Callable:
        return function(data, *args, **kwargs)

    return wrapped_func


def wrap_for_agg(
    function: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable:
    """Wrap a function for use with Pandas `agg`.

    This includes special logic that will force Pandas `agg` to always treat
    the function as an aggregation function. Details:

    When passed a function, Pandas `agg` will either:
    1) Behave like Pandas `apply` and treat the function as a N->N mapping
      function (i.e. calls the function once for every value in the Series
      that `agg` is being called on), OR
    2) Treat the function as a N->1 aggregation function (i.e. calls the
      function once on the entire Series)
    Pandas `agg` will use behavior #1 unless an error is raised when doing so.

    We want to force Pandas `agg` to use behavior #2. To do this, we will wrap
    the function with logic that checks that a Series is being passed in, and
    raises a TypeError otherwise. When Pandas `agg` is attempting to use
    behavior #1 but sees the TypeError, it will fall back to behavior #2.

    Parameters
    ----------
    function : Callable
        An aggregation function to be used with Pandas `agg`.
    args : Tuple[Any, ...]
        args to be passed to function when it is called by Pandas `agg`
    kwargs : Dict[str, Any]
        kwargs to be passed to function when it is called by Pandas `agg`
    """
    assert callable(function), f"function {function} is not callable"

    @functools.wraps(function)
    def wrapped_func(
        data: Any,
        function: Callable = function,
        args: tuple[Any, ...] = args,
        kwargs: dict[str, Any] = kwargs,
    ) -> Callable:
        # `data` will be a scalar here if Pandas `agg` is trying to behave like
        # like Pandas `apply`.
        if not isinstance(data, pd.Series):
            # Force `agg` to NOT behave like `apply`. We want Pandas to use
            # `function` as an aggregation function, not as a mapping function.
            raise TypeError(
                f"This function expects a Series, but saw an object of type "
                f"{type(data)} instead."
            )
        return function(data, *args, **kwargs)

    return wrapped_func


class Summarize(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if isinstance(function, str):
            return getattr(grouped_data, function)(*args, **kwargs)

        if not callable(function):
            raise TypeError(f"Object {function} is not callable or a string")

        if isinstance(grouped_data, pd.core.groupby.generic.SeriesGroupBy) and len(
            grouped_data
        ):
            # `SeriesGroupBy.agg` does not allow np.arrays to be returned
            # from UDFs. To avoid `SeriesGroupBy.agg`, we will call the
            # aggregation function manually on each group. (#2768)
            aggs = {}
            for k, v in grouped_data:
                func_args = [d.get_group(k) for d in args]
                aggs[k] = function(v, *func_args, **kwargs)
                grouped_col_name = v.name
            return (
                pd.Series(aggs)
                .rename(grouped_col_name)
                .rename_axis(grouped_data.grouper.names)
            )
        else:
            return grouped_data.agg(wrap_for_agg(function, args, kwargs))


class Transform(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        # If this is a multi column UDF, then we cannot use
        # "transform" here (Data must be 1-dimensional)
        # Instead, we need to use "apply", which can return a non
        # numeric type, e.g, tuple of two double.
        if self.output_type.is_struct():
            res = grouped_data.apply(function, *args, **kwargs)
        else:
            res = grouped_data.transform(function, *args, **kwargs)

        # The result series uses the name of the input. We should
        # unset it to avoid confusion, when result is not guaranteed
        # to be the same series / have the same type after transform
        res.name = None
        return res


@functools.singledispatch
def compute_window_spec(dtype, obj):
    raise com.IbisTypeError(
        f"Unknown dtype type {dtype} and object {obj} for compute_window_spec"
    )


@compute_window_spec.register(dt.Integer)
def compute_window_spec_none(_, obj):
    """Helper method only used for row-based windows.

    Window spec in ibis is an inclusive window bound. A bound of 0
    indicates the current row. Window spec in Pandas indicates window
    size. Therefore, we must add 1 to the ibis window bound to get the
    expected behavior.
    """
    from ibis.backends.pandas.core import execute

    value = execute(obj)
    return value + 1


@compute_window_spec.register(dt.Interval)
def compute_window_spec_interval(_, obj):
    from ibis.backends.pandas.core import execute

    value = execute(obj)
    return pd.tseries.frequencies.to_offset(value)


def window_agg_built_in(
    frame: pd.DataFrame,
    windowed: pd.core.window.Window,
    function: str,
    max_lookback: ops.Literal,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> pd.Series:
    """Apply window aggregation with built-in aggregators."""
    assert isinstance(function, str)
    method = operator.methodcaller(function, *args, **kwargs)

    if max_lookback is not None:
        agg_method = method

        def sliced_agg(s):
            return agg_method(s.iloc[-max_lookback.value :])

        method = operator.methodcaller("apply", sliced_agg, raw=False)

    result = method(windowed)
    index = result.index
    result.index = pd.MultiIndex.from_arrays(
        [frame.index] + list(map(index.get_level_values, range(index.nlevels))),
        names=[frame.index.name] + index.names,
    )
    return result


def create_window_input_iter(
    grouped_data: SeriesGroupBy | pd.Series,
    masked_window_lower_indices: pd.Series,
    masked_window_upper_indices: pd.Series,
) -> Iterator[np.ndarray]:
    # create a generator for each input series
    # the generator will yield a slice of the
    # input series for each valid window
    data = getattr(grouped_data, "obj", grouped_data).values
    lower_indices_array = masked_window_lower_indices.values
    upper_indices_array = masked_window_upper_indices.values
    for i in range(len(lower_indices_array)):
        lower_index = lower_indices_array[i]
        upper_index = upper_indices_array[i]
        yield data[lower_index:upper_index]


def window_agg_udf(
    grouped_data: SeriesGroupBy,
    function: Callable,
    window_lower_indices: pd.Series,
    window_upper_indices: pd.Series,
    mask: pd.Series,
    result_index: pd.Index,
    dtype: np.dtype,
    max_lookback: int,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> pd.Series:
    """Apply window aggregation with UDFs.

    Notes
    -----
    Use custom logic to computing rolling window UDF instead of
    using pandas's rolling function.
    This is because pandas's rolling function doesn't support
    multi param UDFs.
    """
    assert len(window_lower_indices) == len(window_upper_indices)
    assert len(window_lower_indices) == len(mask)

    # Reset index here so we don't need to deal with mismatching
    # indices
    window_lower_indices = window_lower_indices.reset_index(drop=True)
    window_upper_indices = window_upper_indices.reset_index(drop=True)
    mask = mask.reset_index(drop=True)

    # Compute window indices and manually roll
    # over the window.

    # If an window has only nan values, we output nan for
    # the window result. This follows pandas rolling apply
    # behavior.

    # The first input column is in grouped_data, but there may
    # be additional input columns in args.
    inputs = (grouped_data,) + args

    masked_window_lower_indices = window_lower_indices[mask].astype("i8")
    masked_window_upper_indices = window_upper_indices[mask].astype("i8")

    input_iters = [
        create_window_input_iter(
            arg, masked_window_lower_indices, masked_window_upper_indices
        )
        if isinstance(arg, (pd.Series, SeriesGroupBy))
        else itertools.repeat(arg)
        for arg in inputs
    ]

    valid_result = pd.Series(
        function(*(next(gen) for gen in input_iters))
        for i in range(len(masked_window_lower_indices))
    )

    valid_result = pd.Series(valid_result)
    valid_result.index = masked_window_lower_indices.index
    result = pd.Series(index=mask.index, dtype=dtype)
    result[mask] = valid_result
    result.index = result_index

    return result


class Window(AggregationContext):
    __slots__ = ("construct_window",)

    def __init__(self, kind, *args, **kwargs):
        super().__init__(
            parent=kwargs.pop("parent", None),
            group_by=kwargs.pop("group_by", None),
            order_by=kwargs.pop("order_by", None),
            output_type=kwargs.pop("output_type"),
            max_lookback=kwargs.pop("max_lookback", None),
        )
        self.construct_window = operator.methodcaller(kind, *args, **kwargs)

    def agg(
        self,
        grouped_data: pd.Series | SeriesGroupBy,
        function: str | Callable,
        *args: Any,
        **kwargs: Any,
    ) -> pd.Series:
        # avoid a pandas warning about numpy arrays being passed through
        # directly
        group_by = self.group_by
        order_by = self.order_by

        assert group_by or order_by

        # Get the DataFrame from which the operand originated
        # (passed in when constructing this context object in
        # execute_node(ops.Window))
        parent = self.parent
        frame = getattr(parent, "obj", parent)
        obj = getattr(grouped_data, "obj", grouped_data)
        name = obj.name
        if frame[name] is not obj or name in group_by or name in order_by:
            name = f"{name}_{ibis.util.guid()}"
            frame = frame.assign(**{name: obj})

        # set the index to our order_by keys and append it to the existing
        # index
        # TODO: see if we can do this in the caller, when the context
        # is constructed rather than pulling out the data
        columns = group_by + order_by + [name]
        # Create a new frame to avoid mutating the original one
        indexed_by_ordering = frame[columns].copy()
        # placeholder column to compute window_sizes below
        indexed_by_ordering["_placeholder"] = 0
        indexed_by_ordering = indexed_by_ordering.set_index(order_by)

        # regroup if needed
        if group_by:
            grouped_frame = indexed_by_ordering.groupby(group_by, group_keys=False)
        else:
            grouped_frame = indexed_by_ordering
        grouped = grouped_frame[name]

        if callable(function):
            # To compute the window_size, we need to construct a
            # RollingGroupby and compute count using construct_window.
            # However, if the RollingGroupby is not numeric, e.g.,
            # we are calling window UDF on a timestamp column, we
            # cannot compute rolling count directly because:
            # (1) windowed.count() will exclude NaN observations
            #     , which results in incorrect window sizes.
            # (2) windowed.apply(len, raw=True) will include NaN
            #     obversations, but doesn't work on non-numeric types.
            #     https://github.com/pandas-dev/pandas/issues/23002
            # To deal with this, we create a _placeholder column

            windowed_frame = self.construct_window(grouped_frame)
            window_sizes = windowed_frame["_placeholder"].count().reset_index(drop=True)
            mask = ~(window_sizes.isna())
            window_upper_indices = pd.Series(range(len(window_sizes))) + 1
            window_lower_indices = window_upper_indices - window_sizes
            # The result Series of udf may need to be trimmed by
            # timecontext. In order to do so, 'time' must be added
            # as an index to the Series, if present. Here We extract
            # time column from the parent Dataframe `frame`.
            if get_time_col() in frame:
                result_index = construct_time_context_aware_series(obj, frame).index
            else:
                result_index = obj.index
            result = window_agg_udf(
                grouped_data,
                function,
                window_lower_indices,
                window_upper_indices,
                mask,
                result_index,
                self.dtype,
                self.max_lookback,
                *args,
                **kwargs,
            )
        else:
            # perform the per-group rolling operation
            windowed = self.construct_window(grouped)
            result = window_agg_built_in(
                frame,
                windowed,
                function,
                self.max_lookback,
                *args,
                **kwargs,
            )
        try:
            return result.astype(self.dtype, copy=False)
        except (TypeError, ValueError):
            return result


class Cumulative(Window):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__("expanding", *args, **kwargs)


class Moving(Window):
    __slots__ = ()

    def __init__(self, start, max_lookback, *args, **kwargs):
        from ibis.backends.pandas.core import timedelta_types

        start = compute_window_spec(start.dtype, start.value)
        if isinstance(start, timedelta_types + (pd.offsets.DateOffset,)):
            closed = "both"
        else:
            closed = None

        super().__init__(
            "rolling",
            start,
            *args,
            max_lookback=max_lookback,
            closed=closed,
            min_periods=1,
            **kwargs,
        )

    def short_circuit_method(self, grouped_data, function):
        raise AttributeError("No short circuit method for rolling operations")
