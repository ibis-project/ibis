from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Union

import dask.dataframe as dd

import ibis
from ibis.backends.pandas.aggcontext import (
    AggregationContext,
    compute_window_spec,
    construct_time_context_aware_series,
    get_time_col,
    window_agg_udf,
    wrap_for_agg,
)
from ibis.backends.pandas.aggcontext import Transform as PandasTransform

if TYPE_CHECKING:
    from dask.dataframe.groupby import SeriesGroupBy

# TODO Consolidate this logic with the pandas aggcontext.
# This file is almost a direct port of the pandas aggcontext.
# https://github.com/ibis-project/ibis/issues/5911


class Summarize(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if isinstance(function, str):
            return getattr(grouped_data, function)(*args, **kwargs)

        if not callable(function):
            raise TypeError(f"Object {function} is not callable or a string")

        elif isinstance(grouped_data, dd.Series):
            return grouped_data.reduction(wrap_for_agg(function, args, kwargs))
        else:
            return grouped_data.agg(wrap_for_agg(function, args, kwargs))


class Transform(PandasTransform):
    def agg(self, grouped_data, function, *args, **kwargs):
        res = super().agg(grouped_data, function, *args, **kwargs)
        index_name = res.index.name if res.index.name is not None else "index"
        res = res.reset_index().set_index(index_name).iloc[:, 0]
        return res


def dask_window_agg_built_in(
    frame: dd.DataFrame,
    windowed: dd.rolling.Rolling,
    function: str,
    max_lookback: int,
    *args: tuple[Any],
    **kwargs: dict[str, Any],
) -> dd.Series:
    """Apply window aggregation with built-in aggregators."""
    assert isinstance(function, str)
    method = operator.methodcaller(function, *args, **kwargs)

    if max_lookback is not None:
        agg_method = method

        def sliced_agg(s):
            return agg_method(s.iloc[-max_lookback:])

        method = operator.methodcaller("apply", sliced_agg, raw=False)

    result = method(windowed)
    # No MultiIndex support in dask
    result.index = frame.index
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
        grouped_data: Union[dd.Series, SeriesGroupBy],
        function: Union[str, Callable],
        *args: Any,
        **kwargs: Any,
    ) -> dd.Series:
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
        grouped_meta = getattr(grouped_data, "_meta_nonempty", grouped_data)
        obj = getattr(grouped_meta, "obj", grouped_data)
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
            window_upper_indices = dd.Series(range(len(window_sizes))) + 1
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
            result = dask_window_agg_built_in(
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
            # The dtypes in result could have been promoted during the agg
            # computation. Trying to downcast the type back with self.dtype will
            # fail but we want to result with the promoted types anyways.
            return result


class Cumulative(Window):
    __slots__ = ()

    def __init__(self, window, *args, **kwargs):
        super().__init__("rolling", *args, window=window, min_periods=1, **kwargs)


class Moving(Window):
    __slots__ = ()

    def __init__(self, start, max_lookback, *args, **kwargs):
        start = compute_window_spec(start.dtype, start.value)

        super().__init__(
            "rolling",
            start,
            *args,
            max_lookback=max_lookback,
            min_periods=1,
            **kwargs,
        )

    def short_circuit_method(self, grouped_data, function):
        raise AttributeError("No short circuit method for rolling operations")
