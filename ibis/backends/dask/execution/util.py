from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Union

import dask.dataframe as dd
import dask.delayed
import pandas as pd

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.base.df.scope import Scope
from ibis.backends.dask.core import execute
from ibis.common import graph

if TYPE_CHECKING:
    import numpy as np
    from dask.dataframe.groupby import SeriesGroupBy

    from ibis.backends.base.df.timecontext import TimeContext
    from ibis.backends.pandas.trace import TraceTwoLevelDispatcher
    from ibis.expr.operations.sortkeys import SortKey

DispatchRule = tuple[tuple[Union[type, tuple], ...], Callable]

TypeRegistrationDict = dict[
    Union[type[ops.Node], tuple[type[ops.Node], ...]], list[DispatchRule]
]


def register_types_to_dispatcher(
    dispatcher: TraceTwoLevelDispatcher, types: TypeRegistrationDict
):
    """Perform registrations in bulk.

    Many dask operations utilize the functions defined in the pandas backend
    without modification.
    """
    for ibis_op, registration_list in types.items():
        for types_to_register, fn in registration_list:
            dispatcher.register(ibis_op, *types_to_register)(fn)


def make_meta_series(
    dtype: np.dtype,
    name: str | None = None,
    meta_index: pd.Index | None = None,
):
    if isinstance(meta_index, pd.MultiIndex):
        index_names = meta_index.names
        series_index = pd.MultiIndex(
            levels=[[]] * len(index_names),
            codes=[[]] * len(index_names),
            names=index_names,
        )
    elif isinstance(meta_index, pd.Index):
        series_index = pd.Index([], name=meta_index.name)
    else:
        series_index = pd.Index([])

    return pd.Series(
        [],
        index=series_index,
        dtype=dtype,
        name=name,
    )


def make_selected_obj(gs: SeriesGroupBy) -> dd.DataFrame | dd.Series:
    """Select a column from a `pandas.DataFrameGroupBy`."""
    # TODO profile this for data shuffling
    # We specify drop=False in the case that we are grouping on the column
    # we are selecting
    if isinstance(gs.obj, dd.Series):
        return gs.obj
    else:
        return gs.obj.set_index(gs.index, drop=False)[gs._meta._selected_obj.name]


def coerce_to_output(
    result: Any, node: ops.Node, index: pd.Index | None = None
) -> dd.Series | dd.DataFrame:
    """Cast the result to either a Series of DataFrame, renaming as needed.

    Reimplementation of `coerce_to_output` in the pandas backend, but
    creates dask objects and adds special handling for dd.Scalars.

    Parameters
    ----------
    result: Any
        The result to cast
    node: ibis.expr.operations.Node
        The operation node associated with the result
    index: pd.Index
        Optional. If passed, scalar results will be broadcasted according
        to the index.

    Returns
    -------
    result: A `dd.Series` or `dd.DataFrame`

    Raises
    ------
    ValueError
        If unable to coerce result

    Examples
    --------
    Examples below use pandas objects for legibility, but functionality is the
    same on dask objects.

    >>> coerce_to_output(pd.Series(1), expr)  # quartodoc: +SKIP # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr)  # quartodoc: +SKIP # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr, [1, 2, 3])  # quartodoc: +SKIP # doctest: +SKIP
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1, 2, 3], expr)  # quartodoc: +SKIP # doctest: +SKIP
    0    [1, 2, 3]
    Name: result, dtype: object
    """
    result_name = node.name

    if isinstance(result, (pd.DataFrame, dd.DataFrame)):
        result = result.apply(dict, axis=1)
        return result.rename(result_name)

    if isinstance(result, (pd.Series, dd.Series)):
        # Series from https://github.com/ibis-project/ibis/issues/2711
        return result.rename(result_name)

    if isinstance(result, dd.core.Scalar):
        # wrap the scalar in a series
        out_dtype = _pandas_dtype_from_dd_scalar(result)
        out_len = 1 if index is None else len(index)
        meta = make_meta_series(dtype=out_dtype, name=result_name)
        # Specify `divisions` so that the created Dask object has
        # known divisions (to be concatenatable with Dask objects
        # created using `dd.from_pandas`)
        series = dd.from_delayed(
            _wrap_dd_scalar(result, result_name, out_len),
            meta=meta,
            divisions=(0, out_len - 1),
        )
        return series

    # Wrap `result` in a single-element Series.
    return dd.from_pandas(pd.Series([result], name=result_name), npartitions=1)


@dask.delayed
def _wrap_dd_scalar(x, name=None, series_len=1):
    return pd.Series([x for _ in range(series_len)], name=name)


def _pandas_dtype_from_dd_scalar(x: dd.core.Scalar):
    try:
        return x.dtype
    except AttributeError:
        return pd.Series([x._meta]).dtype


def safe_concat(dfs: list[dd.Series | dd.DataFrame]) -> dd.DataFrame:
    """Concatenate a list of `dd.Series` or `dd.DataFrame` objects into a DataFrame.

    This will use `DataFrame.concat` if all pieces are the same length.
    Otherwise we will iterratively join.

    When axis=1 and divisions are unknown, Dask `DataFrame.concat` can only
    operate on objects with equal lengths, otherwise it will raise a
    ValueError in `concat_and_check`.

    See https://github.com/dask/dask/blob/2c2e837674895cafdb0612be81250ef2657d947e/dask/dataframe/multi.py#L907.

    Note - Repeatedly joining dataframes is likely to be quite slow, but this
    should be hit rarely in real usage. A situation that triggers this slow
    path is aggregations where aggregations return different numbers of rows
    (see `test_aggregation_group_by` for a specific example).
    TODO - performance.
    """
    if len(dfs) == 1:
        maybe_df = dfs[0]
        if isinstance(maybe_df, dd.Series):
            return maybe_df.to_frame()
        else:
            return maybe_df

    lengths = list(map(len, dfs))
    if len(set(lengths)) != 1:
        result = dfs[0].to_frame()

        for other in dfs[1:]:
            result = result.join(other.to_frame(), how="outer")
    else:
        result = dd.concat(dfs, axis=1)

    return result


def compute_sort_key(
    key: str | SortKey,
    data: dd.DataFrame,
    timecontext: TimeContext,
    scope: Scope | None = None,
    **kwargs,
):
    """Compute a sort key.

    We use this function instead of the pandas.execution.util so that we
    use the dask `execute` method.

    This function borrows the logic in the pandas backend. `by` can be a
    string or an expression. If `by.get_name()` raises an exception, we must
    `execute` the expression and sort by the new derived column.
    """
    name = ibis.util.guid()
    if key.shape.is_columnar():
        if key.name in data:
            return name, data[key.name]
        if isinstance(key, str):
            return key, None
        else:
            if scope is None:
                scope = Scope()
            scope = scope.merge_scopes(
                Scope({t: data}, timecontext)
                for t in an.find_immediate_parent_tables(key)
            )
            new_column = execute(key, scope=scope, **kwargs)
            new_column.name = name
            return name, new_column
    else:
        raise NotImplementedError(
            "Scalar sort keys are not yet supported in the dask backend"
        )


def compute_sorted_frame(
    df: dd.DataFrame,
    order_by: list[str | SortKey],
    group_by: list[str | SortKey] | None = None,
    timecontext=None,
    **kwargs,
) -> dd.DataFrame:
    sort_keys = []
    ascending = []

    if group_by is None:
        group_by = []

    for value in group_by:
        sort_keys.append(value)
        ascending.append(True)

    for key in order_by:
        sort_keys.append(key)
        ascending.append(key.ascending)

    new_columns = {}
    computed_sort_keys = []
    for key in sort_keys:
        computed_sort_key, temporary_column = compute_sort_key(
            key, df, timecontext, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)
        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(
        computed_sort_keys, ascending=ascending, kind="mergesort"
    )
    # TODO: we'll eventually need to return this frame with the temporary
    # columns and drop them in the caller (maybe using post_execute?)
    ngrouping_keys = len(group_by)
    return (
        result,
        computed_sort_keys[:ngrouping_keys],
        computed_sort_keys[ngrouping_keys:],
    )


def assert_identical_grouping_keys(*args):
    indices = [arg.index for arg in args]
    # Depending on whether group_by was called like group_by("col") or
    # group_by(["cold"]) index will be a string or a list
    if isinstance(indices[0], list):
        indices = [tuple(index) for index in indices]
    grouping_keys = set(indices)
    if len(grouping_keys) != 1:
        raise AssertionError(f"Differing grouping keys passed: {grouping_keys}")


def add_globally_consecutive_column(
    df: dd.DataFrame | dd.Series,
    col_name: str = "_ibis_index",
    set_as_index: bool = True,
) -> dd.DataFrame:
    """Add a column that is globally consecutive across the distributed data.

    By construction, this column is already sorted and can be used to partition
    the data.
    This column can act as if we had a global index across the distributed data.
    This index needs to be consecutive in the range of [0, len(df)), allows
    downstream operations to work properly.
    The default index of dask dataframes is to be consecutive within each partition.

    Important properties:

    - Each row has a unique id (i.e. a value in this column)
    - The global index that's added is consecutive in the same order that the rows currently are in.
    - IDs within each partition are already sorted

    We also do not explicitly deal with overflow in the bounds.

    Parameters
    ----------
    df : dd.DataFrame
        Dataframe to add the column to
    col_name: str
        Name of the column to use. Default is _ibis_index
    set_as_index: bool
        If True, will set the consecutive column as the index. Default is True.

    Returns
    -------
    dd.DataFrame
        New dask dataframe with sorted partitioned index
    """
    if isinstance(df, dd.Series):
        df = df.to_frame()

    if col_name in df.columns:
        raise ValueError(f"Column {col_name} is already present in DataFrame")

    df = df.assign(**{col_name: 1})
    df = df.assign(**{col_name: df[col_name].cumsum() - 1})
    if set_as_index:
        df = df.reset_index(drop=True)
        df = df.set_index(col_name, sorted=True)
    return df


def is_row_order_preserving(nodes) -> bool:
    """Detects if the operation preserves row ordering.

    Certain operations we know will not affect the ordering of rows in
    the dataframe (for example elementwise operations on ungrouped
    dataframes). In these cases we may be able to avoid expensive joins
    and assign directly into the parent dataframe.
    """

    def _is_row_order_preserving(node: ops.Node):
        if isinstance(node, (ops.Reduction, ops.Window)):
            return (graph.halt, False)
        else:
            return (graph.proceed, True)

    return graph.traverse(_is_row_order_preserving, nodes)


def rename_index(df: dd.DataFrame, new_index_name: str) -> dd.DataFrame:
    # No elegant way to rename index
    # https://github.com/dask/dask/issues/4950
    df = df.map_partitions(pd.DataFrame.rename_axis, new_index_name, axis="index")
    return df
