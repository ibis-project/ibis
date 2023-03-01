from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Type, Union

import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.backends.pandas.execution.util as pd_util
import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util
from ibis.backends.base.df.scope import Scope
from ibis.backends.base.df.timecontext import TimeContext
from ibis.backends.dask.core import execute
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher
from ibis.common import graph

DispatchRule = Tuple[Tuple[Union[Type, Tuple], ...], Callable]

TypeRegistrationDict = Dict[
    Union[Type[ops.Node], Tuple[Type[ops.Node], ...]], List[DispatchRule]
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

    >>> coerce_to_output(pd.Series(1), expr)  # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr)  # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr, [1,2,3])  # doctest: +SKIP
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1,2,3], expr)  # doctest: +SKIP
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

    return dd.from_pandas(pd_util.coerce_to_output(result, node, index), npartitions=1)


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
    should be hit rarely in real usage. A situtation that triggeres this slow
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
    key: ops.SortKey,
    data: dd.DataFrame,
    timecontext: TimeContext | None = None,
    scope: Scope = None,
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
    if key.name in data:
        return name, data[key.name]
    else:
        if scope is None:
            scope = Scope()
        scope = scope.merge_scopes(
            Scope({t: data}, timecontext) for t in an.find_immediate_parent_tables(key)
        )
        new_column = execute(key, scope=scope, **kwargs)
        new_column.name = name
        return name, new_column


def compute_sorted_frame(
    df: dd.DataFrame,
    order_by: ir.Value,
    timecontext: TimeContext | None = None,
    **kwargs,
) -> dd.DataFrame:
    sort_col_name, temporary_column = compute_sort_key(
        order_by, df, timecontext, **kwargs
    )
    result = df.assign(**{sort_col_name: temporary_column})
    result = result.set_index(sort_col_name).reset_index(drop=True)
    return result


def assert_identical_grouping_keys(*args):
    indices = [arg.index for arg in args]
    # Depending on whether group_by was called like group_by("col") or
    # group_by(["cold"]) index will be a string or a list
    if isinstance(indices[0], list):
        indices = [tuple(index) for index in indices]
    grouping_keys = set(indices)
    if len(grouping_keys) != 1:
        raise AssertionError(f"Differing grouping keys passed: {grouping_keys}")


def add_partitioned_sorted_column(
    df: dd.DataFrame | dd.Series,
) -> dd.DataFrame:
    """Add a column that is already partitioned and sorted.

    This column acts as if we had a global index across the distributed data.

    Important properties:

    - Each row has a unique id (i.e. a value in this column)
    - IDs within each partition are already sorted
    - Any id in partition $N_{t}$ is less than any id in partition $N_{t+1}$

    We do this by designating a sufficiently large space of integers per
    partition via a base and adding the existing index to that base. See
    `helper` below.

    Though the space per partition is bounded, real world usage should not
    hit these bounds. We also do not explicity deal with overflow in the
    bounds.

    Parameters
    ----------
    df : dd.DataFrame
        Dataframe to add the column to

    Returns
    -------
    dd.DataFrame
        New dask dataframe with sorted partitioned index

    Examples
    --------
    >>> ddf = dd.from_pandas(pd.DataFrame({'a': [1, 2,3, 4]}), npartitions=2)
    >>> ddf  # doctest: +SKIP
    Dask DataFrame Structure:
                    a
    npartitions=2
    0              int64
    2                ...
    3                ...
    Dask Name: from_pandas, 2 task
    >>> ddf.compute()  # doctest: +SKIP
       a
    0  1
    1  2
    2  3
    3  4
    >>> ddf = add_partitioned_sorted_column(ddf)
    >>> ddf  # doctest: +SKIP
    Dask DataFrame Structure:
                    a
    npartitions=2
    0              int64
    4294967296       ...
    8589934592       ...
    Dask Name: set_index, 8 tasks
    Name: result, dtype: int64
    >>> ddf.compute()  # doctest: +SKIP
                a
    _ibis_index
    0            1
    1            2
    4294967296   3
    4294967297   4
    """
    if isinstance(df, dd.Series):
        df = df.to_frame()

    col_name = "_ibis_index"

    if col_name in df.columns:
        raise ValueError(f"Column {col_name} is already present in DataFrame")

    def helper(
        df: pd.Series | pd.DataFrame,
        partition_info: dict[str, Any],  # automatically injected by dask
        col_name: str,
    ):
        """Assigns a column with a unique id for each row."""
        if len(df) > (2**31):
            raise ValueError(
                f"Too many items in partition {partition_info} to add"
                "partitioned sorted column without overflowing."
            )
        base = partition_info["number"] << 32
        return df.assign(**{col_name: [base + idx for idx in df.index]})

    original_meta = df._meta.dtypes.to_dict()
    new_meta = {**original_meta, **{col_name: "int64"}}
    df = df.reset_index(drop=True)
    df = df.map_partitions(helper, col_name=col_name, meta=new_meta)
    # Divisions include the minimum value of every partition's index and the
    # maximum value of the last partition's index
    divisions = tuple(x << 32 for x in range(df.npartitions + 1))

    df = df.set_index(col_name, sorted=True, divisions=divisions)

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
