from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.backends.pandas.execution.util as pd_util
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.pandas.client import ibis_dtype_to_pandas
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher
from ibis.expr import datatypes as dt
from ibis.expr import types as ir
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..core import execute

DispatchRule = Tuple[Tuple[Union[Type, Tuple], ...], Callable]

TypeRegistrationDict = Dict[
    Union[Type[ops.Node], Tuple[Type[ops.Node], ...]], List[DispatchRule]
]


def register_types_to_dispatcher(
    dispatcher: TraceTwoLevelDispatcher, types: TypeRegistrationDict
):
    """
    Many dask operations utilize the functions defined in the pandas backend
    without modification. This function helps perform registrations in bulk
    """
    for ibis_op, registration_list in types.items():
        for types_to_register, fn in registration_list:
            dispatcher.register(ibis_op, *types_to_register)(fn)


def make_meta_series(dtype, name=None, index_name=None):
    return pd.Series(
        [],
        index=pd.Index([], name=index_name),
        dtype=dtype,
        name=name,
    )


def make_selected_obj(gs: SeriesGroupBy) -> Union[dd.DataFrame, dd.Series]:
    """
    When you select a column from a `pandas.DataFrameGroupBy` the underlying
    `.obj` reflects that selection. This function emulates that behavior.
    """
    # TODO profile this for data shuffling
    # We specify drop=False in the case that we are grouping on the column
    # we are selecting
    if isinstance(gs.obj, dd.Series):
        return gs.obj
    else:
        return gs.obj.set_index(gs.index, drop=False)[
            gs._meta._selected_obj.name
        ]


def coerce_to_output(
    result: Any, expr: ir.Expr, index: Optional[pd.Index] = None
) -> Union[dd.Series, dd.DataFrame]:
    """Cast the result to either a Series of DataFrame, renaming as needed.

    Reimplementation of `coerce_to_output` in the pandas backend, but
    creates dask objects and adds special handling for dd.Scalars.

    Parameters
    ----------
    result: Any
        The result to cast
    expr: ibis.expr.types.Expr
        The expression associated with the result
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
    For dataframe outputs, see ``_coerce_to_dataframe``. Examples below use
    pandas objects for legibility, but functionality is the same on dask
    objects.

    >>> coerce_to_output(pd.Series(1), expr)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, expr, [1,2,3])
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1,2,3], expr)
    0    [1, 2, 3]
    Name: result, dtype: object
    """
    result_name = expr.get_name()
    dataframe_exprs = (
        ir.DestructColumn,
        ir.StructColumn,
        ir.DestructScalar,
        ir.StructScalar,
    )
    if isinstance(expr, dataframe_exprs):
        return _coerce_to_dataframe(
            result, expr.type().names, expr.type().types
        )
    elif isinstance(result, (pd.Series, dd.Series)):
        # Series from https://github.com/ibis-project/ibis/issues/2711
        return result.rename(result_name)
    elif isinstance(expr.op(), ops.Reduction):
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
        else:
            return dd.from_pandas(
                pd_util.coerce_to_output(result, expr, index), npartitions=1
            )
    else:
        raise ValueError(f"Cannot coerce_to_output. Result: {result}")


@dask.delayed
def _wrap_dd_scalar(x, name=None, series_len=1):
    return pd.Series([x for _ in range(series_len)], name=name)


def _pandas_dtype_from_dd_scalar(x: dd.core.Scalar):
    try:
        return x.dtype
    except AttributeError:
        return pd.Series([x._meta]).dtype


def _coerce_to_dataframe(
    data: Any,
    column_names: List[str],
    types: List[dt.DataType],
) -> dd.DataFrame:
    """
    Clone of ibis.util.coerce_to_dataframe that deals well with dask types

    Coerce the following shapes to a DataFrame.

    The following shapes are allowed:
    (1) A list/tuple of Series -> each series is a column
    (2) A list/tuple of scalars -> each scalar is a column
    (3) A Dask Series of list/tuple -> each element inside becomes a column
    (4) dd.DataFrame -> the data is unchanged

    Examples
    --------
    Note: these examples demonstrate functionality with pandas objects in order
    to make them more legible, but this works the same with dask.

    >>> coerce_to_dataframe(pd.DataFrame({'a': [1, 2, 3]}), ['b'])
       b
    0  1
    1  2
    2  3
    >>> coerce_to_dataframe(pd.Series([[1, 2, 3]]), ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    >>> coerce_to_dataframe(pd.Series([range(3), range(3)]), ['a', 'b', 'c'])
       a  b  c
    0  0  1  2
    1  0  1  2
    >>> coerce_to_dataframe([pd.Series(x) for x in [1, 2, 3]], ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    >>>  coerce_to_dataframe([1, 2, 3], ['a', 'b', 'c'])
       a  b  c
    0  1  2  3
    """
    if isinstance(data, dd.DataFrame):
        result = data

    elif isinstance(data, dd.Series):
        # This takes a series where the values are iterables and converts each
        # value into its own row in a new dataframe.

        # NOTE - We add a detailed meta here so we do not drop the key index
        # downstream. This seems to be fixed in versions of dask > 2020.12.0
        dtypes = map(ibis_dtype_to_pandas, types)

        series = [
            data.apply(
                _select_item_in_iter,
                selection=i,
                meta=make_meta_series(dtype, index_name=data.index.name),
            )
            for i, dtype in enumerate(dtypes)
        ]
        result = dd.concat(series, axis=1)

    elif isinstance(data, (tuple, list)):
        if len(data) == 0:
            result = dd.from_pandas(
                pd.DataFrame(columns=column_names), npartitions=1
            )
        elif isinstance(data[0], dd.Series):
            result = dd.concat(data, axis=1)
        else:
            result = dd.from_pandas(
                pd.concat([pd.Series([v]) for v in data], axis=1),
                npartitions=1,
            )
    else:
        raise ValueError(f"Cannot coerce to DataFrame: {data}")

    result.columns = column_names
    return result


def _select_item_in_iter(t, selection):
    return t[selection]


def safe_concat(dfs: List[Union[dd.Series, dd.DataFrame]]) -> dd.DataFrame:
    """
    Concat a list of `dd.Series` or `dd.DataFrame` objects into one DataFrame

    This will use `DataFrame.concat` if all pieces are the same length.
    Otherwise we will iterratively join.

    When axis=1 and divisions are unknown, Dask `DataFrame.concat` can only
    operate on objects with equal lengths, otherwise it will raise a
    ValueError in `concat_and_check`.

    See https://github.com/dask/dask/blob/2c2e837674895cafdb0612be81250ef2657d947e/dask/dataframe/multi.py#L907 # noqa

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
    timecontext: Optional[TimeContext] = None,
    scope: Scope = None,
    **kwargs,
):
    """
    Note - we use this function instead of the pandas.execution.util so that we
    use the dask `execute` method

    This function borrows the logic in the pandas backend. ``by`` can be a
    string or an expression. If ``by.get_name()`` raises an exception, we must
    ``execute`` the expression and sort by the new derived column.
    """
    by = key.to_expr()
    name = ibis.util.guid()
    try:
        if isinstance(by, str):
            return name, data[by]
        return name, data[by.get_name()]
    except com.ExpressionError:
        if scope is None:
            scope = Scope()
        scope = scope.merge_scopes(
            Scope({t: data}, timecontext) for t in by.op().root_tables()
        )
        new_column = execute(by, scope=scope, **kwargs)
        new_column.name = name
        return name, new_column


def compute_sorted_frame(
    df: dd.DataFrame,
    order_by: ir.SortExpr,
    timecontext: Optional[TimeContext] = None,
    **kwargs,
) -> dd.DataFrame:
    sort_col_name, temporary_column = compute_sort_key(
        order_by.op(), df, timecontext, **kwargs
    )
    result = df.assign(**{sort_col_name: temporary_column})
    result = result.set_index(sort_col_name).reset_index(drop=True)
    return result


def assert_identical_grouping_keys(*args):
    indices = [arg.index for arg in args]
    # Depending on whether groupby was called like groupby("col") or
    # groupby(["cold"]) index will be a string or a list
    if isinstance(indices[0], list):
        indices = [tuple(index) for index in indices]
    grouping_keys = set(indices)
    if len(grouping_keys) != 1:
        raise AssertionError(
            f"Differing grouping keys passed: {grouping_keys}"
        )


def safe_scalar_type(output_meta):
    """
    Patch until https://github.com/dask/dask/pull/7627 is merged and that
    version of dask is used in ibis
    """
    if isinstance(output_meta, pd.DatetimeTZDtype):
        output_meta = pd.Timestamp(1, tz=output_meta.tz, unit=output_meta.unit)

    return output_meta


def add_partitioned_sorted_column(
    df: Union[dd.DataFrame, dd.Series],
) -> dd.DataFrame:
    """Add a column that is already partitioned and sorted
    This columns acts as if we had a global index across the distributed data.
    Important properties:
    - Each row has a unique id (i.e. a value in this column)
    - IDs within each partition are already sorted
    - Any id in partition N_{t} is less than any id in partition N_{t+1}
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
    >>> ddf
    Dask DataFrame Structure:
                    a
    npartitions=2
    0              int64
    2                ...
    3                ...
    Dask Name: from_pandas, 2 task
    >>> ddf.compute()
       a
    0  1
    1  2
    2  3
    3  4
    >>> ddf = add_partitioned_sorted_column(ddf)
    >>> ddf
    Dask DataFrame Structure:
                    a
    npartitions=2
    0              int64
    4294967296       ...
    8589934592       ...
    Dask Name: set_index, 8 tasks
    Name: result, dtype: int64
    >>> ddf.compute()
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
        df: Union[pd.Series, pd.DataFrame],
        partition_info: Dict[str, Any],  # automatically injected by dask
        col_name: str,
    ):
        """Assigns a column with a unique id for each row"""
        if len(df) > (2 ** 31):
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


def dask_array_select(condlist, choicelist, default=0):
    # shim taken from dask.array 2021.6.1
    # https://github.com/ibis-project/ibis/issues/2847
    try:
        from dask.array import select

        return select(condlist, choicelist, default)
    except ImportError:

        def _select(*args, **kwargs):
            split_at = len(args) // 2
            condlist = args[:split_at]
            choicelist = args[split_at:]
            return np.select(condlist, choicelist, **kwargs)

        if len(condlist) != len(choicelist):
            raise ValueError(
                "list of cases must be same length as list of conditions"
            )

        if len(condlist) == 0:
            raise ValueError(
                "select with an empty condition list is not possible"
            )

        choicelist = [da.asarray(choice) for choice in choicelist]

        try:
            intermediate_dtype = da.result_type(*choicelist)
        except TypeError as e:
            msg = "Choicelist elements do not have a common dtype."
            raise TypeError(msg) from e

        blockwise_shape = tuple(range(choicelist[0].ndim))

        condargs = [
            arg for elem in condlist for arg in (elem, blockwise_shape)
        ]
        choiceargs = [
            arg for elem in choicelist for arg in (elem, blockwise_shape)
        ]

        return da.blockwise(
            _select,
            blockwise_shape,
            *condargs,
            *choiceargs,
            dtype=intermediate_dtype,
            name="select",
            default=default,
        )
