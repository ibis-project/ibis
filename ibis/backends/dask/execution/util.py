from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dask.dataframe as dd
import pandas as pd
from dask.dataframe.groupby import SeriesGroupBy

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.pandas.trace import TraceTwoLevelDispatcher
from ibis.expr import types as ir
from ibis.expr.scope import Scope
from ibis.expr.typing import TimeContext

from ..core import execute

DispatchRule = Tuple[Tuple[Type], Callable]

TypeRegistrationDict = Dict[ops.Node, List[DispatchRule]]


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


def make_selected_obj(gs: SeriesGroupBy):
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


def maybe_wrap_scalar(result: Any, expr: ir.Expr) -> Any:
    """
    A partial implementation of `coerce_to_output` in the pandas backend.

    Currently only wraps scalars, but will change when udfs are added to the
    dask backend.
    """
    result_name = expr.get_name()
    if isinstance(result, dd.core.Scalar) and isinstance(
        expr.op(), ops.Reduction
    ):
        # TODO - computation
        return dd.from_pandas(
            pd.Series(result.compute(), name=result_name), npartitions=1
        )
    else:
        return result.rename(result_name)


def safe_concat(dfs: List[Union[dd.Series, dd.DataFrame]]) -> dd.DataFrame:
    """
    Concat a list of `dd.Series` or `dd.DataFrame` objects into one DataFrame

    This will use `DataFrame.concat` if all pieces are the same length.
    Otherwise we will iterratively join.

    When axis=1 and divisions are unknown, Dask `DataFrame.concat` can only
    operate on objects with equal lengths, otherwise it will raise a
    ValueError in `concat_and_check`.

    See https://github.com/dask/dask/blob/2c2e837674895cafdb0612be81250ef2657d947e/dask/dataframe/multi.py#L907 # noqa

    Note - this is likely to be quite slow, but this should be hit rarely in
    real usage. A situtation that triggeres this slow path is aggregations
    where aggregations return different numbers of rows (see
    `test_aggregation_group_by` for a specific example).

    TODO - performance.
    """
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
    Note - we use this function instead of the pandas.execution.util so that
    we use the dask `execute` method
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
