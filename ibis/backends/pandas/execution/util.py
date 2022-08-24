from typing import Any, Optional, Union

import pandas as pd
import toolz

import ibis.common.exceptions as com
import ibis.expr.analysis as an
import ibis.util
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.execution import constants
from ibis.expr import operations as ops
from ibis.expr.scope import Scope


def get_join_suffix_for_op(op: ops.TableColumn, join_op: ops.Join):
    (root_table,) = an.find_immediate_parent_tables(op)
    left_root, right_root = an.find_immediate_parent_tables(
        [join_op.left, join_op.right]
    )
    return {
        left_root: constants.LEFT_JOIN_SUFFIX,
        right_root: constants.RIGHT_JOIN_SUFFIX,
    }[root_table]


def compute_sort_key(key, data, timecontext, scope=None, **kwargs):
    try:
        if isinstance(key, str):
            return key, None
        return key.resolve_name(), None
    except com.ExpressionError:
        if scope is None:
            scope = Scope()
        scope = scope.merge_scopes(
            Scope({t: data}, timecontext)
            for t in an.find_immediate_parent_tables(key)
        )
        new_column = execute(key, scope=scope, **kwargs)
        name = ibis.util.guid()
        new_column.name = name
        return name, new_column


def compute_sorted_frame(
    df, order_by, group_by=(), timecontext=None, **kwargs
):
    computed_sort_keys = []
    sort_keys = list(toolz.concatv(group_by, order_by))
    ascending = [getattr(key, "ascending", True) for key in sort_keys]
    new_columns = {}

    for i, key in enumerate(sort_keys):
        computed_sort_key, temporary_column = compute_sort_key(
            key, df, timecontext, **kwargs
        )
        computed_sort_keys.append(computed_sort_key)

        if temporary_column is not None:
            new_columns[computed_sort_key] = temporary_column

    result = df.assign(**new_columns)
    result = result.sort_values(
        computed_sort_keys, ascending=ascending, kind='mergesort'
    )
    # TODO: we'll eventually need to return this frame with the temporary
    # columns and drop them in the caller (maybe using post_execute?)
    ngrouping_keys = len(group_by)
    return (
        result,
        computed_sort_keys[:ngrouping_keys],
        computed_sort_keys[ngrouping_keys:],
    )


def coerce_to_output(
    result: Any, node: ops.Node, index: Optional[pd.Index] = None
) -> Union[pd.Series, pd.DataFrame]:
    """Cast the result to either a Series or DataFrame.

    This method casts result of an execution to a Series or DataFrame,
    depending on the type of the expression and shape of the result.

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
    result: A Series or DataFrame

    Examples
    --------
    For dataframe outputs, see ``ibis.util.coerce_to_dataframe``.

    >>> coerce_to_output(pd.Series(1), node)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, node)
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, node, [1,2,3])
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1,2,3], node)
    0    [1, 2, 3]
    Name: result, dtype: object
    """
    result_name = node.resolve_name()

    if isinstance(result, pd.DataFrame):
        rows = result.to_dict(orient="records")
        return pd.Series(rows, name=result_name)

    # columnar result
    if isinstance(result, pd.Series):
        return result.rename(result_name)

    # Wrap `result` into a single-element Series.
    return pd.Series([result], name=result_name)
