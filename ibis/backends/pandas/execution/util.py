from __future__ import annotations

from typing import Any

import pandas as pd

import ibis.expr.analysis as an
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.base.df.scope import Scope
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.execution import constants


def get_grouping(grouper):
    # this is such an annoying hack
    assert isinstance(grouper, list)
    if len(grouper) == 1:
        return grouper[0]
    return grouper


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
    if key.shape.is_columnar():
        if key.name in data:
            return key.name, None
        else:
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
    else:
        raise NotImplementedError(
            "Scalar sort keys are not yet supported in the pandas backend"
        )


def compute_sorted_frame(df, order_by, group_by=(), timecontext=None, **kwargs):
    sort_keys = []
    ascending = []

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


def coerce_to_output(
    result: Any, node: ops.Node, index: pd.Index | None = None
) -> pd.Series | pd.DataFrame:
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

    >>> coerce_to_output(pd.Series(1), node)  # quartodoc: +SKIP # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, node)  # quartodoc: +SKIP # doctest: +SKIP
    0    1
    Name: result, dtype: int64
    >>> coerce_to_output(1, node, [1, 2, 3])  # quartodoc: +SKIP # doctest: +SKIP
    1    1
    2    1
    3    1
    Name: result, dtype: int64
    >>> coerce_to_output([1, 2, 3], node)  # quartodoc: +SKIP # doctest: +SKIP
    0    [1, 2, 3]
    Name: result, dtype: object
    """
    if isinstance(result, pd.DataFrame):
        rows = result.to_dict(orient="records")
        return pd.Series(rows, name=node.name)

    # columnar result
    if isinstance(result, pd.Series):
        return result.rename(node.name)

    # Wrap `result` into a single-element Series.
    return pd.Series([result], name=node.name)
