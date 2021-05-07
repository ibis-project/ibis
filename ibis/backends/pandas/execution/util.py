import operator
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import toolz

import ibis.common.exceptions as com
import ibis.util
from ibis.expr import operations as ops
from ibis.expr import types as ir
from ibis.expr.scope import Scope

from ..core import execute


def compute_sort_key(key, data, timecontext, scope=None, **kwargs):
    by = key.to_expr()
    try:
        if isinstance(by, str):
            return by, None
        return by.get_name(), None
    except com.ExpressionError:
        if scope is None:
            scope = Scope()
        scope = scope.merge_scopes(
            Scope({t: data}, timecontext) for t in by.op().root_tables()
        )
        new_column = execute(by, scope=scope, **kwargs)
        name = ibis.util.guid()
        new_column.name = name
        return name, new_column


def compute_sorted_frame(
    df, order_by, group_by=(), timecontext=None, **kwargs
):
    computed_sort_keys = []
    sort_keys = list(toolz.concatv(group_by, order_by))
    ascending = [getattr(key.op(), 'ascending', True) for key in sort_keys]
    new_columns = {}

    for i, key in enumerate(map(operator.methodcaller('op'), sort_keys)):
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
    result: Any, expr: ir.Expr, index: Optional[pd.Index] = None
) -> Union[pd.Series, pd.DataFrame]:
    """ Cast the result to either a Series or DataFrame.

    This method casts result of an execution to a Series or DataFrame,
    depending on the type of the expression and shape of the result.

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
    result: A Series or DataFrame
    """
    result_name = getattr(expr, '_name', None)

    if isinstance(expr, (ir.DestructColumn, ir.StructColumn)):
        return ibis.util.coerce_to_dataframe(result, expr.type().names)
    elif isinstance(expr, (ir.DestructScalar, ir.StructScalar)):
        # Here there are two cases, if this is groupby aggregate,
        # then the result e a Series of tuple/list, or
        # if this is non grouped aggregate, then the result
        return ibis.util.coerce_to_dataframe(result, expr.type().names)
    elif isinstance(result, pd.Series):
        return result.rename(result_name)
    elif isinstance(result, np.ndarray):
        return pd.Series(result, name=result_name)
    elif isinstance(expr.op(), ops.Reduction):
        # We either wrap a scalar into a single element Series
        # or broadcast the scalar to a multi element Series
        if index is None:
            return pd.Series(result, name=result_name)
        else:
            return pd.Series(
                np.repeat(result, len(index)), index=index, name=result_name,
            )
    else:
        raise ValueError(f"Cannot coerce_to_output. Result: {result}")
