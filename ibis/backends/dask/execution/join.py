import operator

import dask.dataframe as dd
from pandas import Timedelta

import ibis.expr.operations as ops
import ibis.util
from ibis.backends.pandas.execution.join import (
    _compute_join_column,
    _extract_predicate_names,
    _validate_columns,
)

from ..dispatch import execute_node
from ..execution import constants


@execute_node.register(ops.MaterializedJoin, dd.DataFrame)
def execute_materialized_join(op, df, **kwargs):
    return df


@execute_node.register(
    ops.AsOfJoin, dd.DataFrame, dd.DataFrame, (Timedelta, type(None))
)
def execute_asof_join(op, left, right, tolerance, **kwargs):
    overlapping_columns = frozenset(left.columns) & frozenset(right.columns)
    left_on, right_on = _extract_predicate_names(op.predicates)
    left_by, right_by = _extract_predicate_names(op.by)
    _validate_columns(
        overlapping_columns, left_on, right_on, left_by, right_by
    )

    return dd.merge_asof(
        left=left,
        right=right,
        left_on=left_on,
        right_on=right_on,
        left_by=left_by or None,
        right_by=right_by or None,
        tolerance=tolerance,
    )


@execute_node.register(ops.CrossJoin, dd.DataFrame, dd.DataFrame)
def execute_cross_join(op, left, right, **kwargs):
    """Execute a cross join in dask.

    Notes
    -----
    We create a dummy column of all :data:`True` instances and use that as the
    join key. This results in the desired Cartesian product behavior guaranteed
    by cross join.

    """
    # generate a unique name for the temporary join key
    key = f"cross_join_{ibis.util.guid()}"
    join_key = {key: True}
    new_left = left.assign(**join_key)
    new_right = right.assign(**join_key)

    # inner/outer doesn't matter because every row matches every other row
    result = dd.merge(
        new_left,
        new_right,
        how='inner',
        on=key,
        suffixes=constants.JOIN_SUFFIXES,
    )

    # remove the generated key
    del result[key]

    return result


# TODO - execute_join - #2553
@execute_node.register(ops.Join, dd.DataFrame, dd.DataFrame)
def execute_join(op, left, right, **kwargs):
    op_type = type(op)

    try:
        how = constants.JOIN_TYPES[op_type]
    except KeyError:
        raise NotImplementedError(f'{op_type.__name__} not supported')

    left_op = op.left.op()
    right_op = op.right.op()

    on = {left_op: [], right_op: []}
    for predicate in map(operator.methodcaller('op'), op.predicates):
        if not isinstance(predicate, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with dask'
            )
        new_left_column, left_pred_root = _compute_join_column(
            predicate.left, **kwargs
        )
        on[left_pred_root].append(new_left_column)

        new_right_column, right_pred_root = _compute_join_column(
            predicate.right, **kwargs
        )
        on[right_pred_root].append(new_right_column)

    df = dd.merge(
        left,
        right,
        how=how,
        left_on=on[left_op],
        right_on=on[right_op],
        suffixes=constants.JOIN_SUFFIXES,
    )
    return df
