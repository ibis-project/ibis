import operator

import dask.dataframe as dd
from pandas import Timedelta

import ibis.expr.operations as ops
import ibis.util
from ibis.dask.core import execute
from ibis.dask.dispatch import execute_node
from ibis.dask.execution import constants


def _compute_join_column(column_expr, **kwargs):
    column_op = column_expr.op()

    if isinstance(column_op, ops.TableColumn):
        new_column = column_op.name
    else:
        new_column = execute(column_expr, **kwargs)
    (root_table,) = column_op.root_tables()
    return new_column, root_table


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
    key = "cross_join_{}".format(ibis.util.guid())
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


@execute_node.register(ops.Join, dd.DataFrame, dd.DataFrame)
def execute_materialized_join(op, left, right, **kwargs):
    op_type = type(op)

    try:
        how = constants.JOIN_TYPES[op_type]
    except KeyError:
        raise NotImplementedError('{} not supported'.format(op_type.__name__))

    left_op = op.left.op()
    right_op = op.right.op()

    on = {left_op: [], right_op: []}
    # TODO - logic here is broken
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


def _extract_predicate_names(predicates):
    lefts = []
    rights = []
    for predicate in map(operator.methodcaller('op'), predicates):
        if not isinstance(predicate, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with dask'
            )
        left_name = predicate.left._name
        right_name = predicate.right._name
        lefts.append(left_name)
        rights.append(right_name)
    return lefts, rights


def _validate_columns(orig_columns, *key_lists):
    overlapping_columns = orig_columns.difference(
        item for sublist in key_lists for item in sublist
    )
    if overlapping_columns:
        raise ValueError(
            'left and right DataFrame columns overlap on {} in a join. '
            'Please specify the columns you want to select from the join, '
            'e.g., join[left.column1, right.column2, ...]'.format(
                overlapping_columns
            )
        )
