import operator

import pandas as pd

import ibis.expr.operations as ops
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.dispatch import execute_node
from ibis.backends.pandas.execution import constants


def _compute_join_column(column_expr, **kwargs):
    column_op = column_expr.op()

    if isinstance(column_op, ops.TableColumn):
        new_column = column_op.name
    else:
        new_column = execute(column_expr, **kwargs)
    (root_table,) = column_op.root_tables()
    return new_column, root_table


@execute_node.register(ops.CrossJoin, pd.DataFrame, pd.DataFrame, tuple)
def execute_cross_join(op, left, right, predicates, **kwargs):
    """Execute a cross join in pandas.

    Notes
    -----
    We create a dummy column of all :data:`True` instances and use that as the
    join key. This results in the desired Cartesian product behavior guaranteed
    by cross join.

    """
    assert not predicates, "cross join predicates must be empty"
    return pd.merge(
        left,
        right,
        how='cross',
        copy=False,
        suffixes=constants.JOIN_SUFFIXES,
    )


def _get_semi_anti_join_filter(op, left, right, predicates, **kwargs):
    left_on, right_on = _construct_join_predicate_columns(
        op,
        predicates,
        **kwargs,
    )
    inner = left.merge(
        right[right_on].drop_duplicates(),
        on=left_on,
        how="left",
        indicator=True,
    )
    return (inner["_merge"] == "both").values


@execute_node.register(ops.LeftSemiJoin, pd.DataFrame, pd.DataFrame, tuple)
def execute_left_semi_join(op, left, right, predicates, **kwargs):
    """Execute a left semi join in pandas."""
    inner_filt = _get_semi_anti_join_filter(
        op,
        left,
        right,
        predicates,
        **kwargs,
    )
    return left.loc[inner_filt, :]


@execute_node.register(ops.LeftAntiJoin, pd.DataFrame, pd.DataFrame, tuple)
def execute_left_anti_join(op, left, right, predicates, **kwargs):
    """Execute a left anti join in pandas."""
    inner_filt = _get_semi_anti_join_filter(
        op,
        left,
        right,
        predicates,
        **kwargs,
    )
    return left.loc[~inner_filt, :]


def _construct_join_predicate_columns(op, predicates, **kwargs):
    left_op = op.left.op()
    right_op = op.right.op()

    on = {left_op: [], right_op: []}

    for predicate in map(operator.methodcaller('op'), predicates):
        if not isinstance(predicate, ops.Equals):
            raise TypeError(
                'Only equality join predicates supported with pandas'
            )
        new_left_column, left_pred_root = _compute_join_column(
            predicate.left, **kwargs
        )
        on[left_pred_root].append(new_left_column)

        new_right_column, right_pred_root = _compute_join_column(
            predicate.right, **kwargs
        )
        on[right_pred_root].append(new_right_column)
    return on[left_op], on[right_op]


@execute_node.register(ops.Join, pd.DataFrame, pd.DataFrame, tuple)
def execute_join(op, left, right, predicates, **kwargs):
    op_type = type(op)

    try:
        how = constants.JOIN_TYPES[op_type]
    except KeyError:
        raise NotImplementedError(f'{op_type.__name__} not supported')

    left_on, right_on = _construct_join_predicate_columns(
        op, predicates, **kwargs
    )

    df = pd.merge(
        left,
        right,
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=constants.JOIN_SUFFIXES,
    )
    return df


@execute_node.register(
    ops.AsOfJoin,
    pd.DataFrame,
    pd.DataFrame,
    tuple,
    (pd.Timedelta, type(None)),
    tuple,
)
def execute_asof_join(op, left, right, by, tolerance, predicates, **kwargs):
    overlapping_columns = frozenset(left.columns) & frozenset(right.columns)
    left_on, right_on = _extract_predicate_names(predicates)
    left_by, right_by = _extract_predicate_names(by)
    _validate_columns(
        overlapping_columns, left_on, right_on, left_by, right_by
    )

    return pd.merge_asof(
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
                'Only equality join predicates supported with pandas'
            )
        left_name = predicate.left.get_name()
        right_name = predicate.right.get_name()
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
