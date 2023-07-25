from __future__ import annotations

import itertools

import pandas as pd

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.pandas.core import execute
from ibis.backends.pandas.dispatch import execute_node
from ibis.backends.pandas.execution import constants
from ibis.common.exceptions import UnsupportedOperationError


def _compute_join_column(column, **kwargs):
    if isinstance(column, ops.TableColumn):
        new_column = column.name
    else:
        new_column = execute(column, **kwargs)
    root_table, *_ = an.find_immediate_parent_tables(column)
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
        how="cross",
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
    on = {op.left: [], op.right: []}

    for predicate in predicates:
        if not isinstance(predicate, ops.Equals):
            raise TypeError("Only equality join predicates supported with pandas")
        new_left_column, left_pred_root = _compute_join_column(predicate.left, **kwargs)
        on[left_pred_root].append(new_left_column)

        new_right_column, right_pred_root = _compute_join_column(
            predicate.right, **kwargs
        )
        on[right_pred_root].append(new_right_column)
    return on[op.left], on[op.right]


@execute_node.register(ops.Join, pd.DataFrame, pd.DataFrame, tuple)
def execute_join(op, left, right, predicates, **kwargs):
    op_type = type(op)

    try:
        how = constants.JOIN_TYPES[op_type]
    except KeyError:
        raise UnsupportedOperationError(f"{op_type.__name__} not supported")

    left_on, right_on = _construct_join_predicate_columns(op, predicates, **kwargs)

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
    left_on, right_on = _extract_predicate_names(predicates)
    left_by, right_by = _extract_predicate_names(by)

    # Add default join suffixes to predicates and groups and rename the
    # corresponding columns before the `merge_asof`. If we don't do this and the
    # predicates have the same column name, we lose the original RHS column
    # values in the output. Instead, the RHS values are copies of the LHS values.
    # xref https://github.com/ibis-project/ibis/issues/6080
    left_on_suffixed = [x + constants.JOIN_SUFFIXES[0] for x in left_on]
    right_on_suffixed = [x + constants.JOIN_SUFFIXES[1] for x in right_on]

    left_by_suffixed = [x + constants.JOIN_SUFFIXES[0] for x in left_by]
    right_by_suffixed = [x + constants.JOIN_SUFFIXES[1] for x in right_by]

    left = left.rename(
        columns=dict(
            itertools.chain(
                zip(left_on, left_on_suffixed), zip(left_by, left_by_suffixed)
            )
        )
    )
    right = right.rename(
        columns=dict(
            itertools.chain(
                zip(right_on, right_on_suffixed), zip(right_by, right_by_suffixed)
            )
        )
    )

    return pd.merge_asof(
        left=left,
        right=right,
        left_on=left_on_suffixed,
        right_on=right_on_suffixed,
        left_by=left_by_suffixed or None,
        right_by=right_by_suffixed or None,
        tolerance=tolerance,
        suffixes=constants.JOIN_SUFFIXES,
    )


def _extract_predicate_names(predicates):
    lefts = []
    rights = []
    for predicate in predicates:
        if not isinstance(predicate, ops.Equals):
            raise TypeError("Only equality join predicates supported with pandas")
        left_name = predicate.left.name
        right_name = predicate.right.name
        lefts.append(left_name)
        rights.append(right_name)
    return lefts, rights
