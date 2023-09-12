from __future__ import annotations

import dask.dataframe as dd
from pandas import Timedelta

import ibis.expr.operations as ops
import ibis.util
from ibis.backends.dask.dispatch import execute_node
from ibis.backends.dask.execution import constants
from ibis.backends.pandas.execution.join import (
    _compute_join_column,
    _extract_predicate_names,
)


@execute_node.register(
    ops.AsOfJoin,
    dd.DataFrame,
    dd.DataFrame,
    tuple,
    (Timedelta, type(None)),
    tuple,
)
def execute_asof_join(op, left, right, by, tolerance, predicates, **kwargs):
    left_on, right_on = _extract_predicate_names(predicates)
    left_by, right_by = _extract_predicate_names(by)

    assert 0 <= len(left_on) <= 1, f"len(left_on) == {len(left_on)}"
    assert 0 <= len(right_on) <= 1, f"len(right_on) == {len(right_on)}"

    on = left_on if left_on == right_on else None
    return dd.merge_asof(
        left=left,
        right=right,
        # NB: dask 2022.4.1 contains a bug from
        # https://github.com/dask/dask/pull/8857 that keeps a column if `on` is
        # non-empty without checking whether `left_on` is non-empty, this
        # check works around that
        on=on,
        left_on=left_on if on is None else None,
        right_on=right_on if on is None else None,
        left_by=left_by or None,
        right_by=right_by or None,
        tolerance=tolerance,
        suffixes=constants.JOIN_SUFFIXES,
    )


@execute_node.register(ops.CrossJoin, dd.DataFrame, dd.DataFrame, tuple)
def execute_cross_join(op, left, right, predicates, **kwargs):
    """Execute a cross join in dask.

    Notes
    -----
    We create a dummy column of all :data:`True` instances and use that as the
    join key. This results in the desired Cartesian product behavior guaranteed
    by cross join.
    """
    assert not predicates, "cross join should have an empty predicate set"
    # generate a unique name for the temporary join key
    key = f"cross_join_{ibis.util.guid()}"
    join_key = {key: True}
    new_left = left.assign(**join_key)
    new_right = right.assign(**join_key)

    # inner/outer doesn't matter because every row matches every other row
    result = dd.merge(
        new_left,
        new_right,
        how="inner",
        on=key,
        suffixes=constants.JOIN_SUFFIXES,
    )

    # remove the generated key
    del result[key]

    return result


# TODO - execute_join - #2553
@execute_node.register(ops.Join, dd.DataFrame, dd.DataFrame, tuple)
def execute_join(op, left, right, predicates, **kwargs):
    op_type = type(op)

    try:
        how = constants.JOIN_TYPES[op_type]
    except KeyError:
        raise NotImplementedError(f"{op_type.__name__} not supported")

    on = {op.left: [], op.right: []}
    for predicate in predicates:
        if not isinstance(predicate, ops.Equals):
            raise TypeError("Only equality join predicates supported with dask")
        new_left_column, left_pred_root = _compute_join_column(predicate.left, **kwargs)
        on[left_pred_root].append(new_left_column)

        new_right_column, right_pred_root = _compute_join_column(
            predicate.right, **kwargs
        )
        on[right_pred_root].append(new_right_column)

    df = dd.merge(
        left,
        right,
        how=how,
        left_on=on[op.left],
        right_on=on[op.right],
        suffixes=constants.JOIN_SUFFIXES,
    )
    return df
