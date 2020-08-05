""" Module for scope
    Scope is a dictionary mapping :class:`~ibis.expr.operations.Node`
    subclass instances to concrete data, and the time context associate
    with it(if any).
    Scope is used in many backends to cache data for calculated ops.
"""
from typing import Optional

from ibis.expr.typing import TimeContext
from ibis.timecontext.util import TimeContextRelation, compare_timecontext


def set_scope_item(op, result, timecontext: Optional[TimeContext]):
    """make a scope item to be set in scope

    Parameters
    ----------
    op: ibis.expr.operations.Node, key in scope.
    result : scalar, pd.Series, pd.DataFrame, concrete data.
    timecontext: Optional[TimeContext], time context associate with the
    result.

    Returns
    -------
    set_scope_item : Dict, a key value pair that could merge into scope later
    """
    return {op: {'value': result, 'timecontext': timecontext}}


def get_scope_item(scope, op, timecontext: Optional[TimeContext] = None):
    """ Given a op and timecontext, get result from scope
    Parameters
    ----------
    scope: collections.Mapping
    a dictionary mapping :class:`~ibis.expr.operations.Node`
    subclass instances to concrete data, and the time context associate
    with it(if any).
    op: ibis.expr.operations.Node, key in scope.
    timecontext: Optional[TimeContext]

    Returns
    -------
    result : scalar, pd.Series, pd.DataFrame
    """
    if op not in scope:
        return None
    # for ops without timecontext
    if timecontext is None:
        return scope[op].get('value', None)
    else:
        # For op with timecontext, ther are some ops cannot use cached result
        # with a different (larger) timecontext to get the correct result.
        # For example, a groupby followed by count, if we use a larger or
        # smaller dataset from cache, we will probably get an error in result.
        # Such ops with global aggregation, ops whose result is depending on
        # other rows in result Dataframe, cannot use cached result with
        # different time context to optimize calculation. These are time
        # context sensitive operations. Since these cases are rare in
        # acutal use case, we just enable optimization for all nodes for now.
        old_timecontext = scope[op].get('timecontext', None)
        if old_timecontext:
            relation = compare_timecontext(timecontext, old_timecontext)
            if relation == TimeContextRelation.SUBSET:
                return scope[op].get('value', None)
    return None
