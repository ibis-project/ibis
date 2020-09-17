""" Module for scope
    The motivation of Scope is to cache data for calculated ops.

    `scope` in Scope class is the main cache. It is a dictionary mapping
    ibis node instances to concrete data, and the time context associate
    with it (if any).

    When there are no time contexts associate with the cached result, getting
    and setting values in Scope would be as simple as get and set in a normal
    dictonary. With time contexts, we need the following logic for getting
    and setting items in scope:

    Before setting the value op in scope we need to perform the following
    check first:

    Test if `op` is in `scope` yet
    - No, then put `op` in `scope`, set 'timecontext' to be the current
    `timecontext` (None if `timecontext` is not present), set 'value' to be
    the actual data.
    - Yes, then get the time context stored in `scope` for `op` as
    `old_timecontext`, and compare it with current `timecontext`:
    If current `timecontext` is a subset of `_timecontext`, that means we
    already cached a larger range of data. Do nothing and we will trim data in
    later execution process.
    If current `timecontext` is a superset of `old_timecontext`, that means we
    need to update cache. Set 'value' to be the current data and set
    'timecontext' to be the current `timecontext` for `op`.
    If current `timecontext` is neither a subset nor a superset of
    `old_timcontext`, but they overlap, or not overlap at all (For example
    when there is a window that looks forward, over a window that looks
    back), in this case, we should not trust the data stored either because
    the data stored in scope doesn't cover the current time context.
    For simplicity, we update cache in this case, instead of merge data of
    different time contexts.

"""
from collections import namedtuple
from typing import Any, Dict, List, Optional

from ibis.expr.operations import Node
from ibis.expr.timecontext import TimeContextRelation, compare_timecontext
from ibis.expr.typing import TimeContext

ScopeItem = namedtuple('item', ['value', 'timecontext'])


class Scope:
    def __init__(self, items: Dict[str, ScopeItem] = None):
        self._items = items or {}

    def __contains__(self, op):
        return op in self._items

    def __iter__(self):
        return iter(self._items.keys())

    def get_value(
        self, op: Node, timecontext: Optional[TimeContext] = None
    ) -> Any:
        """ Given a op and timecontext, get the result from scope

        Parameters
        ----------
        scope : collections.Mapping
            a dictionary mapping :class:`~ibis.expr.operations.Node`
            subclass instances to concrete data, and the time context associate
            with it (if any).
        op: ibis.expr.operations.Node
            key in scope.
        timecontext: Optional[TimeContext]

        Returns
        -------
        result: the cached result, an object whose types may differ in
        different backends.
        """
        if op not in self:
            return None

        # for ops without timecontext
        if timecontext is None:
            return self._items[op].value
        else:
            # For op with timecontext, ther are some ops cannot use cached
            # result with a different (larger) timecontext to get the
            # correct result.
            # For example, a groupby followed by count, if we use a larger or
            # smaller dataset from cache, we will get an error in result.
            # Such ops with global aggregation, ops whose result is
            # depending on other rows in result Dataframe, cannot use cached
            # result with different time context to optimize calculation.
            # These are time context sensitive operations. Since these cases
            # are rare in acutal use case, we just enable optimization for
            # all nodes for now.
            cached_timecontext = self._items[op].timecontext
            if cached_timecontext:
                relation = compare_timecontext(timecontext, cached_timecontext)
                if relation == TimeContextRelation.SUBSET:
                    return self._items[op].value
            else:
                return self._items[op].value
        return None

    def merge_scope(self, other_scope: 'Scope', overwrite=False) -> 'Scope':
        """merge items in other_scope into this scope

        Parameters
        ----------
        other_scope: Scope
            Scope to be merged with
        overwrite: bool
            if set to be True, force overwrite `value` if `op` already
            exists.

        Returns
        -------
        Scope
            a new Scope instance with items in two scope merged.
        """
        result = Scope()

        for op in self:
            result._items[op] = self._items[op]

        for op in other_scope:
            # if get_scope returns a not None value, then data is already
            # cached in scope and it is at least a greater range than
            # the current timecontext, so we drop the item. Otherwise
            # add it into scope.
            v = other_scope._items[op]
            if overwrite or result.get_value(op, v.timecontext) is None:
                result._items[op] = v
        return result

    def merge_scopes(
        self, other_scopes: List['Scope'], overwrite=False
    ) -> 'Scope':
        """merge items in other_scopes into this scope

        Parameters
        ----------
        other_scopes: List[Scope]
            scopes to be merged with
        overwrite: Bool
            if set to be True, force overwrite value if op already exists.

        Returns
        -------
        Scope
            a new Scope instance with items in two scope merged.
        """
        result = Scope()
        for op in self:
            result._items[op] = self._items[op]

        for s in other_scopes:
            result = result.merge_scope(s, overwrite)
        return result


def make_scope(
    op: Node, result: Any, timecontext: Optional[TimeContext] = None
) -> 'Scope':
    """make a Scope instance, adding (op, result, timecontext) into the
       scope

    Parameters
    ----------
    op: ibis.expr.operations.Node
        key in scope.
    result : Object
        concrete data, type could be different for different backends.
    timecontext: Optional[TimeContext]
        time context associate with the result.

    Returns
    -------
    Scope
        a new Scope instance with op in it.
    """
    return Scope({op: ScopeItem(result, timecontext)})
