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

ScopeItem = namedtuple('item', ['timecontext', 'value'])


class Scope:
    def __init__(
        self,
        param: Dict[Node, Any] = None,
        timecontext: Optional[TimeContext] = None,
    ):
        """ Take a dict of `op`, `result`, create a new scope and save
        those pairs in scope. Associate None as timecontext by default.
        This is mostly used to init a scope with a set of given params.
        """
        self._items = (
            {op: ScopeItem(timecontext, value) for op, value in param.items()}
            if param
            else {}
        )

    def __contains__(self, op):
        """ Given an `op`, return if `op` is present in Scope.
        Note that this `__contain__` method doesn't take `timecontext`
        as a parameter. This could be used to iterate all keys in
        current scope, or any case that doesn't care about value, just
        simply test if `op` is in scope or not.
        When trying to get value in scope, use `get_value(op, timecontext)`
        instead. Because the cached data could be trusted only if:
        1. `op` is in `scope`, and,
        2. The `timecontext` associated with `op` is a time context equal
           to, or larger than the current time context.
        """
        return op in self._items

    def __iter__(self):
        return iter(self._items.keys())

    def set_value(
        self, op: Node, timecontext: Optional[TimeContext], value: Any
    ) -> None:
        """ Set values in scope.

            Given an `op`, `timecontext` and `value`, set `op` and
            `(value, timecontext)` in scope.

        Parameters
        ----------
        scope : collections.Mapping
            a dictionary mapping :class:`~ibis.expr.operations.Node`
            subclass instances to concrete data, and the time context associate
            with it (if any).
        op: ibis.expr.operations.Node
            key in scope.
        timecontext: Optional[TimeContext]
        value: Any
            the cached result to save in scope, an object whose type may
            differ in different backends.
        """
        # Note that this set method doesn't simply override and set, but
        # takes time context into consideration.
        # If there is a value associated with the key, but time context is
        # smaller than the current time context we are going to set,
        # `get_value` will return None and we will proceed to set the new
        # value in scope.
        if self.get_value(op, timecontext) is None:
            self._items[op] = ScopeItem(timecontext, value)

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
