""" Module for scope
    The motivation of Scope is to cache data for calculated ops.

    `scope` in Scope class is the main cache. It is a dictionary mapping
    ibis node instances to concrete data, and the time context associate
    with it(if any).

    `scope` has the structure of:
    {
        op[ibis.expr.operations.Node]: {
            "value": v[Object]
            "timecontext": t[TimeContext]
        }
        ...
    }

    `scope` uses `op` as the key, `v` is the cached result of executing `op`,
    the type of the result may differ in different backends. `t` is the time
    context associate with the result, set it to None if there is no time
    context in execution.

    When there are no time contexts associate with the cached result, getting
    and setting values in Scope would be as simple as get and set in a normal
    dictonary. With time contexts, we need the following logic for getting
    and setting items in scope:

    Set scope kv pair: before setting the value op in scope we need to perform
    the following check first:

    Test if op is in scope yet
    - No, then put op in scope, set timecontext to be the current timecontext
    (None if timecontext is not present), set value to be the DataFrame or
    Series of the actual data.
    - Yes, then get the timecontext stored in scope for op as old_timecontext,
    and compare it with current timecontext:
    If current timecontext is a subset of old timecontext, that means we
    already cached a larger range of data. Do nothing and we will trim data in
    later execution process.
    If current timecontext is a superset of old timecontext, that means we
    need to update cache. Set value to be the current data and set timecontext
    to be the current timecontext for op.
    If current timecontext is neither a subset nor a superset of old
    timcontext, but they overlap, or not overlap at all. For example this will
    happen when there is a window that looks forward, over a window that looks
    back. So in this case, we should not trust the data stored either, and go
    on to execute this node. For simplicity, we update cache in this case as
    well.

"""
from typing import Optional

from ibis.expr.timecontext import TimeContextRelation, compare_timecontext
from ibis.expr.typing import TimeContext


class Scope:
    def __init__(self, items):
        self.items = items

    def _get_items(self):
        """ Get all items in scope.
        """
        return self.items

    @staticmethod
    def make_scope(op, result, timecontext: Optional[TimeContext]):
        """make a Scope instance, adding (op, result, timecontext) into the
           scope

        Parameters
        ----------
        op: ibis.expr.operations.Node, key in scope.
        result : scalar, pd.Series, pd.DataFrame, concrete data.
        timecontext: Optional[TimeContext], time context associate with the
        result.

        Returns
        -------
        Scope: a new Scope instance with op in it.
        """
        return Scope({op: {'value': result, 'timecontext': timecontext}})

    @staticmethod
    def from_scope(other_scope):
        """make a Scope instance, copying other_scope
        """
        scope = Scope({})
        scope.merge_scope(other_scope)
        return scope

    def get(self, op, timecontext: Optional[TimeContext] = None):
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
        result: the cached result, an object whose types may differ in
        different backends.
        """
        if op not in self.items:
            return None
        # for ops without timecontext
        if timecontext is None:
            return self.items[op].get('value', None)
        else:
            # For op with timecontext, ther are some ops cannot use cached
            # result with a different (larger) timecontext to get the
            # correct result.
            # For example, a groupby followed by count, if we use a larger or
            # smaller dataset from cache, we will probably get an error in
            # result. Such ops with global aggregation, ops whose result is
            # depending on other rows in result Dataframe, cannot use cached
            # result with different time context to optimize calculation.
            # These are time context sensitive operations. Since these cases
            # are rare in acutal use case, we just enable optimization for
            # all nodes for now.
            cached_timecontext = self.items[op].get('timecontext', None)
            if cached_timecontext:
                relation = compare_timecontext(timecontext, cached_timecontext)
                if relation == TimeContextRelation.SUBSET:
                    return self.items[op].get('value', None)
            else:
                return self.items[op].get('value', None)
        return None

    def merge_scope(self, other_scope, overwrite=False):
        """merge items in other_scope into this scope

        Parameters
        ----------
        other_scope: Scope to be merged with
        overwrite: bool, if set to be True, force overwrite value if op
            already exists.
        """
        for op, v in other_scope._get_items().items():
            # if get_scope returns a not None value, then data is already
            # cached in scope and it is at least a greater range than
            # the current timecontext, so we drop the item. Otherwise
            # add it into scope.
            if overwrite or self.get(op, v['timecontext']) is None:
                self.items[op] = v
