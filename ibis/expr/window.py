# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.util as util


class Window(object):

    """
    A generic window function clause, patterned after SQL window clauses for
    the time being. Can be expanded to cover more use cases as they arise.

    Using None for preceding or following currently indicates unbounded. Use 0
    for current_value
    """

    def __init__(self, group_by=None, order_by=None,
                 preceding=None, following=None):
        if group_by is None:
            group_by = []

        if order_by is None:
            order_by = []

        self._group_by = util.promote_list(group_by)
        self._order_by = util.promote_list(order_by)
        self._order_by = [ops.SortKey(expr)
                         if isinstance(expr, ir.Expr)
                         else expr
                         for expr in self._order_by]

        self.preceding = preceding
        self.following = following

        self._validate_frame()

    def _validate_frame(self):
        pass

    def bind(self, table):
        # Internal API, ensure that any unresolved expr references (as strings,
        # say) are bound to the table being windowed
        groups = table._resolve(self._group_by)
        sorts = [ops.to_sort_key(table, k) for k in self._order_by]
        return self._replace(group_by=groups, order_by=sorts)

    def combine(self, window):
        kwds = dict(
            preceding=self.preceding or window.preceding,
            following=self.following or window.following,
            group_by=self._group_by + window._group_by,
            order_by=self._order_by + window._order_by
        )
        return Window(**kwds)

    def group_by(self, expr):
        new_groups = self._group_by + [expr]
        return self._replace(group_by=new_groups)

    def _replace(self, **kwds):
        new_kwds = dict(
            group_by=kwds.get('group_by', self._group_by),
            order_by=kwds.get('order_by', self._order_by),
            preceding=kwds.get('preceding', self.preceding),
            following=kwds.get('following', self.following)
        )
        return Window(**new_kwds)

    def order_by(self, expr):
        new_sorts = self._order_by + [expr]
        return self._replace(order_by=new_sorts)

    def equals(self, other):
        if not isinstance(other, Window):
            return False

        if (len(self._group_by) != len(other._group_by)
                or not ir.all_equal(self._group_by, other._group_by)):
            return False

        if (len(self._order_by) != len(other._order_by)
                or not ir.all_equal(self._order_by, other._order_by)):
            return False

        return (self.preceding == other.preceding
                and self.following == other.following)


def window(preceding=None, following=None, group_by=None, order_by=None):
    """

    """
    return Window(preceding=preceding, following=following,
                  group_by=group_by, order_by=order_by)


def cumulative_window(group_by=None, order_by=None):
    """

    """
    return Window(preceding=None, following=0,
                  group_by=group_by, order_by=order_by)


def trailing_window(periods, group_by=None, order_by=None):
    """

    """
    return Window(preceding=periods, following=0,
                  group_by=group_by, order_by=order_by)
