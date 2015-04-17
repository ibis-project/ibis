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

# User API for grouped data operations

import ibis.expr.types as ir
import ibis.util as util


class GroupedTableExpr(object):

    """
    Helper intermediate construct
    """

    def __init__(self, table, by, having=None):
        if not isinstance(by, (list, tuple)):
            if not isinstance(by, ir.Expr):
                by = table._resolve([by])
            else:
                by = [by]
        else:
            by = table._resolve(by)

        self.table = table
        self.by = by
        self._having = having or []

    def __getattr__(self, attr):
        if hasattr(self.table, attr):
            return self._column_wrapper(attr)

        raise AttributeError("GroupBy has no attribute %r" % attr)

    def _column_wrapper(self, attr):
        col = self.table[attr]
        if isinstance(col, ir.NumericValue):
            return GroupedNumbers(col, self)
        else:
            return GroupedArray(col, self)

    def aggregate(self, metrics):
        return self.table.aggregate(metrics, by=self.by,
                                    having=self._having)

    def having(self, expr):
        """
        Add a post-aggregation result filter (like the having argument in
        `aggregate`), for composability with the group_by API

        Returns
        -------
        grouped : GroupedTableExpr
        """
        exprs = util.promote_list(expr)
        new_having = self._having + exprs
        return GroupedTableExpr(self.table, self.by, having=new_having)

    def count(self, metric_name='count'):
        """
        Convenience function for computing the group sizes (number of rows per
        group) given a grouped table.

        Parameters
        ----------
        metric_name : string, default 'count'
          Name to use for the row count metric

        Returns
        -------
        aggregated : TableExpr
          The aggregated table
        """
        metric = self.table.count().name(metric_name)
        return self.table.aggregate([metric], by=self.by)

    size = count


def _group_agg_dispatch(name):
    def wrapper(self, *args, **kwargs):
        f = getattr(self.arr, name)
        metric = f(*args, **kwargs)
        alias = '{}({})'.format(name, self.arr.get_name())
        return self.parent.aggregate(metric.name(alias))

    wrapper.__name__ = name
    return wrapper


class GroupedArray(object):

    def __init__(self, arr, parent):
        self.arr = arr
        self.parent = parent

    count = _group_agg_dispatch('count')
    size = count
    min = _group_agg_dispatch('min')
    max = _group_agg_dispatch('max')
    approx_nunique = _group_agg_dispatch('approx_nunique')
    approx_median = _group_agg_dispatch('approx_median')
    group_concat = _group_agg_dispatch('group_concat')


class GroupedNumbers(GroupedArray):

    mean = _group_agg_dispatch('mean')
    sum = _group_agg_dispatch('sum')
