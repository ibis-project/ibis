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

from __future__ import absolute_import

import types

import toolz

import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.expr.window as _window
import ibis.util as util


def _resolve_exprs(table, exprs):
    exprs = util.promote_list(exprs)
    return table._resolve(exprs)


_function_types = tuple(
    filter(
        None,
        (
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.FunctionType,
            types.LambdaType,
            types.MethodType,
            getattr(types, 'UnboundMethodType', None),
        ),
    )
)


def _get_group_by_key(table, value):
    if isinstance(value, str):
        return table[value]
    if isinstance(value, _function_types):
        return value(table)
    return value


class GroupedTableExpr:

    """
    Helper intermediate construct
    """

    def __init__(
        self, table, by, having=None, order_by=None, window=None, **expressions
    ):
        self.table = table
        self.by = util.promote_list(by if by is not None else []) + [
            _get_group_by_key(table, v).name(k)
            for k, v in sorted(expressions.items(), key=toolz.first)
        ]
        self._order_by = order_by or []
        self._having = having or []
        self._window = window

    def __getitem__(self, args):
        # Shortcut for projection with window functions
        return self.projection(list(args))

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

    def aggregate(self, metrics=None, **kwds):
        return self.table.aggregate(
            metrics, by=self.by, having=self._having, **kwds
        )

    def having(self, expr):
        """
        Add a post-aggregation result filter (like the having argument in
        `aggregate`), for composability with the group_by API

        Parameters
        ----------
        expr : ibis.expr.types.Expr

        Returns
        -------
        grouped : GroupedTableExpr
        """
        exprs = util.promote_list(expr)
        new_having = self._having + exprs
        return GroupedTableExpr(
            self.table,
            self.by,
            having=new_having,
            order_by=self._order_by,
            window=self._window,
        )

    def order_by(self, expr):
        """
        Expressions to use for ordering data for a window function
        computation. Ignored in aggregations.

        Parameters
        ----------
        expr : value expression or list of value expressions

        Returns
        -------
        grouped : GroupedTableExpr
        """
        exprs = util.promote_list(expr)
        new_order = self._order_by + exprs
        return GroupedTableExpr(
            self.table,
            self.by,
            having=self._having,
            order_by=new_order,
            window=self._window,
        )

    def mutate(self, exprs=None, **kwds):
        """
        Returns a table projection with analytic / window functions
        applied. Any arguments can be functions.

        Parameters
        ----------
        exprs : list, default None
        kwds : key=value pairs

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([
        ...     ('foo', 'string'),
        ...     ('bar', 'string'),
        ...     ('baz', 'double'),
        ... ], name='t')
        >>> t
        UnboundTable[table]
          name: t
          schema:
            foo : string
            bar : string
            baz : float64
        >>> expr = (t.group_by('foo')
        ...          .order_by(ibis.desc('bar'))
        ...          .mutate(qux=lambda x: x.baz.lag(),
        ...                  qux2=t.baz.lead()))
        >>> print(expr)  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: t
          schema:
            foo : string
            bar : string
            baz : float64
        Selection[table]
          table:
            Table: ref_0
          selections:
            Table: ref_0
            qux = WindowOp[float64*]
              qux = Lag[float64*]
                baz = Column[float64*] 'baz' from table
                  ref_0
                offset:
                  None
                default:
                  None
              <ibis.expr.window.Window object at 0x...>
            qux2 = WindowOp[float64*]
              qux2 = Lead[float64*]
                baz = Column[float64*] 'baz' from table
                  ref_0
                offset:
                  None
                default:
                  None
              <ibis.expr.window.Window object at 0x...>

        Returns
        -------
        mutated : TableExpr
        """
        if exprs is None:
            exprs = []
        else:
            exprs = util.promote_list(exprs)

        kwd_names = list(kwds.keys())
        kwd_values = list(kwds.values())
        kwd_values = self.table._resolve(kwd_values)

        for k, v in sorted(zip(kwd_names, kwd_values)):
            exprs.append(v.name(k))

        return self.projection([self.table] + exprs)

    def projection(self, exprs):
        """
        Like mutate, but do not include existing table columns
        """
        w = self._get_window()
        windowed_exprs = []
        exprs = self.table._resolve(exprs)
        for expr in exprs:
            expr = L.windowize_function(expr, w=w)
            windowed_exprs.append(expr)
        return self.table.projection(windowed_exprs)

    def _get_window(self):
        if self._window is None:
            groups = self.by
            sorts = self._order_by
            preceding, following = None, None
        else:
            w = self._window
            groups = w.group_by + self.by
            sorts = w.order_by + self._order_by
            preceding, following = w.preceding, w.following

        sorts = [ops.to_sort_key(self.table, k) for k in sorts]

        groups = _resolve_exprs(self.table, groups)

        return _window.window(
            preceding=preceding,
            following=following,
            group_by=groups,
            order_by=sorts,
        )

    def over(self, window):
        """
        Add a window clause to be applied to downstream analytic expressions
        """
        return GroupedTableExpr(
            self.table,
            self.by,
            having=self._having,
            order_by=self._order_by,
            window=window,
        )

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
        return self.table.aggregate([metric], by=self.by, having=self._having)

    size = count


def _group_agg_dispatch(name):
    def wrapper(self, *args, **kwargs):
        f = getattr(self.arr, name)
        metric = f(*args, **kwargs)
        alias = '{0}({1})'.format(name, self.arr.get_name())
        return self.parent.aggregate(metric.name(alias))

    wrapper.__name__ = name
    return wrapper


class GroupedArray:
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

    def summary(self, exact_nunique=False):
        metric = self.arr.summary(exact_nunique=exact_nunique)
        return self.parent.aggregate(metric)


class GroupedNumbers(GroupedArray):

    mean = _group_agg_dispatch('mean')
    sum = _group_agg_dispatch('sum')

    def summary(self, exact_nunique=False):
        metric = self.arr.summary(exact_nunique=exact_nunique)
        return self.parent.aggregate(metric)
