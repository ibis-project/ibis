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

"""User API for grouping operations."""

from __future__ import annotations

import types
from typing import Iterable, Sequence

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
    """An intermediate table expression to hold grouping information."""

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

    def having(self, expr: ir.BooleanScalar) -> GroupedTableExpr:
        """Add a post-aggregation result filter `expr`.

        Parameters
        ----------
        expr
            An expression that filters based on an aggregate value.

        Returns
        -------
        GroupedTableExpr
            A grouped table expression
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

    def order_by(
        self, expr: ir.ValueExpr | Iterable[ir.ValueExpr]
    ) -> GroupedTableExpr:
        """Sort a grouped table expression by `expr`.

        Notes
        -----
        This API call is ignored in aggregations.

        Parameters
        ----------
        expr
            Expressions to order the results by

        Returns
        -------
        GroupedTableExpr
            A sorted grouped GroupedTableExpr
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

    def mutate(
        self,
        exprs: ir.ValueExpr | Sequence[ir.ValueExpr] | None = None,
        **kwds: ir.ValueExpr,
    ):
        """Return a table projection with window functions applied.

        Any arguments can be functions.

        Parameters
        ----------
        exprs
            List of expressions
        kwds
            Expressions

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([
        ...     ('foo', 'string'),
        ...     ('bar', 'string'),
        ...     ('baz', 'double'),
        ... ], name='t')
        >>> t
        UnboundTable[t]
          foo string
          bar string
          baz float64
        >>> expr = (t.group_by('foo')
        ...          .order_by(ibis.desc('bar'))
        ...          .mutate(qux=lambda x: x.baz.lag(),
        ...                  qux2=t.baz.lead()))
        >>> print(expr)
        r0 := UnboundTable[t]
          foo string
          bar string
          baz float64
        Selection[r0]
          selections:
            r0
            qux:  WindowOp(Lag(r0.baz), window=Window(group_by=[r0.foo], order_by=[desc|r0.bar], how='rows'))
            qux2: WindowOp(Lead(r0.baz), window=Window(group_by=[r0.foo], order_by=[desc|r0.bar], how='rows'))

        Returns
        -------
        TableExpr
            A table expression with window functions applied
        """  # noqa: E501
        if exprs is None:
            exprs = []
        else:
            exprs = util.promote_list(exprs)

        kwd_keys = list(kwds.keys())
        kwd_values = self.table._resolve(list(kwds.values()))

        for k, v in zip(kwd_keys, kwd_values):
            exprs.append(v.name(k))

        return self.projection([self.table] + exprs)

    def projection(self, exprs):
        """Project new columns out of the grouped table.

        See Also
        --------
        ibis.expr.groupby.GroupedTableExpr.mutate
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

        sorts = [ops.sortkeys._to_sort_key(k, table=self.table) for k in sorts]

        groups = _resolve_exprs(self.table, groups)

        return _window.window(
            preceding=preceding,
            following=following,
            group_by=groups,
            order_by=sorts,
        )

    def over(self, window: _window.Window) -> GroupedTableExpr:
        """Add a window frame clause to be applied to child analytic expressions.

        Parameters
        ----------
        window
            Window to add to child analytic expressions

        Returns
        -------
        GroupedTableExpr
            A new grouped table expression
        """
        return GroupedTableExpr(
            self.table,
            self.by,
            having=self._having,
            order_by=self._order_by,
            window=window,
        )

    def count(self, metric_name: str = 'count') -> ir.TableExpr:
        """Computing the number of rows per group.

        Parameters
        ----------
        metric_name
            Name to use for the row count metric

        Returns
        -------
        TableExpr
            The aggregated table
        """
        metric = self.table.count().name(metric_name)
        return self.table.aggregate([metric], by=self.by, having=self._having)

    size = count


def _group_agg_dispatch(name):
    def wrapper(self, *args, **kwargs):
        f = getattr(self.arr, name)
        metric = f(*args, **kwargs)
        alias = f'{name}({self.arr.get_name()})'
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
