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

import ibis.expr.analysis as an
import ibis.expr.types as ir
import ibis.expr.window as _window
from ibis import util
from ibis.expr.deferred import Deferred

_function_types = tuple(
    filter(
        None,
        (
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.FunctionType,
            types.LambdaType,
            types.MethodType,
            getattr(types, "UnboundMethodType", None),
        ),
    )
)


def _get_group_by_key(table, value):
    if isinstance(value, str):
        return table[value]
    elif isinstance(value, _function_types):
        return value(table)
    elif isinstance(value, Deferred):
        return value.resolve(table)
    elif isinstance(value, ir.Expr):
        return an.sub_immediate_parents(value.op(), table.op()).to_expr()
    else:
        return value


class GroupedTable:
    """An intermediate table expression to hold grouping information."""

    def __init__(
        self, table, by, having=None, order_by=None, window=None, **expressions
    ):
        self.table = table
        self.by = [_get_group_by_key(table, v) for v in util.promote_list(by)] + [
            _get_group_by_key(table, v).name(k) for k, v in expressions.items()
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
        """Compute aggregates over a group by."""
        return self.table.aggregate(metrics, by=self.by, having=self._having, **kwds)

    agg = aggregate

    def having(self, expr: ir.BooleanScalar) -> GroupedTable:
        """Add a post-aggregation result filter `expr`.

        Parameters
        ----------
        expr
            An expression that filters based on an aggregate value.

        Returns
        -------
        GroupedTable
            A grouped table expression
        """
        return self.__class__(
            self.table,
            self.by,
            having=self._having + util.promote_list(expr),
            order_by=self._order_by,
            window=self._window,
        )

    def order_by(self, expr: ir.Value | Iterable[ir.Value]) -> GroupedTable:
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
        GroupedTable
            A sorted grouped GroupedTable
        """
        return self.__class__(
            self.table,
            self.by,
            having=self._having,
            order_by=self._order_by + util.promote_list(expr),
            window=self._window,
        )

    def mutate(
        self, exprs: ir.Value | Sequence[ir.Value] | None = None, **kwds: ir.Value
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
        ...          .mutate(qux=lambda x: x.baz.lag(), qux2=t.baz.lead()))
        >>> print(expr)
        r0 := UnboundTable[t]
          foo string
          bar string
          baz float64
        Selection[r0]
          selections:
            r0
            qux:  Window(Lag(r0.baz), window=Window(group_by=[r0.foo], order_by=[desc|r0.bar], how='rows'))
            qux2: Window(Lead(r0.baz), window=Window(group_by=[r0.foo], order_by=[desc|r0.bar], how='rows'))

        Returns
        -------
        Table
            A table expression with window functions applied
        """
        if exprs is None:
            exprs = []
        else:
            exprs = util.promote_list(exprs)

        for name, expr in kwds.items():
            expr = self.table._ensure_expr(expr)
            exprs.append(expr.name(name))

        return self.projection([self.table, *exprs])

    def projection(self, exprs):
        """Project new columns out of the grouped table.

        See Also
        --------
        [`GroupedTable.mutate`][ibis.expr.types.groupby.GroupedTable.mutate]
        """
        w = self._get_window()
        windowed_exprs = []
        for expr in util.promote_list(exprs):
            expr = self.table._ensure_expr(expr)
            expr = an.windowize_function(expr, w=w)
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

        return _window.window(
            preceding=preceding,
            following=following,
            group_by=list(map(self.table._ensure_expr, util.promote_list(groups))),
            order_by=list(map(self.table._ensure_expr, util.promote_list(sorts))),
        )

    def over(self, window: _window.Window) -> GroupedTable:
        """Apply a window over the input expressions.

        Parameters
        ----------
        window
            Window to add to the input

        Returns
        -------
        GroupedTable
            A new grouped table expression
        """
        return self.__class__(
            self.table,
            self.by,
            having=self._having,
            order_by=self._order_by,
            window=window,
        )

    def count(self, metric_name: str = 'count') -> ir.Table:
        """Computing the number of rows per group.

        Parameters
        ----------
        metric_name
            Name to use for the row count metric

        Returns
        -------
        Table
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
        """Summarize a column.

        Parameters
        ----------
        exact_nunique
            Whether to compute an exact count distinct.
        """
        metric = self.arr.summary(exact_nunique=exact_nunique)
        return self.parent.aggregate(metric)


class GroupedNumbers(GroupedArray):
    mean = _group_agg_dispatch('mean')
    sum = _group_agg_dispatch('sum')
