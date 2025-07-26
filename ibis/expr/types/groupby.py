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

from typing import TYPE_CHECKING, Union

from public import public

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.common.deferred import Deferred, deferrable
from ibis.common.grounds import Annotable, Concrete
from ibis.common.selectors import Expandable
from ibis.common.typing import VarTuple  # noqa: TC001
from ibis.expr.rewrites import rewrite_window_input
from ibis.util import experimental

if TYPE_CHECKING:
    from collections.abc import Sequence


@public
class GroupedTable(Concrete):
    """An intermediate table expression to hold grouping information."""

    table: ops.Relation
    # groupings is allowed to be empty when there are some form of grouping
    # sets provided
    #
    # groupings are *strictly* the things the user has explicitly requested to
    # group by that are not part of a grouping set
    groupings: VarTuple[ops.Value]
    orderings: VarTuple[ops.SortKey] = ()
    havings: VarTuple[ops.Value[dt.Boolean]] = ()

    grouping_sets: VarTuple[VarTuple[VarTuple[ir.Value]]] = ()
    rollups: VarTuple[VarTuple[ir.Value]] = ()
    cubes: VarTuple[VarTuple[ir.Value]] = ()

    def __getitem__(self, args):
        # Shortcut for projection with window functions
        return self.select(*args)

    def __getattr__(self, attr):
        try:
            field = getattr(self.table.to_expr(), attr)
        except AttributeError as e:
            raise AttributeError(f"GroupedTable has no attribute {attr}") from e

        if isinstance(field, ir.NumericValue):
            return GroupedNumbers(field, self)
        else:
            return GroupedArray(field, self)

    def aggregate(self, *metrics, **kwds) -> ir.Table:
        """Compute aggregates over a group by."""
        metrics = self.table.to_expr().bind(*metrics, **kwds)
        return self.table.to_expr().aggregate(
            metrics,
            by=self.groupings,
            having=self.havings,
            grouping_sets=self.grouping_sets,
            rollups=self.rollups,
            cubes=self.cubes,
        )

    agg = aggregate

    def having(self, *predicates: ir.BooleanScalar) -> GroupedTable:
        """Add a post-aggregation result filter `expr`.

        ::: {.callout-warning}
        ## Expressions like `x is None` return `bool` and **will not** generate a SQL comparison to `NULL`
        :::

        Parameters
        ----------
        predicates
            Expressions that filters based on an aggregate value.

        Returns
        -------
        GroupedTable
            A grouped table expression

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {"grouper": ["a", "a", "a", "b", "b", "c"], "values": [1, 2, 3, 1, 2, 1]}
        ... )
        >>> expr = (
        ...     t.group_by(t.grouper)
        ...     .having(t.count() < 3)
        ...     .aggregate(values_count=t.count(), values_sum=t.values.sum())
        ...     .order_by(t.grouper)
        ... )
        >>> expr
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ grouper ┃ values_count ┃ values_sum ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
        │ string  │ int64        │ int64      │
        ├─────────┼──────────────┼────────────┤
        │ b       │            2 │          3 │
        │ c       │            1 │          1 │
        └─────────┴──────────────┴────────────┘
        """
        table = self.table.to_expr()
        havings = table.bind(*predicates)
        return self.copy(havings=self.havings + havings)

    def order_by(self, *by: ir.Value) -> GroupedTable:
        """Sort a grouped table expression by `expr`.

        Notes
        -----
        This API call is ignored in aggregations.

        Parameters
        ----------
        by
            Expressions to order the results by

        Returns
        -------
        GroupedTable
            A sorted grouped GroupedTable
        """
        table = self.table.to_expr()
        orderings = table.bind(*by)
        return self.copy(orderings=self.orderings + orderings)

    def mutate(
        self, *exprs: ir.Value | Sequence[ir.Value], **kwexprs: ir.Value
    ) -> ir.Table:
        """Return a table projection with window functions applied.

        Any arguments can be functions.

        Parameters
        ----------
        exprs
            List of expressions
        kwexprs
            Expressions

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> (
        ...     t.select("species", "bill_length_mm")
        ...     .group_by("species")
        ...     .mutate(centered_bill_len=ibis._.bill_length_mm - ibis._.bill_length_mm.mean())
        ...     .order_by(s.all())
        ... )
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ species ┃ bill_length_mm ┃ centered_bill_len ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ string  │ float64        │ float64           │
        ├─────────┼────────────────┼───────────────────┤
        │ Adelie  │           32.1 │         -6.691391 │
        │ Adelie  │           33.1 │         -5.691391 │
        │ Adelie  │           33.5 │         -5.291391 │
        │ Adelie  │           34.0 │         -4.791391 │
        │ Adelie  │           34.1 │         -4.691391 │
        │ Adelie  │           34.4 │         -4.391391 │
        │ Adelie  │           34.5 │         -4.291391 │
        │ Adelie  │           34.6 │         -4.191391 │
        │ Adelie  │           34.6 │         -4.191391 │
        │ Adelie  │           35.0 │         -3.791391 │
        │ …       │              … │                 … │
        └─────────┴────────────────┴───────────────────┘

        Returns
        -------
        Table
            A table expression with window functions applied
        """
        exprs = self._selectables(*exprs, **kwexprs)
        return self.table.to_expr().mutate(exprs)

    def select(self, *exprs, **kwexprs) -> ir.Table:
        """Project new columns out of the grouped table.

        See Also
        --------
        [`GroupedTable.mutate`](#ibis.expr.types.groupby.GroupedTable.mutate)
        """
        exprs = self._selectables(*exprs, **kwexprs)
        return self.table.to_expr().select(exprs)

    def _selectables(self, *exprs, **kwexprs):
        """Project new columns out of the grouped table.

        See Also
        --------
        [`GroupedTable.mutate`](#ibis.expr.types.groupby.GroupedTable.mutate)
        """
        if self.grouping_sets or self.rollups or self.cubes:
            raise exc.UnsupportedOperationError(
                "Grouping sets, rollups, and cubes are not supported in grouped `mutate` or `select`"
            )
        table = self.table.to_expr()
        values = table.bind(*exprs, **kwexprs)
        window = ibis.window(group_by=self.groupings, order_by=self.orderings)
        return [rewrite_window_input(expr.op(), window).to_expr() for expr in values]

    projection = select

    def over(
        self,
        window=None,
        *,
        rows=None,
        range=None,
        group_by=None,
        order_by=None,
    ) -> GroupedTable:
        """Apply a window over the input expressions.

        Parameters
        ----------
        window
            Window to add to the input
        rows
            Whether to use the `ROWS` window clause
        range
            Whether to use the `RANGE` window clause
        group_by
            Grouping key
        order_by
            Ordering key

        Returns
        -------
        GroupedTable
            A new grouped table expression
        """
        if window is None:
            window = ibis.window(
                rows=rows,
                range=range,
                group_by=group_by,
                order_by=order_by,
            )

        # TODO: reject grouping sets here
        return self.__class__(
            self.table,
            self.by,
            having=self._having,
            order_by=self._order_by,
            window=window,
        )

    def count(self) -> ir.Table:
        """Computing the number of rows per group.

        Returns
        -------
        Table
            The aggregated table
        """
        table = self.table.to_expr()
        return table.aggregate(
            table.count(),
            by=self.groupings,
            having=self.havings,
            grouping_sets=self.grouping_sets,
            rollups=self.rollups,
            cubes=self.cubes,
        )

    size = count


def _group_agg_dispatch(name):
    def wrapper(self, *args, **kwargs):
        f = getattr(self.arr, name)
        metric = f(*args, **kwargs)
        alias = f"{name}({self.arr.get_name()})"
        return self.parent.aggregate(metric.name(alias))

    wrapper.__name__ = name
    return wrapper


@public
class GroupedArray:
    def __init__(self, arr, parent):
        self.arr = arr
        self.parent = parent

    count = _group_agg_dispatch("count")
    size = count
    min = _group_agg_dispatch("min")
    max = _group_agg_dispatch("max")
    approx_nunique = _group_agg_dispatch("approx_nunique")
    approx_median = _group_agg_dispatch("approx_median")
    group_concat = _group_agg_dispatch("group_concat")


@public
class GroupedNumbers(GroupedArray):
    mean = _group_agg_dispatch("mean")
    sum = _group_agg_dispatch("sum")


class GroupingSets(Annotable, Expandable):
    exprs: VarTuple[VarTuple[Union[str, ir.Value, Deferred]]]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        # produce all unique expressions in the grouping set, rollup or cube
        values = []
        for expr in self.exprs:
            values.append(tuple(table.bind(expr)))
        return values


class GroupingSetsShorthand(Annotable, Expandable):
    exprs: VarTuple[Union[str, ir.Value, Deferred]]

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        # produce all unique expressions in the grouping set, rollup or cube
        values = []
        for expr in self.exprs:
            values.extend(table.bind(expr))
        return values


class Rollup(GroupingSetsShorthand):
    pass


class Cube(GroupingSetsShorthand):
    pass


@public
@experimental
def rollup(*dims):
    """Construct a rollup.

    Rollups are a shorthand for grouping sets that are sequentially more coarse
    grained aggregations.

    Conceptually, a rollup is a union of a grouping sets, where each grouping
    set is a superset of the previous one.

    Here's some SQL showing `ROLLUP` equivalence to standard issue `GROUP BY`:

    ```sql
    -- 1. grouping set is a, b
    SELECT a, b, count(*) n
    FROM t
    GROUP BY a, b

    UNION ALL

    --- 2. grouping set is a (rolled up from a, b)
    SELECT a, NULL, count(*) n
    FROM t
    GROUP BY a

    UNION ALL

    -- 3. no grouping set, i.e., all rows (rolled up from a)
    SELECT NULL, NULL, count(*) n
    FROM t
    ```

    See Also
    --------
    cube
    grouping_sets

    Examples
    --------
    >>> import ibis
    >>> from ibis import _
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t.head()
    ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string    │ float64        │ float64       │ int64             │ … │
    ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
    │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
    │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
    │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
    │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
    └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
    >>> (
    ...     t.group_by(ibis.rollup(_.island, _.sex))
    ...     .agg(mean_bill_length=_.bill_length_mm.mean())
    ...     .order_by(
    ...         _.island.asc(nulls_first=True),
    ...         _.sex.asc(nulls_first=True),
    ...         _.mean_bill_length.desc(),
    ...     )
    ... )
    ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ island    ┃ sex    ┃ mean_bill_length ┃
    ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ string    │ string │ float64          │
    ├───────────┼────────┼──────────────────┤
    │ NULL      │ NULL   │        43.921930 │
    │ Biscoe    │ NULL   │        45.625000 │
    │ Biscoe    │ NULL   │        45.257485 │
    │ Biscoe    │ female │        43.307500 │
    │ Biscoe    │ male   │        47.119277 │
    │ Dream     │ NULL   │        44.167742 │
    │ Dream     │ NULL   │        37.500000 │
    │ Dream     │ female │        42.296721 │
    │ Dream     │ male   │        46.116129 │
    │ Torgersen │ NULL   │        38.950980 │
    │ …         │ …      │                … │
    └───────────┴────────┴──────────────────┘
    """
    return Rollup(dims)


@public
@experimental
def cube(*dims):
    """Construct a cube.

    ::: {.callout-note}
    ## Cubes can be very expensive to compute.
    :::

    Cubes are a shorthand for grouping sets that contain all possible ways
    to aggregate a set of grouping keys.

    Conceptually, a cube is a union of a grouping sets, where each grouping
    set is a member of the set of all sets of grouping keys (the power set).

    See Also
    --------
    rollup
    grouping_sets

    Examples
    --------
    >>> import ibis
    >>> from ibis import _
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t.head()
    ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string    │ float64        │ float64       │ int64             │ … │
    ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
    │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
    │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
    │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
    │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
    └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
    >>> (
    ...     t.group_by(ibis.cube("island", "sex"))
    ...     .agg(mean_bill_length=_.bill_length_mm.mean())
    ...     .order_by(
    ...         _.island.asc(nulls_first=True),
    ...         _.sex.asc(nulls_first=True),
    ...         _.mean_bill_length.desc(),
    ...     )
    ... )
    ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ island ┃ sex    ┃ mean_bill_length ┃
    ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ string │ string │ float64          │
    ├────────┼────────┼──────────────────┤
    │ NULL   │ NULL   │        43.921930 │
    │ NULL   │ NULL   │        41.300000 │
    │ NULL   │ female │        42.096970 │
    │ NULL   │ male   │        45.854762 │
    │ Biscoe │ NULL   │        45.625000 │
    │ Biscoe │ NULL   │        45.257485 │
    │ Biscoe │ female │        43.307500 │
    │ Biscoe │ male   │        47.119277 │
    │ Dream  │ NULL   │        44.167742 │
    │ Dream  │ NULL   │        37.500000 │
    │ …      │ …      │                … │
    └────────┴────────┴──────────────────┘
    """
    return Cube(dims)


@public
@experimental
def grouping_sets(*dims):
    """Construct a grouping set.

    See Also
    --------
    rollup
    cube

    >>> import ibis
    >>> from ibis import _
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t.head()
    ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string    │ float64        │ float64       │ int64             │ … │
    ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
    │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
    │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
    │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
    │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
    └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
    >>> (
    ...     t.group_by(ibis.grouping_sets((), _.island, (_.island, _.sex)))
    ...     .agg(mean_bill_length=_.bill_length_mm.mean())
    ...     .order_by(
    ...         _.island.asc(nulls_first=True),
    ...         _.sex.asc(nulls_first=True),
    ...         _.mean_bill_length.desc(),
    ...     )
    ... )
    ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ island    ┃ sex    ┃ mean_bill_length ┃
    ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ string    │ string │ float64          │
    ├───────────┼────────┼──────────────────┤
    │ NULL      │ NULL   │        43.921930 │
    │ Biscoe    │ NULL   │        45.625000 │
    │ Biscoe    │ NULL   │        45.257485 │
    │ Biscoe    │ female │        43.307500 │
    │ Biscoe    │ male   │        47.119277 │
    │ Dream     │ NULL   │        44.167742 │
    │ Dream     │ NULL   │        37.500000 │
    │ Dream     │ female │        42.296721 │
    │ Dream     │ male   │        46.116129 │
    │ Torgersen │ NULL   │        38.950980 │
    │ …         │ …      │                … │
    └───────────┴────────┴──────────────────┘

    The previous example is equivalent to using a rollup:

    >>> (
    ...     t.group_by(ibis.rollup(_.island, _.sex))
    ...     .agg(mean_bill_length=_.bill_length_mm.mean())
    ...     .order_by(
    ...         _.island.asc(nulls_first=True),
    ...         _.sex.asc(nulls_first=True),
    ...         _.mean_bill_length.desc(),
    ...     )
    ... )
    ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ island    ┃ sex    ┃ mean_bill_length ┃
    ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ string    │ string │ float64          │
    ├───────────┼────────┼──────────────────┤
    │ NULL      │ NULL   │        43.921930 │
    │ Biscoe    │ NULL   │        45.625000 │
    │ Biscoe    │ NULL   │        45.257485 │
    │ Biscoe    │ female │        43.307500 │
    │ Biscoe    │ male   │        47.119277 │
    │ Dream     │ NULL   │        44.167742 │
    │ Dream     │ NULL   │        37.500000 │
    │ Dream     │ female │        42.296721 │
    │ Dream     │ male   │        46.116129 │
    │ Torgersen │ NULL   │        38.950980 │
    │ …         │ …      │                … │
    └───────────┴────────┴──────────────────┘
    """
    return GroupingSets(tuple(map(tuple, map(ibis.util.promote_list, dims))))


@experimental
@deferrable
def group_id(first, *rest) -> ir.IntegerScalar:
    """Return the grouping ID for a set of columns.

    Input columns must be part of the group by clause.

    ::: {.callout-note}
    ## This function can only be called in a group by context.
    :::

    Returns
    -------
    IntegerScalar
        An integer whose bits represent whether the `i`th
        group is present in the current row's aggregated value.

    Examples
    --------
    >>> import ibis
    >>> from ibis import _
    >>> ibis.options.interactive = True
    >>> t = ibis.examples.penguins.fetch()
    >>> t.head()
    ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
    ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
    │ string  │ string    │ float64        │ float64       │ int64             │ … │
    ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
    │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
    │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
    │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
    │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
    │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
    └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
    >>> (
    ...     t.group_by(ibis.rollup(_.island, _.sex))
    ...     .agg(
    ...         group_id=ibis.group_id(_.island, _.sex),
    ...         mean_bill_length=_.bill_length_mm.mean(),
    ...     )
    ...     .relocate(_.group_id)
    ...     .order_by(
    ...         _.group_id.desc(),
    ...         _.island.asc(nulls_first=True),
    ...         _.sex.asc(nulls_first=True),
    ...         _.mean_bill_length.desc(),
    ...     )
    ... )
    ┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
    ┃ group_id ┃ island    ┃ sex    ┃ mean_bill_length ┃
    ┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
    │ int64    │ string    │ string │ float64          │
    ├──────────┼───────────┼────────┼──────────────────┤
    │        3 │ NULL      │ NULL   │        43.921930 │
    │        1 │ Biscoe    │ NULL   │        45.257485 │
    │        1 │ Dream     │ NULL   │        44.167742 │
    │        1 │ Torgersen │ NULL   │        38.950980 │
    │        0 │ Biscoe    │ NULL   │        45.625000 │
    │        0 │ Biscoe    │ female │        43.307500 │
    │        0 │ Biscoe    │ male   │        47.119277 │
    │        0 │ Dream     │ NULL   │        37.500000 │
    │        0 │ Dream     │ female │        42.296721 │
    │        0 │ Dream     │ male   │        46.116129 │
    │        … │ …         │ …      │                … │
    └──────────┴───────────┴────────┴──────────────────┘
    """
    return ops.GroupID((first, *rest)).to_expr()
