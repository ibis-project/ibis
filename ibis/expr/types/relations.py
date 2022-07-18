from __future__ import annotations

import collections
import functools
import itertools
import operator
from functools import cached_property
from typing import IO, TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import tabulate
from public import public

import ibis
from ibis import util
from ibis.common import exceptions as com
from ibis.expr.deferred import Deferred
from ibis.expr.types.core import Expr

if TYPE_CHECKING:
    from ibis.expr import schema as sch
    from ibis.expr import types as ir
    from ibis.expr.types.generic import Column
    from ibis.expr.types.groupby import GroupedTable


_ALIASES = (f"_ibis_view_{n:d}" for n in itertools.count())


def _regular_join_method(
    name: str,
    how: Literal[
        "inner",
        "left",
        "outer",
        "right",
        "semi",
        "anti",
        "any_inner",
        "any_left",
    ],
):
    def f(
        self: Table,
        right: Table,
        predicates: str
        | Sequence[
            str | tuple[str | ir.Column, str | ir.Column] | ir.BooleanValue
        ] = (),
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Table:
        f"""Perform a{'n' * how.startswith(tuple("aeiou"))} {how} join between two tables.

        Parameters
        ----------
        right
            Right table to join
        predicates
            Boolean or column names to join on
        suffixes
            Left and right suffixes that will be used to rename overlapping
            columns.

        Returns
        -------
        Table
            Joined table
        """  # noqa: E501
        return self.join(right, predicates, how=how, suffixes=suffixes)

    f.__name__ = name
    return f


@public
class Table(Expr):
    def __contains__(self, name):
        return name in self.schema()

    def _repr_html_(self) -> str | None:
        if not ibis.options.interactive:
            return None

        return self.execute()._repr_html_()

    def __getitem__(self, what):
        from ibis.expr.types.analytic import TopK
        from ibis.expr.types.generic import Column
        from ibis.expr.types.logical import BooleanColumn

        if isinstance(what, (str, int)):
            return self.get_column(what)

        if isinstance(what, slice):
            step = what.step
            if step is not None and step != 1:
                raise ValueError('Slice step can only be 1')
            start = what.start or 0
            stop = what.stop

            if stop is None or stop < 0:
                raise ValueError('End index must be a positive number')

            if start < 0:
                raise ValueError('Start index must be a positive number')

            return self.limit(stop - start, offset=start)

        what = bind_expr(self, what)

        if isinstance(what, TopK):
            return what._to_semi_join(self)[self]
        elif isinstance(what, (list, tuple, Table)):
            # Projection case
            return self.projection(what)
        elif isinstance(what, BooleanColumn):
            # Boolean predicate
            return self.filter([what])
        elif isinstance(what, Column):
            # Projection convenience
            return self.projection(what)
        else:
            raise NotImplementedError(
                'Selection rows or columns with {} objects is not '
                'supported'.format(type(what).__name__)
            )

    def __len__(self):
        raise com.ExpressionError('Use .count() instead')

    def __setstate__(self, instance_dictionary):
        self.__dict__ = instance_dictionary

    def __getattr__(self, key):
        try:
            schema = self.schema()
        except com.IbisError:
            raise AttributeError(key)

        if key not in schema:
            raise AttributeError(key)

        try:
            return self.get_column(key)
        except com.IbisTypeError:
            raise AttributeError(key)

    def __dir__(self):
        return sorted(frozenset(dir(type(self)) + self.columns))

    # TODO(kszucs): should be removed
    def _ensure_expr(self, expr):
        if isinstance(expr, str):
            return self[expr]
        elif isinstance(expr, (int, np.integer)):
            return self[self.schema().name_at_position(expr)]
        elif isinstance(expr, Deferred):
            return expr.resolve(self)
        elif not isinstance(expr, Expr):
            return expr(self)
        else:
            return expr

    def get_columns(self, iterable: Iterable[str]) -> list[Column]:
        """
        Get multiple columns from the table

        Examples
        --------
        >>> import ibis
        >>> table = ibis.table(
        ...    [
        ...        ('a', 'int64'),
        ...        ('b', 'string'),
        ...        ('c', 'timestamp'),
        ...        ('d', 'float'),
        ...    ],
        ...    name='t'
        ... )
        >>> a, b, c = table.get_columns(['a', 'b', 'c'])

        Returns
        -------
        list[ir.Column]
            List of column expressions
        """
        return [self.get_column(x) for x in iterable]

    def get_column(self, name: str) -> Column:
        """
        Get a reference to a single column from the table

        Returns
        -------
        Column
            A column named `name`.
        """
        import ibis.expr.operations as ops

        ref = ops.TableColumn(self, name)
        return ref.to_expr()

    @cached_property
    def columns(self):
        return list(self.schema().names)

    def schema(self) -> sch.Schema:
        """
        Get the schema for this table (if one is known)

        Returns
        -------
        Schema
            The table's schema.
        """
        return self.op().schema

    def group_by(
        self,
        by=None,
        **additional_grouping_expressions: Any,
    ) -> GroupedTable:
        """
        Create a grouped table expression.

        Parameters
        ----------
        by
            Grouping expressions
        additional_grouping_expressions
            Named grouping expressions

        Examples
        --------
        >>> import ibis
        >>> pairs = [('a', 'int32'), ('b', 'timestamp'), ('c', 'double')]
        >>> t = ibis.table(pairs)
        >>> b1, b2 = t.a, t.b
        >>> result = t.group_by([b1, b2]).aggregate(sum_of_c=t.c.sum())

        Returns
        -------
        GroupedTable
            A grouped table expression
        """
        from ibis.expr.types.groupby import GroupedTable

        return GroupedTable(self, by, **additional_grouping_expressions)

    groupby = group_by

    def rowid(self) -> ir.IntegerValue:
        """A numbering expression representing the row number of the results.

        It can be 0 or 1 indexed depending on the backend. Check the backend
        documentation for specifics.

        Notes
        -----
        This function is different from the window function `row_number`
        (even if they are conceptually the same), and different from `rowid` in
        backends where it represents the physical location
        (e.g. Oracle or PostgreSQL's ctid).

        Returns
        -------
        IntegerColumn
            An integer column

        Examples
        --------
        >>> my_table[my_table.rowid(), my_table.name].execute()  # doctest: +SKIP
        1|Ibis
        2|pandas
        3|Dask
        """  # noqa: E501
        from ibis.expr import operations as ops

        return ops.RowID().to_expr()

    def view(self) -> Table:
        """Create a new table expression distinct from the current one.

        Use this API for any self-referencing operations like a self-join.

        Returns
        -------
        Table
            Table expression
        """
        from ibis.expr import operations as ops

        return ops.SelfReference(self).to_expr()

    def difference(self, right: Table) -> Table:
        """Compute the set difference of two table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        right
            Table expression

        Returns
        -------
        Table
            The rows present in `left` that are not present in `right`.
        """
        from ibis.expr import operations as ops

        return ops.Difference(self, right).to_expr()

    def aggregate(
        self,
        metrics: Sequence[ir.Scalar] | None = None,
        by: Sequence[ir.Value] | None = None,
        having: Sequence[ir.BooleanValue] | None = None,
        **kwargs: ir.Value,
    ) -> Table:
        """Aggregate a table with a given set of reductions grouping by `by`.

        Parameters
        ----------
        metrics
            Aggregate expressions
        by
            Grouping expressions
        having
            Post-aggregation filters
        kwargs
            Named aggregate expressions

        Returns
        -------
        Table
            An aggregate table expression
        """
        metrics = [] if metrics is None else util.promote_list(metrics)
        metrics.extend(
            self._ensure_expr(expr).name(name) for name, expr in kwargs.items()
        )

        op = self.op().aggregate(
            self,
            [
                metric
                if util.is_iterable(metric)
                else self._ensure_expr(metric)
                for metric in metrics
            ],
            by=list(
                map(
                    self._ensure_expr,
                    util.promote_list(by if by is not None else []),
                )
            ),
            having=list(
                map(
                    self._ensure_expr,
                    util.promote_list(having if having is not None else []),
                )
            ),
        )
        return op.to_expr()

    def distinct(self) -> Table:
        """Compute the set of unique rows in the table."""
        from ibis.expr import operations as ops

        return ops.Distinct(self).to_expr()

    def limit(self, n: int, offset: int = 0) -> Table:
        """Select the first `n` rows at beginning of table starting at `offset`.

        Parameters
        ----------
        n
            Number of rows to include
        offset
            Number of rows to skip first

        Returns
        -------
        Table
            The first `n` rows of `table` starting at `offset`
        """
        from ibis.expr import operations as ops

        op = ops.Limit(self, n, offset=offset)
        return op.to_expr()

    def head(self, n: int = 5) -> Table:
        """Select the first `n` rows of a table.

        The result set is not deterministic without a sort.

        Parameters
        ----------
        n
            Number of rows to include, defaults to 5

        Returns
        -------
        Table
            `table` limited to `n` rows
        """
        return self.limit(n=n)

    def sort_by(
        self,
        sort_exprs: str
        | ir.Column
        | ir.SortKey
        | tuple[str | ir.Column, bool]
        | Sequence[tuple[str | ir.Column, bool]],
    ) -> Table:
        """Sort table by `sort_exprs`

        Parameters
        ----------
        sort_exprs
            Sort specifications

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')])
        >>> ab_sorted = t.sort_by([('a', True), ('b', False)])

        Returns
        -------
        Table
            Sorted table
        """
        if isinstance(sort_exprs, tuple):
            sort_exprs = [sort_exprs]
        elif sort_exprs is None:
            sort_exprs = []
        else:
            sort_exprs = util.promote_list(sort_exprs)
        return self.op().sort_by(self, sort_exprs).to_expr()

    def union(
        self,
        right: Table,
        distinct: bool = False,
    ) -> Table:
        """Compute the set union of two table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        right
            Table expression
        distinct
            Only union distinct rows not occurring in the calling table (this
            can be very expensive, be careful)

        Returns
        -------
        Table
            Union of table and `right`
        """
        from ibis.expr import operations as ops

        return ops.Union(self, right, distinct=distinct).to_expr()

    def intersect(self, right: Table) -> Table:
        """Compute the set intersection of two table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        right
            Table expression

        Returns
        -------
        Table
            The rows common amongst `left` and `right`.
        """
        from ibis.expr import operations as ops

        return ops.Intersection(self, right).to_expr()

    def to_array(self) -> ir.Column:
        """View a single column table as an array.

        Returns
        -------
        Value
            A single column view of a table
        """
        from ibis.expr import operations as ops

        schema = self.schema()
        if len(schema) != 1:
            raise com.ExpressionError(
                'Table must have exactly one column when viewed as array'
            )

        return ops.TableArrayView(self).to_expr()

    def mutate(
        self,
        exprs: Sequence[ir.Expr] | None = None,
        **mutations: ir.Value,
    ) -> Table:
        """Add columns to a table expression.

        Parameters
        ----------
        exprs
            List of named expressions to add as columns
        mutations
            Named expressions using keyword arguments

        Returns
        -------
        Table
            Table expression with additional columns

        Examples
        --------
        Using keywords arguments to name the new columns

        >>> import ibis
        >>> table = ibis.table(
        ...     [('foo', 'double'), ('bar', 'double')],
        ...     name='t'
        ... )
        >>> expr = table.mutate(qux=table.foo + table.bar, baz=5)
        >>> expr
        r0 := UnboundTable[t]
          foo float64
          bar float64
        Selection[r0]
          selections:
            r0
            baz: 5
            qux: r0.foo + r0.bar

        Use the [`name`][ibis.expr.types.generic.Value.name] method to name
        the new columns.

        >>> new_columns = [ibis.literal(5).name('baz',),
        ...                (table.foo + table.bar).name('qux')]
        >>> expr2 = table.mutate(new_columns)
        >>> expr.equals(expr2)
        True

        """
        from ibis.expr import analysis as an
        from ibis.expr import rules as rlz

        exprs = [] if exprs is None else util.promote_list(exprs)
        for name, expr in sorted(
            mutations.items(), key=operator.itemgetter(0)
        ):
            if util.is_function(expr):
                value = expr(self)
            elif isinstance(expr, Deferred):
                value = expr.resolve(self)
            else:
                value = rlz.any(expr)
            exprs.append(value.name(name))

        mutation_exprs = an.get_mutation_exprs(exprs, self)
        return self.projection(mutation_exprs)

    def select(
        self,
        exprs: ir.Value | str | Sequence[ir.Value | str],
    ) -> Table:
        """Compute a new table expression using `exprs`.

        Passing an aggregate function to this method will broadcast the
        aggregate's value over the number of rows in the table and
        automatically constructs a window function expression. See the examples
        section for more details.

        Parameters
        ----------
        exprs
            Column expression, string, or list of column expressions and
            strings.

        Returns
        -------
        Table
            Table expression

        Examples
        --------
        Simple projection

        >>> import ibis
        >>> fields = [('a', 'int64'), ('b', 'double')]
        >>> t = ibis.table(fields, name='t')
        >>> proj = t.projection([t.a, (t.b + 1).name('b_plus_1')])
        >>> proj
        r0 := UnboundTable[t]
          a int64
          b float64
        Selection[r0]
          selections:
            a:        r0.a
            b_plus_1: r0.b + 1
        >>> proj2 = t[t.a, (t.b + 1).name('b_plus_1')]
        >>> proj.equals(proj2)
        True

        Aggregate projection

        >>> agg_proj = t[t.a.sum().name('sum_a'), t.b.mean().name('mean_b')]
        >>> agg_proj
        r0 := UnboundTable[t]
          a int64
          b float64
        Selection[r0]
          selections:
            sum_a:  Window(Sum(r0.a), window=Window(how='rows'))
            mean_b: Window(Mean(r0.b), window=Window(how='rows'))

        Note the `Window` objects here.

        Their existence means that the result of the aggregation will be
        broadcast across the number of rows in the input column.
        The purpose of this expression rewrite is to make it easy to write
        column/scalar-aggregate operations like

        >>> t[(t.a - t.a.mean()).name('demeaned_a')]
        r0 := UnboundTable[t]
          a int64
          b float64
        Selection[r0]
          selections:
            demeaned_a: r0.a - Window(Mean(r0.a), window=Window(how='rows'))
        """
        import ibis.expr.analysis as an

        if isinstance(exprs, (Expr, str)):
            exprs = [exprs]

        projector = an.Projector(self, exprs)
        op = projector.get_result()
        return op.to_expr()

    projection = select

    def relabel(self, substitutions: Mapping[str, str]) -> Table:
        """Change table column names, otherwise leaving table unaltered.

        Parameters
        ----------
        substitutions
            Name mapping

        Returns
        -------
        Table
            A relabeled table expression
        """
        observed = set()

        exprs = []
        for c in self.columns:
            expr = self[c]
            if c in substitutions:
                expr = expr.name(substitutions[c])
                observed.add(c)
            exprs.append(expr)

        for c in substitutions:
            if c not in observed:
                raise KeyError(f'{c!r} is not an existing column')

        return self.projection(exprs)

    def drop(self, fields: str | Sequence[str]) -> Table:
        """Remove fields from a table.

        Parameters
        ----------
        fields
            Fields to drop

        Returns
        -------
        Table
            Expression without `fields`
        """
        if not fields:
            # no-op if nothing to be dropped
            return self

        fields = util.promote_list(fields)

        schema = self.schema()
        field_set = frozenset(fields)
        missing_fields = field_set.difference(schema)

        if missing_fields:
            raise KeyError(f'Fields not in table: {missing_fields!s}')

        return self[[field for field in schema if field not in field_set]]

    def filter(
        self,
        predicates: ir.BooleanValue | Sequence[ir.BooleanValue],
    ) -> Table:
        """Select rows from `table` based on `predicates`.

        Parameters
        ----------
        predicates
            Boolean value expressions used to select rows in `table`.

        Returns
        -------
        Table
            Filtered table expression
        """
        from ibis.expr import analysis as an

        resolved_predicates, top_ks = _resolve_predicates(self, predicates)
        child = self
        for predicate, right in top_ks:
            child = child.semi_join(right, predicate)[child]
        return an.apply_filter(
            child,
            [
                an._rewrite_filter(pred.op(), pred)
                for pred in resolved_predicates
            ],
        )

    def count(self) -> ir.IntegerScalar:
        """Compute the number of rows in the table.

        Returns
        -------
        IntegerScalar
            Number of rows in the table
        """
        from ibis.expr import operations as ops

        return ops.Count(self, None).to_expr().name("count")

    def dropna(
        self,
        subset: Sequence[str] | None = None,
        how: Literal["any", "all"] = "any",
    ) -> Table:
        """Remove rows with null values from the table.

        Parameters
        ----------
        subset
            Columns names to consider when dropping nulls. By default all columns
            are considered.
        how
            Determine whether a row is removed if there is at least one null
            value in the row ('any'), or if all row values are null ('all').
            Options are 'any' or 'all'. Default is 'any'.

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')])
        >>> t = t.dropna()  # Drop all rows where any values are null
        >>> t = t.dropna(how='all')  # Only drop rows where all values are null
        >>> t = t.dropna(subset=['a'], how='all')  # Only drop rows where all values in column 'a' are null  # noqa: E501

        Returns
        -------
        Table
            Table expression
        """
        from ibis.expr import operations as ops

        if subset is None:
            subset = []
        subset = util.promote_list(subset)
        return ops.DropNa(self, how, subset).to_expr()

    def fillna(
        self,
        replacements: ir.Scalar | Mapping[str, ir.Scalar],
    ) -> Table:
        """Fill null values in a table expression.

        Parameters
        ----------
        replacements
            Value with which to fill the nulls. If passed as a mapping, the keys
            are column names that map to their replacement value. If passed
            as a scalar, all columns are filled with that value.

        Notes
        -----
        There is potential lack of type stability with the `fillna` API. For
        example, different library versions may impact whether or not a given
        backend promotes integer replacement values to floats.

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.datatypes as dt
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')])
        >>> t = t.fillna(0.0)  # Replace nulls in all columns with 0.0
        >>> t.fillna({c: 0.0 for c, t in t.schema().items() if t == dt.float64})
        r0 := UnboundTable[unbound_table_...]
          a int64
          b string
        r1 := FillNa[r0]
          replacements:
            0.0
        FillNa[r1]
          replacements:
            frozendict({})

        Returns
        -------
        Table
            Table expression
        """  # noqa: E501
        from ibis.expr import operations as ops

        if isinstance(replacements, collections.abc.Mapping):
            columns = replacements.keys()
            table_columns = self.schema().names
            invalid = set(columns) - set(table_columns)
            if invalid:
                raise com.IbisTypeError(
                    f'value {list(invalid)} is not a field in {table_columns}.'
                )
        return ops.FillNa(self, replacements).to_expr()

    def info(self, buf: IO[str] | None = None) -> None:
        """Show column names, types and null counts.

        Parameters
        ----------
        buf
            A writable buffer, defaults to stdout
        """
        metrics = [self[col].count().name(col) for col in self.columns]
        metrics.append(self.count().name("nrows"))

        schema = self.schema()

        *items, (_, n) = self.aggregate(metrics).execute().squeeze().items()

        tabulated = tabulate.tabulate(
            [
                (
                    column,
                    schema[column],
                    f"{n - non_nulls} ({100 * (1.0 - non_nulls / n):>3.3g}%)",
                )
                for column, non_nulls in items
            ],
            headers=["Column", "Type", "Nulls (%)"],
            colalign=("left", "left", "right"),
        )
        width = tabulated[tabulated.index("\n") + 1 :].index("\n")
        row_count = f"Rows: {n}".center(width)
        footer_line = "-" * width
        print("\n".join([tabulated, footer_line, row_count]), file=buf)

    def set_column(self, name: str, expr: ir.Value) -> Table:
        """Replace an existing column with a new expression.

        Parameters
        ----------
        name
            Column name to replace
        expr
            New data for column

        Returns
        -------
        Table
            Table expression with new columns
        """
        expr = self._ensure_expr(expr)

        if expr._safe_name != name:
            expr = expr.name(name)

        if name not in self:
            raise KeyError(f'{name} is not in the table')

        proj_exprs = []
        for key in self.columns:
            if key == name:
                proj_exprs.append(expr)
            else:
                proj_exprs.append(self[key])

        return self.projection(proj_exprs)

    def join(
        left: Table,
        right: Table,
        predicates: str
        | Sequence[
            str | tuple[str | ir.Column, str | ir.Column] | ir.BooleanColumn
        ] = (),
        how: Literal[
            'inner',
            'left',
            'outer',
            'right',
            'semi',
            'anti',
            'any_inner',
            'any_left',
            'left_semi',
        ] = 'inner',
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Table:
        """Perform a join between two tables.

        Parameters
        ----------
        left
            Left table to join
        right
            Right table to join
        predicates
            Boolean or column names to join on
        how
            Join method
        suffixes
            Left and right suffixes that will be used to rename overlapping
            columns.
        """
        from ibis.expr import analysis as an
        from ibis.expr import operations as ops
        from ibis.expr import types as ir

        _join_classes = {
            'inner': ops.InnerJoin,
            'left': ops.LeftJoin,
            'any_inner': ops.AnyInnerJoin,
            'any_left': ops.AnyLeftJoin,
            'outer': ops.OuterJoin,
            'right': ops.RightJoin,
            'left_semi': ops.LeftSemiJoin,
            'semi': ops.LeftSemiJoin,
            'anti': ops.LeftAntiJoin,
            'cross': ops.CrossJoin,
        }

        klass = _join_classes[how.lower()]
        if isinstance(predicates, ir.Expr):
            predicates = an.flatten_predicate(predicates)

        expr = klass(left, right, predicates).to_expr()

        # semi/anti join only give access to the left table's fields, so
        # there's never overlap
        if how in ("semi", "anti"):
            return expr

        return ops.relations._dedup_join_columns(
            expr,
            left=left,
            right=right,
            suffixes=suffixes,
        )

    def asof_join(
        left: Table,
        right: Table,
        predicates: str
        | ir.BooleanColumn
        | Sequence[str | ir.BooleanColumn] = (),
        by: str | ir.Column | Sequence[str | ir.Column] = (),
        tolerance: str | ir.IntervalScalar | None = None,
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Table:
        """Perform an "as-of" join between `left` and `right`.

        Similar to a left join except that the match is done on nearest key
        rather than equal keys.

        Optionally, match keys with `by` before joining with `predicates`.

        Parameters
        ----------
        left
            Table expression
        right
            Table expression
        predicates
            Join expressions
        by
            column to group by before joining
        tolerance
            Amount of time to look behind when joining
        suffixes
            Left and right suffixes that will be used to rename overlapping
            columns.

        Returns
        -------
        Table
            Table expression
        """
        from ibis.expr import operations as ops

        op = ops.AsOfJoin(
            left=left,
            right=right,
            predicates=predicates,
            by=by,
            tolerance=tolerance,
        )
        return ops.relations._dedup_join_columns(
            op.to_expr(),
            left=left,
            right=right,
            suffixes=suffixes,
        )

    def cross_join(
        left: Table,
        right: Table,
        *rest: Table,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Table:
        """Compute the cross join of a sequence of tables.

        Parameters
        ----------
        left
            Left table
        right
            Right table
        rest
            Additional tables to cross join
        suffixes
            Left and right suffixes that will be used to rename overlapping
            columns.

        Returns
        -------
        Table
            Cross join of `left`, `right` and `rest`

        Examples
        --------
        >>> import ibis
        >>> schemas = [(name, 'int64') for name in 'abcde']
        >>> a, b, c, d, e = [
        ...     ibis.table([(name, type)], name=name) for name, type in schemas
        ... ]
        >>> joined1 = ibis.cross_join(a, b, c, d, e)
        >>> joined1
        r0 := UnboundTable[e]
          e int64
        r1 := UnboundTable[d]
          d int64
        r2 := UnboundTable[c]
          c int64
        r3 := UnboundTable[b]
          b int64
        r4 := UnboundTable[a]
          a int64
        r5 := CrossJoin[r3, r2]
        r6 := CrossJoin[r5, r1]
        r7 := CrossJoin[r6, r0]
        CrossJoin[r4, r7]
        """
        from ibis.expr import operations as ops

        expr = ops.CrossJoin(
            left,
            functools.reduce(Table.cross_join, rest, right),
            [],
        ).to_expr()
        return ops.relations._dedup_join_columns(
            expr,
            left=left,
            right=right,
            suffixes=suffixes,
        )

    @util.deprecated(version="4.0", instead="")
    def prevent_rewrite(self, client=None) -> Table:  # pragma: no cover
        """Prevent optimization from happening below this expression.

        Only valid on SQL-string generating backends.

        Parameters
        ----------
        client
            A client to use to create the SQLQueryResult operation. This can be
            useful if you're compiling an expression that derives from an
            `UnboundTable` operation.

        Returns
        -------
        Table
            An opaque SQL query
        """
        from ibis.expr import operations as ops

        if client is None:
            client = self._find_backend()
        query = client.compile(self)
        return ops.SQLQueryResult(query, self.schema(), client).to_expr()

    inner_join = _regular_join_method("inner_join", "inner")
    left_join = _regular_join_method("left_join", "left")
    outer_join = _regular_join_method("outer_join", "outer")
    right_join = _regular_join_method("right_join", "right")
    semi_join = _regular_join_method("semi_join", "semi")
    anti_join = _regular_join_method("anti_join", "anti")
    any_inner_join = _regular_join_method("any_inner_join", "any_inner")
    any_left_join = _regular_join_method("any_left_join", "any_left")

    @util.deprecated(
        version="3.0",
        instead="remove the `.materialize()` call, it has no effect",
    )
    def materialize(self) -> Table:
        return self

    def alias(self, alias: str) -> ir.Table:
        """Create a table expression with a specific name `alias`.

        This method is useful for exposing an ibis expression to the underlying
        backend for use in the
        [`Table.sql`][ibis.expr.types.relations.Table.sql] method.

        !!! note "`.alias` will create a temporary view"

            `.alias` creates a temporary view in the database.

            This side effect will be removed in a future version of ibis and
            **is not part of the public API**.

        Parameters
        ----------
        alias
            Name of the child expression

        Returns
        -------
        Table
            An table expression

        Examples
        --------
        >>> con = ibis.duckdb.connect("ci/ibis-testing-data/ibis_testing.ddb")
        >>> t = con.table("functional_alltypes")
        >>> expr = t.alias("my_t").sql("SELECT sum(double_col) FROM my_t")
        >>> expr
        r0 := AlchemyTable: functional_alltypes
          index           int64
            ⋮
          month           int32
        r1 := View[r0]: my_t
          schema:
            index           int64
              ⋮
            month           int32
        SQLStringView[r1]: _ibis_view_0
          query: 'SELECT sum(double_col) FROM my_t'
          schema:
            sum(double_col) float64
        """
        import ibis.expr.operations as ops

        expr = ops.View(child=self, name=alias).to_expr()

        # NB: calling compile is necessary so that any temporary views are
        # created so that we can infer the schema without executing the entire
        # query
        expr.compile()
        return expr

    def sql(self, query: str) -> ir.Table:
        """Run a SQL query against a table expression.

        !!! note "The SQL string is backend specific"

            `query` must be valid SQL for the execution backend the expression
            will run against

        See [`Table.alias`][ibis.expr.types.relations.Table.alias] for
        details on using named table expressions in a SQL string.

        Parameters
        ----------
        query
            Query string

        Returns
        -------
        Table
            An opaque table expression

        Examples
        --------
        >>> con = ibis.duckdb.connect("ci/ibis-testing-data/ibis_testing.ddb")
        >>> t = con.table("functional_alltypes")
        >>> expr = t.sql("SELECT sum(double_col) FROM functional_alltypes")
        >>> expr
        r0 := AlchemyTable: functional_alltypes
          index           int64
            ⋮
          month           int32
        SQLStringView[r0]: _ibis_view_1
          query: 'SELECT sum(double_col) FROM functional_alltypes'
          schema:
            sum(double_col) float64
        """
        import ibis.expr.operations as ops

        return ops.SQLStringView(
            child=self,
            name=next(_ALIASES),
            query=query,
        ).to_expr()


def _resolve_predicates(
    table: Table, predicates
) -> tuple[list[ir.BooleanValue], list[tuple[ir.BooleanValue, ir.Table]]]:
    from ibis.expr import analysis as an
    from ibis.expr import operations as ops
    from ibis.expr import types as ir

    if isinstance(predicates, Expr):
        predicates = an.flatten_predicate(predicates)

    predicates = [
        ir.relations.bind_expr(table, pred)
        for pred in util.promote_list(predicates)
    ]

    resolved_predicates = []
    top_ks = []
    for pred in predicates:
        if isinstance(pred, ir.TopK):
            top_ks.append(pred._semi_join_components())
        elif isinstance(pred_op := pred.op(), ops.logical._UnresolvedSubquery):
            resolved_predicates.append(pred_op._resolve(table))
        else:
            resolved_predicates.append(pred)

    return resolved_predicates, top_ks


def bind_expr(table, expr):
    if isinstance(expr, (list, tuple)):
        return [bind_expr(table, x) for x in expr]

    return table._ensure_expr(expr)


@util.deprecated(
    version="4.0",
    instead="Use ibis.expr.analysis.find_first_base_table() instead",
)
def find_base_table(expr):  # pragma: no cover
    from ibis.expr.analysis import find_first_base_table

    return find_first_base_table(expr)


public(TableExpr=Table)
