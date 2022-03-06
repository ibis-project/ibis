from __future__ import annotations

import collections
import functools
import operator
from typing import IO, TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from cached_property import cached_property
from public import public

import ibis

from ... import util
from ...common import exceptions as com
from .core import Expr

if TYPE_CHECKING:
    from .. import schema as sch
    from .. import types as ir
    from .generic import ColumnExpr
    from .groupby import GroupedTableExpr


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
        self: TableExpr,
        right: TableExpr,
        predicates: str
        | Sequence[
            str
            | tuple[str | ir.ColumnExpr, str | ir.ColumnExpr]
            | ir.BooleanValue
        ] = (),
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> TableExpr:
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
        TableExpr
            Joined table
        """  # noqa: E501
        return self.join(right, predicates, how=how, suffixes=suffixes)

    f.__name__ = name
    return f


@public
class TableExpr(Expr):
    @property
    def _factory(self):
        def factory(arg):
            return TableExpr(arg)

        return factory

    def _type_display(self):
        return 'table'

    def _is_valid(self, exprs):
        try:
            self._assert_valid(util.promote_list(exprs))
        except com.RelationError:
            return False
        else:
            return True

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator

        ExprValidator([self]).validate_all(exprs)

    def __contains__(self, name):
        return name in self.schema()

    def _repr_html_(self) -> str | None:
        if not ibis.options.interactive:
            return None

        return self.execute()._repr_html_()

    def __getitem__(self, what):
        from .analytic import AnalyticExpr
        from .generic import ColumnExpr
        from .logical import BooleanColumn

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

        if isinstance(what, AnalyticExpr):
            what = what._table_getitem()

        if isinstance(what, (list, tuple, TableExpr)):
            # Projection case
            return self.projection(what)
        elif isinstance(what, BooleanColumn):
            # Boolean predicate
            return self.filter([what])
        elif isinstance(what, ColumnExpr):
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

    def _resolve(self, exprs):
        exprs = util.promote_list(exprs)
        return list(self._ensure_expr(x) for x in exprs)

    def _ensure_expr(self, expr):
        if isinstance(expr, str):
            return self[expr]
        elif isinstance(expr, (int, np.integer)):
            return self[self.schema().name_at_position(expr)]
        elif not isinstance(expr, Expr):
            return expr(self)
        else:
            return expr

    def _get_type(self, name):
        return self._arg.get_type(name)

    def get_columns(self, iterable: Iterable[str]) -> list[ColumnExpr]:
        """Get multiple columns from the table.

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
        list[ir.ColumnExpr]
            List of column expressions
        """
        return [self.get_column(x) for x in iterable]

    def get_column(self, name: str) -> ColumnExpr:
        """Get a reference to a single column from the table.

        Returns
        -------
        ColumnExpr
            A column named `name`.
        """
        import ibis.expr.operations as ops

        ref = ops.TableColumn(self, name)
        return ref.to_expr()

    @cached_property
    def columns(self):
        return self.schema().names

    def schema(self) -> sch.Schema:
        """Return the table's schema.

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
    ) -> GroupedTableExpr:
        """Create a grouped table expression.

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
        GroupedTableExpr
            A grouped table expression
        """
        from .groupby import GroupedTableExpr

        return GroupedTableExpr(self, by, **additional_grouping_expressions)

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
        >>> my_table[my_table.rowid(), my_table.name].execute()
        1|Ibis
        2|pandas
        3|Dask
        """
        from .. import operations as ops

        return ops.RowID().to_expr()

    def view(self) -> TableExpr:
        """Create a new table expression distinct from the current one.

        Use this API for any self-referencing operations like a self-join.

        Returns
        -------
        TableExpr
            Table expression
        """
        from .. import operations as ops

        return ops.SelfReference(self).to_expr()

    def difference(self, right: TableExpr) -> TableExpr:
        """Compute the set difference of two table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        right
            Table expression

        Returns
        -------
        TableExpr
            The rows present in `left` that are not present in `right`.
        """
        from .. import operations as ops

        return ops.Difference(self, right).to_expr()

    def aggregate(
        self,
        metrics: Sequence[ir.ScalarExpr] | None = None,
        by: Sequence[ir.ValueExpr] | None = None,
        having: Sequence[ir.BooleanValue] | None = None,
        **kwargs: ir.ValueExpr,
    ) -> TableExpr:
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
        TableExpr
            An aggregate table expression
        """
        metrics = [] if metrics is None else util.promote_list(metrics)
        metrics.extend(
            self._ensure_expr(expr).name(name)
            for name, expr in sorted(
                kwargs.items(), key=operator.itemgetter(0)
            )
        )

        op = self.op().aggregate(
            self,
            metrics,
            by=util.promote_list(by if by is not None else []),
            having=util.promote_list(having if having is not None else []),
        )
        return op.to_expr()

    def distinct(self) -> TableExpr:
        """Compute the set of unique rows in the table."""
        from .. import operations as ops

        return ops.Distinct(self).to_expr()

    def limit(self, n: int, offset: int = 0) -> TableExpr:
        """Select the first `n` rows at beginning of table starting at `offset`.

        Parameters
        ----------
        n
            Number of rows to include
        offset
            Number of rows to skip first

        Returns
        -------
        TableExpr
            The first `n` rows of `table` starting at `offset`
        """
        from .. import operations as ops

        op = ops.Limit(self, n, offset=offset)
        return op.to_expr()

    def head(self, n: int = 5) -> TableExpr:
        """Select the first `n` rows of a table.

        The result set is not deterministic without a sort.

        Parameters
        ----------
        n
            Number of rows to include, defaults to 5

        Returns
        -------
        TableExpr
            `table` limited to `n` rows
        """
        return self.limit(n=n)

    def sort_by(
        self,
        sort_exprs: str
        | ir.ColumnExpr
        | ir.SortKey
        | tuple[str | ir.ColumnExpr, bool]
        | Sequence[tuple[str | ir.ColumnExpr, bool]],
    ) -> TableExpr:
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
        TableExpr
            Sorted table
        """
        return (
            self.op()
            .sort_by(
                self,
                [] if sort_exprs is None else util.promote_list(sort_exprs),
            )
            .to_expr()
        )

    def union(
        self,
        right: TableExpr,
        distinct: bool = False,
    ) -> TableExpr:
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
        TableExpr
            Union of table and `right`
        """
        from .. import operations as ops

        return ops.Union(self, right, distinct=distinct).to_expr()

    def intersect(self, right: TableExpr) -> TableExpr:
        """Compute the set intersection of two table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        right
            Table expression

        Returns
        -------
        TableExpr
            The rows common amongst `left` and `right`.
        """
        from .. import operations as ops

        return ops.Intersection(self, right).to_expr()

    def to_array(self) -> ir.ColumnExpr:
        """View a single column table as an array.

        Returns
        -------
        ValueExpr
            A single column view of a table
        """
        from .. import operations as ops

        schema = self.schema()
        if len(schema) != 1:
            raise com.ExpressionError(
                'Table must have exactly one column when viewed as array'
            )

        return ops.TableArrayView(self).to_expr()

    def _safe_get_name(expr):
        try:
            return expr.get_name()
        except com.ExpressionError:
            return None

    def mutate(
        self,
        exprs: Sequence[ir.Expr] | None = None,
        **mutations: ir.ValueExpr,
    ) -> TableExpr:
        """Add columns to a table expression.

        Parameters
        ----------
        exprs
            List of named expressions to add as columns
        mutations
            Named expressions using keyword arguments

        Returns
        -------
        TableExpr
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
        >>> expr  # doctest: +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: t
          schema:
            foo : float64
            bar : float64
        <BLANKLINE>
        Selection[table]
          table:
            Table: ref_0
          selections:
            Table: ref_0
            baz = Literal[int8]
              5
            qux = Add[float64*]
              left:
                foo = Column[float64*] 'foo' from table
                  ref_0
              right:
                bar = Column[float64*] 'bar' from table
                  ref_0

        Use the [`Expr.name`][ibis.expr.types.Expr.name] method to name the new
        columns.

        >>> new_columns = [ibis.literal(5).name('baz',),
        ...                (table.foo + table.bar).name('qux')]
        >>> expr2 = table.mutate(new_columns)
        >>> expr.equals(expr2)
        True

        """
        from .. import analysis as an
        from .. import rules as rlz

        exprs = [] if exprs is None else util.promote_list(exprs)
        for name, expr in sorted(
            mutations.items(), key=operator.itemgetter(0)
        ):
            if util.is_function(expr):
                value = expr(self)
            else:
                value = rlz.any(expr)
            exprs.append(value.name(name))

        mutation_exprs = an.get_mutation_exprs(exprs, self)
        return self.projection(mutation_exprs)

    def select(
        self,
        exprs: ir.ValueExpr | str | Sequence[ir.ValueExpr | str],
    ) -> TableExpr:
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
        TableExpr
            Table expression

        Examples
        --------
        Simple projection

        >>> import ibis
        >>> fields = [('a', 'int64'), ('b', 'double')]
        >>> t = ibis.table(fields, name='t')
        >>> proj = t.projection([t.a, (t.b + 1).name('b_plus_1')])
        >>> proj  # doctest: +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: t
          schema:
            a : int64
            b : float64
        <BLANKLINE>
        Selection[table]
          table:
            Table: ref_0
          selections:
            a = Column[int64*] 'a' from table
              ref_0
            b_plus_1 = Add[float64*]
              left:
                b = Column[float64*] 'b' from table
                  ref_0
              right:
                Literal[int8]
                  1
        >>> proj2 = t[t.a, (t.b + 1).name('b_plus_1')]
        >>> proj.equals(proj2)
        True

        Aggregate projection

        >>> agg_proj = t[t.a.sum().name('sum_a'), t.b.mean().name('mean_b')]
        >>> agg_proj  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        ref_0
        UnboundTable[table]
          name: t
          schema:
            a : int64
            b : float64
        <BLANKLINE>
        Selection[table]
          table:
            Table: ref_0
          selections:
            sum_a = WindowOp[int64*]
              sum_a = Sum[int64]
                a = Column[int64*] 'a' from table
                  ref_0
                where:
                  None
              <ibis.expr.window.Window object at 0x...>
            mean_b = WindowOp[float64*]
              mean_b = Mean[float64]
                b = Column[float64*] 'b' from table
                  ref_0
                where:
                  None
              <ibis.expr.window.Window object at 0x...>

        Note the [`Window`][ibis.expr.window.Window] objects here.

        Their existence means that the result of the aggregation will be
        broadcast across the number of rows in the input column.
        The purpose of this expression rewrite is to make it easy to write
        column/scalar-aggregate operations like

        >>> t[(t.a - t.a.mean()).name('demeaned_a')]
        """
        import ibis.expr.analysis as an

        if isinstance(exprs, (Expr, str)):
            exprs = [exprs]

        projector = an.Projector(self, exprs)
        op = projector.get_result()
        return op.to_expr()

    projection = select

    def relabel(self, substitutions: Mapping[str, str]) -> TableExpr:
        """Change table column names, otherwise leaving table unaltered.

        Parameters
        ----------
        substitutions
            Name mapping

        Returns
        -------
        TableExpr
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

    def drop(self, fields: str | Sequence[str]) -> TableExpr:
        """Remove fields from a table.

        Parameters
        ----------
        fields
            Fields to drop

        Returns
        -------
        TableExpr
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
    ) -> TableExpr:
        """Select rows from `table` based on `predicates`.

        Parameters
        ----------
        predicates
            Boolean value expressions used to select rows in `table`.

        Returns
        -------
        TableExpr
            Filtered table expression
        """
        from .. import analysis as an

        resolved_predicates = _resolve_predicates(self, predicates)
        return an.apply_filter(self, resolved_predicates)

    def count(self) -> ir.IntegerScalar:
        """Compute the number of rows in the table.

        Returns
        -------
        IntegerScalar
            Number of rows in the table
        """
        from .. import operations as ops

        return ops.Count(self, None).to_expr().name("count")

    def dropna(
        self,
        subset: Sequence[str] | None = None,
        how: Literal["any", "all"] = "any",
    ) -> TableExpr:
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
        TableExpr
            Table expression
        """
        from .. import operations as ops

        if subset is None:
            subset = []
        subset = util.promote_list(subset)
        return ops.DropNa(self, how, subset).to_expr()

    def fillna(
        self,
        replacements: ir.ScalarExpr | Mapping[str, ir.ScalarExpr],
    ) -> TableExpr:
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
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')])
        >>> t = t.fillna(0.0)  # Replace nulls in all columns with 0.0
        >>> t.fillna({c: 0.0 for c, t in t.schema().items() if t == dt.float64})  # Replace all na values in all columns of a given type with the same value  # noqa: E501

        Returns
        -------
        TableExpr
            Table expression
        """
        from .. import operations as ops

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
        metrics = [self.count().name("nrows")]
        for col in self.columns:
            metrics.append(self[col].count().name(col))

        metrics = self.aggregate(metrics).execute().loc[0]

        names = ["Column", "------"] + self.columns
        types = ["Type", "----"] + [repr(x) for x in self.schema().types]
        counts = ["Non-null #", "----------"] + [str(x) for x in metrics[1:]]
        col_metrics = util.adjoin(2, names, types, counts)
        result = f"Table rows: {metrics[0]}\n\n{col_metrics}"

        print(result, file=buf)

    def set_column(self, name: str, expr: ir.ValueExpr) -> TableExpr:
        """Replace an existing column with a new expression.

        Parameters
        ----------
        table
            Table expression
        name
            Column name to replace
        expr
            New data for column

        Returns
        -------
        TableExpr
            Table expression
        """
        expr = self._ensure_expr(expr)

        if expr._name != name:
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
        left: TableExpr,
        right: TableExpr,
        predicates: str
        | Sequence[
            str
            | tuple[str | ir.ColumnExpr, str | ir.ColumnExpr]
            | ir.BooleanColumn
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
    ) -> TableExpr:
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
        from .. import analysis as an
        from .. import operations as ops
        from .. import types as ir

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
        left: TableExpr,
        right: TableExpr,
        predicates: str
        | ir.BooleanColumn
        | Sequence[str | ir.BooleanColumn] = (),
        by: str | ir.ColumnExpr | Sequence[str | ir.ColumnExpr] = (),
        tolerance: str | ir.IntervalScalar | None = None,
        *,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> TableExpr:
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
        TableExpr
            Table expression
        """
        from .. import operations as ops

        expr = ops.AsOfJoin(left, right, predicates, by, tolerance).to_expr()
        return ops.relations._dedup_join_columns(
            expr,
            left=left,
            right=right,
            suffixes=suffixes,
        )

    def cross_join(
        left: TableExpr,
        right: TableExpr,
        *rest: TableExpr,
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> TableExpr:
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
        TableExpr
            Cross join of `left`, `right` and `rest`

        Examples
        --------
        >>> import ibis
        >>> schemas = [(name, 'int64') for name in 'abcde']
        >>> a, b, c, d, e = [
        ...     ibis.table([(name, type)], name=name) for name, type in schemas
        ... ]
        >>> joined1 = ibis.cross_join(a, b, c, d, e)
        >>> joined1  # doctest: +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: a
          schema:
            a : int64
        ref_1
        UnboundTable[table]
          name: b
          schema:
            b : int64
        ref_2
        UnboundTable[table]
          name: c
          schema:
            c : int64
        ref_3
        UnboundTable[table]
          name: d
          schema:
            d : int64
        ref_4
        UnboundTable[table]
          name: e
          schema:
            e : int64
        CrossJoin[table]
          left:
            Table: ref_0
          right:
            CrossJoin[table]
              left:
                CrossJoin[table]
                  left:
                    CrossJoin[table]
                      left:
                        Table: ref_1
                      right:
                        Table: ref_2
                  right:
                    Table: ref_3
              right:
                Table: ref_4
        """
        from .. import operations as ops

        expr = ops.CrossJoin(
            left,
            functools.reduce(TableExpr.cross_join, rest, right),
            [],
        ).to_expr()
        return ops.relations._dedup_join_columns(
            expr,
            left=left,
            right=right,
            suffixes=suffixes,
        )

    def prevent_rewrite(self, client=None) -> TableExpr:
        """Prevent optimization from happening below this expression.

        Only valid on SQL-string generating backends.

        Parameters
        ----------
        client
            A client to use to create the SQLQueryResult operation. This is
            useful if you're compiling an expression that derives from an
            [`UnboundTable`][ibis.expr.operations.UnboundTable] operation.

        Returns
        -------
        sql_query_result : TableExpr
        """
        from .. import operations as ops

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


def _resolve_predicates(table: TableExpr, predicates) -> list[ir.BooleanValue]:
    from .. import analysis as an
    from .. import types as ir

    if isinstance(predicates, Expr):
        predicates = an.flatten_predicate(predicates)
    predicates = util.promote_list(predicates)
    predicates = [ir.relations.bind_expr(table, x) for x in predicates]
    resolved_predicates = []
    for pred in predicates:
        if isinstance(pred, ir.AnalyticExpr):
            pred = pred.to_filter()
        resolved_predicates.append(pred)

    return resolved_predicates


def bind_expr(table, expr):
    if isinstance(expr, (list, tuple)):
        return [bind_expr(table, x) for x in expr]

    return table._ensure_expr(expr)


# TODO: move to analysis
def find_base_table(expr):
    if isinstance(expr, TableExpr):
        return expr

    for arg in expr.op().flat_args():
        if isinstance(arg, Expr):
            r = find_base_table(arg)
            if isinstance(r, TableExpr):
                return r
