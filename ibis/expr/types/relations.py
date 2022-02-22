from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from public import public

import ibis
import ibis.common.exceptions as com
import ibis.util as util

from .core import Expr

if TYPE_CHECKING:
    import ibis.expr.schema as sch

    from .generic import ColumnExpr
    from .groupby import GroupedTableExpr


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

    def __rich_console__(self, console, options):
        from rich.table import Table

        nrows = ibis.options.repr.rows
        result = self.limit(nrows + 1).execute()

        table = Table(highlight=True)

        columns = self.columns
        for column in columns:
            table.add_column(column, justify="center")

        for row in result.iloc[:-1].itertuples(index=False):
            table.add_row(*map(repr, row))

        if len(result) > nrows:
            table.add_row(*itertools.repeat("⋮", len(columns)))

        return console.render(table, options=options)

    def __contains__(self, name):
        return name in self.schema()

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

    @property
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

        Notes
        -----
        `group_by` and `groupby` are equivalent.

        `groupby` exists for familiarity for pandas users.

        Returns
        -------
        GroupedTableExpr
            A grouped table expression
        """
        from .groupby import GroupedTableExpr

        return GroupedTableExpr(self, by, **additional_grouping_expressions)

    groupby = group_by


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
