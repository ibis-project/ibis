from __future__ import annotations

import collections
import contextlib
import functools
import itertools
import re
import warnings
from keyword import iskeyword
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Mapping, Sequence

import toolz
from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis import util
from ibis.expr.deferred import Deferred
from ibis.expr.types.core import Expr, _FixedTextJupyterMixin

if TYPE_CHECKING:
    import pandas as pd

    import ibis.expr.schema as sch
    import ibis.expr.selectors as s
    import ibis.expr.types as ir
    from ibis.common.typing import SupportsSchema
    from ibis.expr.selectors import IfAnyAll, Selector
    from ibis.expr.types.groupby import GroupedTable

_ALIASES = (f"_ibis_view_{n:d}" for n in itertools.count())


def _ensure_expr(table, expr):
    import ibis.expr.rules as rlz
    from ibis.expr.selectors import Selector

    # This is different than self._ensure_expr, since we don't want to
    # treat `str` or `int` values as column indices
    if util.is_function(expr):
        return expr(table)
    elif isinstance(expr, Deferred):
        return expr.resolve(table)
    elif isinstance(expr, Selector):
        return expr.expand(table)
    else:
        return rlz.any(expr).to_expr()


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
    def f(  # noqa: D417
        self: Table,
        right: Table,
        predicates: str
        | Sequence[
            str | tuple[str | ir.Column, str | ir.Column] | ir.BooleanValue
        ] = (),
        suffixes: tuple[str, str] = ("_x", "_y"),
    ) -> Table:
        """Perform a join between two tables.

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
        """
        return self.join(right, predicates, how=how, suffixes=suffixes)

    f.__name__ = name
    return f


@public
class Table(Expr, _FixedTextJupyterMixin):
    # Higher than numpy & dask objects
    __array_priority__ = 20

    __array_ufunc__ = None

    def __array__(self, dtype=None):
        return self.execute().__array__(dtype)

    def as_table(self) -> Table:
        """Promote the expression to a table.

        This method is a no-op for table expressions.

        Returns
        -------
        Table
            A table expression

        Examples
        --------
        >>> t = ibis.table(dict(a="int"), name="t")
        >>> s = t.as_table()
        >>> t is s
        True
        """
        return self

    def __contains__(self, name: str) -> bool:
        """Return whether `name` is a column in the table.

        Parameters
        ----------
        name
            Possible column name

        Returns
        -------
        bool
            Whether `name` is a column in `self`

        Examples
        --------
        >>> t = ibis.table(dict(a="string", b="float"), name="t")
        >>> "a" in t
        True
        >>> "c" in t
        False
        """
        return name in self.schema()

    def cast(self, schema: SupportsSchema) -> Table:
        """Cast the columns of a table.

        !!! note "If you need to cast columns to a single type, use [selectors](https://ibis-project.org/blog/selectors/)."

        Parameters
        ----------
        schema
            Mapping, schema or iterable of pairs to use for casting

        Returns
        -------
        Table
            Casted table

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.selectors as s
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.schema()
        ibis.Schema {
          species            string
          island             string
          bill_length_mm     float64
          bill_depth_mm      float64
          flipper_length_mm  int64
          body_mass_g        int64
          sex                string
          year               int64
        }
        >>> cols = ["body_mass_g", "bill_length_mm"]
        >>> t[cols].head()
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ body_mass_g ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ int64       │ float64        │
        ├─────────────┼────────────────┤
        │        3750 │           39.1 │
        │        3800 │           39.5 │
        │        3250 │           40.3 │
        │           ∅ │            nan │
        │        3450 │           36.7 │
        └─────────────┴────────────────┘

        Columns not present in the input schema will be passed through unchanged

        >>> t.columns
        ['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
        >>> expr = t.cast({"body_mass_g": "float64", "bill_length_mm": "int"})
        >>> expr.select(*cols).head()
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ body_mass_g ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ float64     │ int64          │
        ├─────────────┼────────────────┤
        │      3750.0 │             39 │
        │      3800.0 │             40 │
        │      3250.0 │             40 │
        │         nan │              ∅ │
        │      3450.0 │             37 │
        └─────────────┴────────────────┘

        Columns that are in the input `schema` but not in the table raise an error

        >>> t.cast({"foo": "string"})
        Traceback (most recent call last):
            ...
        ibis.common.exceptions.IbisError: Cast schema has fields that are not in the table: ['foo']
        """
        schema = sch.schema(schema)

        cols = []

        columns = self.columns
        if missing_fields := frozenset(schema.names).difference(columns):
            raise com.IbisError(
                f"Cast schema has fields that are not in the table: {sorted(missing_fields)}"
            )

        for col in columns:
            if (new_type := schema.get(col)) is not None:
                new_col = self[col].cast(new_type).name(col)
            else:
                new_col = col
            cols.append(new_col)
        return self.select(*cols)

    def __rich_console__(self, console, options):
        from rich.text import Text

        from ibis.expr.types.pretty import to_rich_table

        if not ibis.options.interactive:
            return console.render(Text(self._repr()), options=options)

        if console.is_jupyter:
            # Rich infers a console width in jupyter notebooks, but since
            # notebooks can use horizontal scroll bars we don't want to apply a
            # limit here. Since rich requires an integer for max_width, we
            # choose an arbitrarily large integer bound. Note that we need to
            # handle this here rather than in `to_rich_table`, as this setting
            # also needs to be forwarded to `console.render`.
            options = options.update(max_width=1_000_000)
            width = None
        else:
            width = options.max_width

        table = to_rich_table(self, width)
        return console.render(table, options=options)

    def __getitem__(self, what):
        """Select items from a table expression.

        This method implements square bracket syntax for table expressions,
        including various forms of projection and filtering.

        Parameters
        ----------
        what
            Selection object. This can be a variety of types including strings, ints, lists.

        Returns
        -------
        Table | Column
            The return type depends on the input. For a single string or int
            input a column is returned, otherwise a table is returned.

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.selectors as s
        >>> from ibis import _
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
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Return a column by name

        >>> t["island"]
        ┏━━━━━━━━━━━┓
        ┃ island    ┃
        ┡━━━━━━━━━━━┩
        │ string    │
        ├───────────┤
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ …         │
        └───────────┘

        Return the second column, starting from index 0

        >>> t.columns[1]
        'island'
        >>> t[1]
        ┏━━━━━━━━━━━┓
        ┃ island    ┃
        ┡━━━━━━━━━━━┩
        │ string    │
        ├───────────┤
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ …         │
        └───────────┘

        Extract a range of rows

        >>> t[:2]
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t[:5]
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t[2:5]
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Select columns

        >>> t[["island", "bill_length_mm"]].head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │            nan │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘
        >>> t["island", "bill_length_mm"].head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │            nan │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘
        >>> t[_.island, _.bill_length_mm].head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │            nan │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘

        Filtering

        >>> t[t.island.lower() != "torgersen"].head()
        ┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string │ float64        │ float64       │ int64             │ … │
        ├─────────┼────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Biscoe │           37.8 │          18.3 │               174 │ … │
        │ Adelie  │ Biscoe │           37.7 │          18.7 │               180 │ … │
        │ Adelie  │ Biscoe │           35.9 │          19.2 │               189 │ … │
        │ Adelie  │ Biscoe │           38.2 │          18.1 │               185 │ … │
        │ Adelie  │ Biscoe │           38.8 │          17.2 │               180 │ … │
        └─────────┴────────┴────────────────┴───────────────┴───────────────────┴───┘

        Selectors

        >>> t[~s.numeric() | (s.numeric() & ~s.c("year"))].head()
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t[s.r["bill_length_mm":"body_mass_g"]].head()
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ float64        │ float64       │ int64             │ int64       │
        ├────────────────┼───────────────┼───────────────────┼─────────────┤
        │           39.1 │          18.7 │               181 │        3750 │
        │           39.5 │          17.4 │               186 │        3800 │
        │           40.3 │          18.0 │               195 │        3250 │
        │            nan │           nan │                 ∅ │           ∅ │
        │           36.7 │          19.3 │               193 │        3450 │
        └────────────────┴───────────────┴───────────────────┴─────────────┘
        """
        from ibis.expr.types.generic import Column
        from ibis.expr.types.logical import BooleanValue

        if isinstance(what, (str, int)):
            return ops.TableColumn(self, what).to_expr()

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

        if isinstance(what, (list, tuple, Table)):
            # Projection case
            return self.select(what)
        elif isinstance(what, BooleanValue):
            # Boolean predicate
            return self.filter([what])
        elif isinstance(what, Column):
            # Projection convenience
            return self.select(what)
        else:
            raise NotImplementedError(
                'Selection rows or columns with {} objects is not '
                'supported'.format(type(what).__name__)
            )

    def __len__(self):
        raise com.ExpressionError('Use .count() instead')

    def __getattr__(self, key: str) -> ir.Column:
        """Return the column name of a table.

        Parameters
        ----------
        key
            Column name

        Returns
        -------
        Column
            Column expression with name `key`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.island
        ┏━━━━━━━━━━━┓
        ┃ island    ┃
        ┡━━━━━━━━━━━┩
        │ string    │
        ├───────────┤
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ Torgersen │
        │ …         │
        └───────────┘
        """
        with contextlib.suppress(com.IbisTypeError):
            return ops.TableColumn(self, key).to_expr()

        # Handle deprecated `groupby` and `sort_by` methods
        if key == "groupby":
            warnings.warn(
                "`Table.groupby` is deprecated and will be removed in 5.0, "
                "use `Table.group_by` instead",
                FutureWarning,
            )
            return self.group_by
        elif key == "sort_by":
            warnings.warn(
                "`Table.sort_by` is deprecated and will be removed in 5.0, "
                "use `Table.order_by` instead",
                FutureWarning,
            )
            return self.order_by

        # A mapping of common attribute typos, mapping them to the proper name
        common_typos = {
            "sort": "order_by",
            "sort_by": "order_by",
            "sortby": "order_by",
            "orderby": "order_by",
            "groupby": "group_by",
        }
        if key in common_typos:
            hint = common_typos[key]
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {key!r}, did you mean {hint!r}"
            )
        raise AttributeError(f"'Table' object has no attribute {key!r}")

    def __dir__(self) -> list[str]:
        out = set(dir(type(self)))
        out.update(c for c in self.columns if c.isidentifier() and not iskeyword(c))
        return sorted(out)

    def _ipython_key_completions_(self) -> list[str]:
        return self.columns

    def _ensure_expr(self, expr):
        import numpy as np

        from ibis.expr.selectors import Selector

        if isinstance(expr, str):
            # treat strings as column names
            return self[expr]
        elif isinstance(expr, (int, np.integer)):
            # treat Python integers as a column index
            return self[self.schema().name_at_position(expr)]
        elif isinstance(expr, Deferred):
            # resolve deferred expressions
            return expr.resolve(self)
        elif isinstance(expr, Selector):
            return expr.expand(self)
        elif callable(expr):
            return expr(self)
        else:
            return expr

    @property
    def columns(self) -> list[str]:
        """The list of columns in this table.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.starwars.fetch()
        >>> t.columns
        ['name',
         'height',
         'mass',
         'hair_color',
         'skin_color',
         'eye_color',
         'birth_year',
         'sex',
         'gender',
         'homeworld',
         'species',
         'films',
         'vehicles',
         'starships']
        """
        return list(self.schema().names)

    def schema(self) -> sch.Schema:
        """Return the schema for this table.

        Returns
        -------
        Schema
            The table's schema.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.starwars.fetch()
        >>> t.schema()
        ibis.Schema {
          name        string
          height      int64
          mass        float64
          hair_color  string
          skin_color  string
          eye_color   string
          birth_year  float64
          sex         string
          gender      string
          homeworld   string
          species     string
          films       string
          vehicles    string
          starships   string
        }
        """
        return self.op().schema

    def group_by(
        self,
        by: str | ir.Value | Iterable[str] | Iterable[ir.Value] | None = None,
        **key_exprs: str | ir.Value | Iterable[str] | Iterable[ir.Value],
    ) -> GroupedTable:
        """Create a grouped table expression.

        Parameters
        ----------
        by
            Grouping expressions
        key_exprs
            Named grouping expressions

        Returns
        -------
        GroupedTable
            A grouped table expression

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"fruit": ["apple", "apple", "banana", "orange"], "price": [0.5, 0.5, 0.25, 0.33]})
        >>> t
        ┏━━━━━━━━┳━━━━━━━━━┓
        ┃ fruit  ┃ price   ┃
        ┡━━━━━━━━╇━━━━━━━━━┩
        │ string │ float64 │
        ├────────┼─────────┤
        │ apple  │    0.50 │
        │ apple  │    0.50 │
        │ banana │    0.25 │
        │ orange │    0.33 │
        └────────┴─────────┘
        >>> t.group_by("fruit").agg(total_cost=_.price.sum(), avg_cost=_.price.mean())
        ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ fruit  ┃ total_cost ┃ avg_cost ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
        │ string │ float64    │ float64  │
        ├────────┼────────────┼──────────┤
        │ apple  │       1.00 │     0.50 │
        │ banana │       0.25 │     0.25 │
        │ orange │       0.33 │     0.33 │
        └────────┴────────────┴──────────┘
        """
        from ibis.expr.types.groupby import GroupedTable

        return GroupedTable(self, by, **key_exprs)

    def rowid(self) -> ir.IntegerValue:
        """A unique integer per row.

        !!! note "This operation is only valid on physical tables"

            Any further meaning behind this expression is backend dependent.
            Generally this corresponds to some index into the database storage
            (for example, sqlite or duckdb's `rowid`).

        For a monotonically increasing row number, see `ibis.row_number`.

        Returns
        -------
        IntegerColumn
            An integer column
        """
        if not isinstance(self.op(), ops.PhysicalTable):
            raise com.IbisTypeError(
                "rowid() is only valid for physical tables, not for generic "
                "table expressions"
            )
        return ops.RowID(self).to_expr()

    def view(self) -> Table:
        """Create a new table expression distinct from the current one.

        Use this API for any self-referencing operations like a self-join.

        Returns
        -------
        Table
            Table expression
        """
        return ops.SelfReference(self).to_expr()

    def difference(self, *tables: Table, distinct: bool = True) -> Table:
        """Compute the set difference of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        tables
            One or more table expressions
        distinct
            Only diff distinct rows not occurring in the calling table

        Returns
        -------
        Table
            The rows present in `self` that are not present in `tables`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t1 = ibis.memtable({"a": [1, 2]})
        >>> t1
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        └───────┘
        >>> t2 = ibis.memtable({"a": [2, 3]})
        >>> t2
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        │     3 │
        └───────┘
        >>> t1.difference(t2)
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        └───────┘
        """
        left = self
        if not tables:
            raise com.IbisTypeError(
                "difference requires a table or tables to compare against"
            )
        for right in tables:
            left = ops.Difference(left, right, distinct=distinct)
        return left.to_expr()

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

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"fruit": ["apple", "apple", "banana", "orange"], "price": [0.5, 0.5, 0.25, 0.33]})
        >>> t
        ┏━━━━━━━━┳━━━━━━━━━┓
        ┃ fruit  ┃ price   ┃
        ┡━━━━━━━━╇━━━━━━━━━┩
        │ string │ float64 │
        ├────────┼─────────┤
        │ apple  │    0.50 │
        │ apple  │    0.50 │
        │ banana │    0.25 │
        │ orange │    0.33 │
        └────────┴─────────┘
        >>> t.aggregate(by=["fruit"], total_cost=_.price.sum(), avg_cost=_.price.mean(), having=_.price.sum() < 0.5)
        ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ fruit  ┃ total_cost ┃ avg_cost ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
        │ string │ float64    │ float64  │
        ├────────┼────────────┼──────────┤
        │ banana │       0.25 │     0.25 │
        │ orange │       0.33 │     0.33 │
        └────────┴────────────┴──────────┘
        """
        import ibis.expr.analysis as an

        metrics = itertools.chain(
            itertools.chain.from_iterable(
                (
                    (_ensure_expr(self, m) for m in metric)
                    if isinstance(metric, (list, tuple))
                    else util.promote_list(_ensure_expr(self, metric))
                )
                for metric in util.promote_list(metrics)
            ),
            (
                e.name(name)
                for name, expr in kwargs.items()
                for e in util.promote_list(_ensure_expr(self, expr))
            ),
        )

        agg = ops.Aggregation(
            self,
            metrics=list(metrics),
            by=util.promote_list(by),
            having=util.promote_list(having),
        )
        agg = an.simplify_aggregation(agg)

        return agg.to_expr()

    agg = aggregate

    def distinct(self) -> Table:
        """Compute the unique rows in `self`.

        Returns
        -------
        Table
            Unique rows of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [1, 1, 2], "b": ["c", "a", "a"]})
        >>> t[["a"]].distinct()
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        └───────┘
        >>> t.distinct()
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ c      │
        │     1 │ a      │
        │     2 │ a      │
        └───────┴────────┘
        """
        return ops.Distinct(self).to_expr()

    def limit(self, n: int, offset: int = 0) -> Table:
        """Select `n` rows from `self` starting at `offset`.

        !!! note "The result set is not deterministic without a call to [`order_by`][ibis.expr.types.relations.Table.order_by]."

        Parameters
        ----------
        n
            Number of rows to include
        offset
            Number of rows to skip first

        Returns
        -------
        Table
            The first `n` rows of `self` starting at `offset`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [1, 1, 2], "b": ["c", "a", "a"]})
        >>> t
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ c      │
        │     1 │ a      │
        │     2 │ a      │
        └───────┴────────┘
        >>> t.limit(2)
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ c      │
        │     1 │ a      │
        └───────┴────────┘

        See Also
        --------
        [`Table.order_by`][ibis.expr.types.relations.Table.order_by]
        """
        return ops.Limit(self, n, offset=offset).to_expr()

    def head(self, n: int = 5) -> Table:
        """Select the first `n` rows of a table.

        !!! note "The result set is not deterministic without a call to [`order_by`][ibis.expr.types.relations.Table.order_by]."

        Parameters
        ----------
        n
            Number of rows to include

        Returns
        -------
        Table
            `self` limited to `n` rows

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [1, 1, 2], "b": ["c", "a", "a"]})
        >>> t
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ c      │
        │     1 │ a      │
        │     2 │ a      │
        └───────┴────────┘
        >>> t.head(2)
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ c      │
        │     1 │ a      │
        └───────┴────────┘

        See Also
        --------
        [`Table.limit`][ibis.expr.types.relations.Table.limit]
        [`Table.order_by`][ibis.expr.types.relations.Table.order_by]
        """
        return self.limit(n=n)

    def order_by(
        self,
        by: str
        | ir.Column
        | tuple[str | ir.Column, bool]
        | Sequence[str]
        | Sequence[ir.Column]
        | Sequence[tuple[str | ir.Column, bool]]
        | None,
    ) -> Table:
        """Sort a table by one or more expressions.

        Parameters
        ----------
        by
            Expressions to sort the table by.

        Returns
        -------
        Table
            Sorted table

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [1, 2, 3], "b": ["c", "b", "a"], "c": [4, 6, 5]})
        >>> t
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     1 │ c      │     4 │
        │     2 │ b      │     6 │
        │     3 │ a      │     5 │
        └───────┴────────┴───────┘
        >>> t.order_by("b")
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     3 │ a      │     5 │
        │     2 │ b      │     6 │
        │     1 │ c      │     4 │
        └───────┴────────┴───────┘
        >>> t.order_by(ibis.desc("c"))
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     2 │ b      │     6 │
        │     3 │ a      │     5 │
        │     1 │ c      │     4 │
        └───────┴────────┴───────┘
        """
        if isinstance(by, tuple):
            by = [by]
        elif by is None:
            by = []
        else:
            by = util.promote_list(by)
        return self.op().order_by(by).to_expr()

    def union(self, *tables: Table, distinct: bool = False) -> Table:
        """Compute the set union of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        *tables
            One or more table expressions
        distinct
            Only return distinct rows

        Returns
        -------
        Table
            A new table containing the union of all input tables.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t1 = ibis.memtable({"a": [1, 2]})
        >>> t1
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        └───────┘
        >>> t2 = ibis.memtable({"a": [2, 3]})
        >>> t2
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        │     3 │
        └───────┘
        >>> t1.union(t2)  # union all by default
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        │     2 │
        │     3 │
        └───────┘
        >>> t1.union(t2, distinct=True).order_by("a")
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        │     3 │
        └───────┘
        """
        left = self
        if not tables:
            raise com.IbisTypeError(
                "union requires a table or tables to compare against"
            )
        for right in tables:
            left = ops.Union(left, right, distinct=distinct)
        return left.to_expr()

    def intersect(self, *tables: Table, distinct: bool = True) -> Table:
        """Compute the set intersection of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        *tables
            One or more table expressions
        distinct
            Only return distinct rows

        Returns
        -------
        Table
            A new table containing the intersection of all input tables.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t1 = ibis.memtable({"a": [1, 2]})
        >>> t1
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        └───────┘
        >>> t2 = ibis.memtable({"a": [2, 3]})
        >>> t2
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        │     3 │
        └───────┘
        >>> t1.intersect(t2)
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        └───────┘
        """
        left = self
        if not tables:
            raise com.IbisTypeError(
                "intersect requires a table or tables to compare against"
            )
        for right in tables:
            left = ops.Intersection(left, right, distinct=distinct)
        return left.to_expr()

    def to_array(self) -> ir.Column:
        """View a single column table as an array.

        Returns
        -------
        Value
            A single column view of a table
        """
        schema = self.schema()
        if len(schema) != 1:
            raise com.ExpressionError(
                'Table must have exactly one column when viewed as array'
            )

        return ops.TableArrayView(self).to_expr()

    def mutate(
        self, exprs: Sequence[ir.Expr] | None = None, **mutations: ir.Value
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
        >>> import ibis
        >>> import ibis.expr.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch().select("species", "year", "bill_length_mm")
        >>> t
        ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ species ┃ year  ┃ bill_length_mm ┃
        ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string  │ int64 │ float64        │
        ├─────────┼───────┼────────────────┤
        │ Adelie  │  2007 │           39.1 │
        │ Adelie  │  2007 │           39.5 │
        │ Adelie  │  2007 │           40.3 │
        │ Adelie  │  2007 │            nan │
        │ Adelie  │  2007 │           36.7 │
        │ Adelie  │  2007 │           39.3 │
        │ Adelie  │  2007 │           38.9 │
        │ Adelie  │  2007 │           39.2 │
        │ Adelie  │  2007 │           34.1 │
        │ Adelie  │  2007 │           42.0 │
        │ …       │     … │              … │
        └─────────┴───────┴────────────────┘

        Add a new column from a per-element expression

        >>> t.mutate(next_year=_.year + 1).head()
        ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
        ┃ species ┃ year  ┃ bill_length_mm ┃ next_year ┃
        ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ string  │ int64 │ float64        │ int64     │
        ├─────────┼───────┼────────────────┼───────────┤
        │ Adelie  │  2007 │           39.1 │      2008 │
        │ Adelie  │  2007 │           39.5 │      2008 │
        │ Adelie  │  2007 │           40.3 │      2008 │
        │ Adelie  │  2007 │            nan │      2008 │
        │ Adelie  │  2007 │           36.7 │      2008 │
        └─────────┴───────┴────────────────┴───────────┘

        Add a new column based on an aggregation. Note the automatic broadcasting.

        >>> t.select("species", bill_demean=_.bill_length_mm - _.bill_length_mm.mean()).head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ species ┃ bill_demean ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ string  │ float64     │
        ├─────────┼─────────────┤
        │ Adelie  │    -4.82193 │
        │ Adelie  │    -4.42193 │
        │ Adelie  │    -3.62193 │
        │ Adelie  │         nan │
        │ Adelie  │    -7.22193 │
        └─────────┴─────────────┘

        Mutate across multiple columns

        >>> t.mutate(s.across(s.numeric() & ~s.c("year"), _ - _.mean())).head()
        ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ species ┃ year  ┃ bill_length_mm ┃
        ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string  │ int64 │ float64        │
        ├─────────┼───────┼────────────────┤
        │ Adelie  │  2007 │       -4.82193 │
        │ Adelie  │  2007 │       -4.42193 │
        │ Adelie  │  2007 │       -3.62193 │
        │ Adelie  │  2007 │            nan │
        │ Adelie  │  2007 │       -7.22193 │
        └─────────┴───────┴────────────────┘
        """
        import ibis.expr.analysis as an

        exprs = [] if exprs is None else util.promote_list(exprs)
        exprs = itertools.chain(
            itertools.chain.from_iterable(
                util.promote_list(_ensure_expr(self, expr)) for expr in exprs
            ),
            (
                e.name(name)
                for name, expr in mutations.items()
                for e in util.promote_list(_ensure_expr(self, expr))
            ),
        )
        mutation_exprs = an.get_mutation_exprs(list(exprs), self)
        return self.select(mutation_exprs)

    def select(
        self,
        *exprs: ir.Value | str | Iterable[ir.Value | str],
        **named_exprs: ir.Value | str,
    ) -> Table:
        """Compute a new table expression using `exprs` and `named_exprs`.

        Passing an aggregate function to this method will broadcast the
        aggregate's value over the number of rows in the table and
        automatically constructs a window function expression. See the examples
        section for more details.

        For backwards compatibility the keyword argument `exprs` is reserved
        and cannot be used to name an expression. This behavior will be removed
        in v4.

        Parameters
        ----------
        exprs
            Column expression, string, or list of column expressions and
            strings.
        named_exprs
            Column expressions

        Returns
        -------
        Table
            Table expression

        Examples
        --------
        >>> import ibis
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
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Simple projection

        >>> t.select("island", "bill_length_mm").head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │            nan │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘

        Projection by zero-indexed column position

        >>> t.select(0, 4).head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ species ┃ flipper_length_mm ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ string  │ int64             │
        ├─────────┼───────────────────┤
        │ Adelie  │               181 │
        │ Adelie  │               186 │
        │ Adelie  │               195 │
        │ Adelie  │                 ∅ │
        │ Adelie  │               193 │
        └─────────┴───────────────────┘

        Projection with renaming and compute in one call

        >>> t.select(next_year=t.year + 1).head()
        ┏━━━━━━━━━━━┓
        ┃ next_year ┃
        ┡━━━━━━━━━━━┩
        │ int64     │
        ├───────────┤
        │      2008 │
        │      2008 │
        │      2008 │
        │      2008 │
        │      2008 │
        └───────────┘

        Projection with aggregation expressions

        >>> t.select("island", bill_mean=t.bill_length_mm.mean()).head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━┓
        ┃ island    ┃ bill_mean ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ string    │ float64   │
        ├───────────┼───────────┤
        │ Torgersen │  43.92193 │
        │ Torgersen │  43.92193 │
        │ Torgersen │  43.92193 │
        │ Torgersen │  43.92193 │
        │ Torgersen │  43.92193 │
        └───────────┴───────────┘

        Projection with a selector

        >>> import ibis.expr.selectors as s
        >>> t.select(s.numeric() & ~s.c("year")).head()
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ float64        │ float64       │ int64             │ int64       │
        ├────────────────┼───────────────┼───────────────────┼─────────────┤
        │           39.1 │          18.7 │               181 │        3750 │
        │           39.5 │          17.4 │               186 │        3800 │
        │           40.3 │          18.0 │               195 │        3250 │
        │            nan │           nan │                 ∅ │           ∅ │
        │           36.7 │          19.3 │               193 │        3450 │
        └────────────────┴───────────────┴───────────────────┴─────────────┘

        Projection + aggregation across multiple columns

        >>> from ibis import _
        >>> t.select(s.across(s.numeric() & ~s.c("year"), _.mean())).head()
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ float64        │ float64       │ float64           │ float64     │
        ├────────────────┼───────────────┼───────────────────┼─────────────┤
        │       43.92193 │      17.15117 │        200.915205 │ 4201.754386 │
        │       43.92193 │      17.15117 │        200.915205 │ 4201.754386 │
        │       43.92193 │      17.15117 │        200.915205 │ 4201.754386 │
        │       43.92193 │      17.15117 │        200.915205 │ 4201.754386 │
        │       43.92193 │      17.15117 │        200.915205 │ 4201.754386 │
        └────────────────┴───────────────┴───────────────────┴─────────────┘
        """
        import ibis.expr.analysis as an
        from ibis.expr.selectors import Selector

        exprs = list(
            itertools.chain(
                itertools.chain.from_iterable(
                    util.promote_list(e.expand(self) if isinstance(e, Selector) else e)
                    for e in exprs
                ),
                (
                    self._ensure_expr(expr).name(name)
                    for name, expr in named_exprs.items()
                ),
            )
        )

        if not exprs:
            raise com.IbisTypeError(
                "You must select at least one column for a valid projection"
            )

        op = an.Projector(self, exprs).get_result()

        return op.to_expr()

    projection = select

    def relabel(
        self,
        substitutions: Mapping[str, str]
        | Callable[[str], str | None]
        | Literal["snake_case"],
    ) -> Table:
        """Rename columns in the table.

        Parameters
        ----------
        substitutions
            A mapping or function from old to new column names. If a column
            isn't in the mapping (or if the callable returns None) it is left
            with its original name. May also pass the string ``"snake_case"``,
            which will relabel all columns to use a ``snake_case`` naming
            convention.

        Returns
        -------
        Table
            A relabeled table expressi

        Examples
        --------
        >>> import ibis
        >>> import ibis.expr.selectors as s
        >>> ibis.options.interactive = True
        >>> first3 = s.r[:3]  # first 3 columns
        >>> t = ibis.examples.penguins_raw_raw.fetch().select(first3)
        >>> t
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ studyName ┃ Sample Number ┃ Species                             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string    │ int64         │ string                              │
        ├───────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708   │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             2 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             3 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             4 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             5 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             6 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             7 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             8 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │             9 │ Adelie Penguin (Pygoscelis adeliae) │
        │ PAL0708   │            10 │ Adelie Penguin (Pygoscelis adeliae) │
        │ …         │             … │ …                                   │
        └───────────┴───────────────┴─────────────────────────────────────┘

        Relabel column names using a mapping from old name to new name

        >>> t.relabel({"studyName": "study_name"}).head(1)
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ study_name ┃ Sample Number ┃ Species                             ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string     │ int64         │ string                              │
        ├────────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708    │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └────────────┴───────────────┴─────────────────────────────────────┘

        Relabel column names using a snake_case convention

        >>> t.relabel("snake_case").head(1)
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ study_name ┃ sample_number ┃ species                             ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string     │ int64         │ string                              │
        ├────────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708    │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └────────────┴───────────────┴─────────────────────────────────────┘

        Relabel column names using a callable

        >>> t.relabel(str.upper).head(1)
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ STUDYNAME ┃ SAMPLE NUMBER ┃ SPECIES                             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string    │ int64         │ string                              │
        ├───────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708   │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └───────────┴───────────────┴─────────────────────────────────────┘
        """
        observed = set()

        if isinstance(substitutions, Mapping):
            rename = substitutions.get
        elif substitutions == "snake_case":

            def rename(c):
                c = c.strip()
                if " " in c:
                    # Handle "space case possibly with-hyphens"
                    return "_".join(c.lower().split()).replace("-", "_")
                # Handle PascalCase, camelCase, and kebab-case
                c = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', c)
                c = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', c)
                c = c.replace("-", "_")
                return c.lower()

        else:
            rename = substitutions

        exprs = []
        for c in self.columns:
            expr = self[c]
            if (name := rename(c)) is not None:
                expr = expr.name(name)
                observed.add(c)
            exprs.append(expr)

        if isinstance(substitutions, Mapping):
            for c in substitutions:
                if c not in observed:
                    raise KeyError(f"{c!r} is not an existing column")

        return self.select(exprs)

    def drop(self, *fields: str | Selector) -> Table:
        """Remove fields from a table.

        Parameters
        ----------
        fields
            Fields to drop. Strings and selectors are accepted.

        Returns
        -------
        Table
            A table with all columns matching `fields` removed.

        Examples
        --------
        >>> import ibis
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
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Drop one or more columns

        >>> t.drop("species").head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string    │ float64        │ float64       │ int64             │ … │
        ├───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t.drop("species", "bill_length_mm").head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━┓
        ┃ island    ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ … ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━┩
        │ string    │ float64       │ int64             │ int64       │ string │ … │
        ├───────────┼───────────────┼───────────────────┼─────────────┼────────┼───┤
        │ Torgersen │          18.7 │               181 │        3750 │ male   │ … │
        │ Torgersen │          17.4 │               186 │        3800 │ female │ … │
        │ Torgersen │          18.0 │               195 │        3250 │ female │ … │
        │ Torgersen │           nan │                 ∅ │           ∅ │ ∅      │ … │
        │ Torgersen │          19.3 │               193 │        3450 │ female │ … │
        └───────────┴───────────────┴───────────────────┴─────────────┴────────┴───┘

        Drop with selectors, mix and match

        >>> import ibis.expr.selectors as s
        >>> t.drop("species", s.startswith("bill_")).head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ island    ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ string    │ int64             │ int64       │ string │ int64 │
        ├───────────┼───────────────────┼─────────────┼────────┼───────┤
        │ Torgersen │               181 │        3750 │ male   │  2007 │
        │ Torgersen │               186 │        3800 │ female │  2007 │
        │ Torgersen │               195 │        3250 │ female │  2007 │
        │ Torgersen │                 ∅ │           ∅ │ ∅      │  2007 │
        │ Torgersen │               193 │        3450 │ female │  2007 │
        └───────────┴───────────────────┴─────────────┴────────┴───────┘
        """
        from ibis import selectors as s

        if not fields:
            # no-op if nothing to be dropped
            return self

        if missing_fields := {f for f in fields if isinstance(f, str)}.difference(
            self.schema().names
        ):
            raise KeyError(f"Fields not in table: {sorted(missing_fields)}")

        sels = (s.c(f) if isinstance(f, str) else f for f in fields)
        return self.select(~s.any_of(*sels))

    def filter(
        self,
        predicates: ir.BooleanValue | Sequence[ir.BooleanValue] | IfAnyAll,
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

        Examples
        --------
        >>> import ibis
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
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t.filter([t.species == "Adelie", t.body_mass_g > 3500]).sex.value_counts().dropna("sex")
        ┏━━━━━━━━┳━━━━━━━━━━━┓
        ┃ sex    ┃ sex_count ┃
        ┡━━━━━━━━╇━━━━━━━━━━━┩
        │ string │ int64     │
        ├────────┼───────────┤
        │ male   │        68 │
        │ female │        22 │
        └────────┴───────────┘
        """
        import ibis.expr.analysis as an

        resolved_predicates = _resolve_predicates(self, predicates)
        predicates = [
            an._rewrite_filter(pred.op() if isinstance(pred, Expr) else pred)
            for pred in resolved_predicates
        ]
        return an.apply_filter(self.op(), predicates).to_expr()

    def count(self, where: ir.BooleanValue | None = None) -> ir.IntegerScalar:
        """Compute the number of rows in the table.

        Parameters
        ----------
        where
            Optional boolean expression to filter rows when counting.

        Returns
        -------
        IntegerScalar
            Number of rows in the table

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": ["foo", "bar", "baz"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ a      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ foo    │
        │ bar    │
        │ baz    │
        └────────┘
        >>> t.count()
        3
        >>> t.count(t.a != "foo")
        2
        >>> type(t.count())
        <class 'ibis.expr.types.numeric.IntegerScalar'>
        """
        return ops.CountStar(self, where).to_expr().name("count")

    def dropna(
        self, subset: Sequence[str] | None = None, how: Literal["any", "all"] = "any"
    ) -> Table:
        """Remove rows with null values from the table.

        Parameters
        ----------
        subset
            Columns names to consider when dropping nulls. By default all columns
            are considered.
        how
            Determine whether a row is removed if there is **at least one null
            value in the row** (`'any'`), or if **all** row values are null
            (`'all'`).

        Returns
        -------
        Table
            Table expression

        Examples
        --------
        >>> import ibis
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
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t.count()
        344
        >>> t.dropna(["bill_length_mm", "body_mass_g"]).count()
        342
        >>> t.dropna(how="all").count()  # no rows where all columns are null
        344
        """
        if subset is not None:
            subset = util.promote_list(subset)
        return ops.DropNa(self, how, subset).to_expr()

    def fillna(
        self,
        replacements: ir.Scalar | Mapping[str, ir.Scalar],
    ) -> Table:
        """Fill null values in a table expression.

        !!! note "There is potential lack of type stability with the `fillna` API"

            For example, different library versions may impact whether a given
            backend promotes integer replacement values to floats.

        Parameters
        ----------
        replacements
            Value with which to fill nulls. If `replacements` is a mapping, the
            keys are column names that map to their replacement value. If
            passed as a scalar all columns are filled with that value.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.sex
        ┏━━━━━━━━┓
        ┃ sex    ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ male   │
        │ female │
        │ female │
        │ ∅      │
        │ female │
        │ male   │
        │ female │
        │ male   │
        │ ∅      │
        │ ∅      │
        │ …      │
        └────────┘
        >>> t.fillna({"sex": "unrecorded"}).sex
        ┏━━━━━━━━━━━━┓
        ┃ sex        ┃
        ┡━━━━━━━━━━━━┩
        │ string     │
        ├────────────┤
        │ male       │
        │ female     │
        │ female     │
        │ unrecorded │
        │ female     │
        │ male       │
        │ female     │
        │ male       │
        │ unrecorded │
        │ unrecorded │
        │ …          │
        └────────────┘

        Returns
        -------
        Table
            Table expression
        """
        schema = self.schema()

        if isinstance(replacements, collections.abc.Mapping):
            for col, val in replacements.items():
                if col not in schema:
                    columns_formatted = ', '.join(map(repr, schema.names))
                    raise com.IbisTypeError(
                        f"Column {col!r} is not found in table. "
                        f"Existing columns: {columns_formatted}."
                    ) from None

                col_type = schema[col]
                val_type = val.type() if isinstance(val, Expr) else dt.infer(val)
                if not dt.castable(val_type, col_type):
                    raise com.IbisTypeError(
                        f"Cannot fillna on column {col!r} of type {col_type} with a "
                        f"value of type {val_type}"
                    )
        else:
            val_type = (
                replacements.type()
                if isinstance(replacements, Expr)
                else dt.infer(replacements)
            )
            for col, col_type in schema.items():
                if col_type.nullable and not dt.castable(val_type, col_type):
                    raise com.IbisTypeError(
                        f"Cannot fillna on column {col!r} of type {col_type} with a "
                        f"value of type {val_type} - pass in an explicit mapping "
                        f"of fill values to `fillna` instead."
                    )
        return ops.FillNa(self, replacements).to_expr()

    def unpack(self, *columns: str) -> Table:
        """Project the struct fields of each of `columns` into `self`.

        Existing fields are retained in the projection.

        Parameters
        ----------
        columns
            String column names to project into `self`.

        Returns
        -------
        Table
            The child table with struct fields of each of `columns` projected.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> lines = '''
        ...     {"name": "a", "pos": {"lat": 10.1, "lon": 30.3}}
        ...     {"name": "b", "pos": {"lat": 10.2, "lon": 30.2}}
        ...     {"name": "c", "pos": {"lat": 10.3, "lon": 30.1}}
        ... '''
        >>> with open("/tmp/lines.json", "w") as f:
        ...     _ = f.write(lines)
        >>> t = ibis.read_json("/tmp/lines.json")
        >>> t
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ name   ┃ pos                                ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string │ struct<lat: float64, lon: float64> │
        ├────────┼────────────────────────────────────┤
        │ a      │ {'lat': 10.1, 'lon': 30.3}         │
        │ b      │ {'lat': 10.2, 'lon': 30.2}         │
        │ c      │ {'lat': 10.3, 'lon': 30.1}         │
        └────────┴────────────────────────────────────┘
        >>> t.unpack("pos")
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
        ┃ name   ┃ lat     ┃ lon     ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
        │ string │ float64 │ float64 │
        ├────────┼─────────┼─────────┤
        │ a      │    10.1 │    30.3 │
        │ b      │    10.2 │    30.2 │
        │ c      │    10.3 │    30.1 │
        └────────┴─────────┴─────────┘

        See Also
        --------
        [`StructValue.lift`][ibis.expr.types.structs.StructValue.lift]
        """
        columns_to_unpack = frozenset(columns)
        result_columns = []
        for column in self.columns:
            if column in columns_to_unpack:
                expr = self[column]
                result_columns.extend(expr[field] for field in expr.names)
            else:
                result_columns.append(column)
        return self[result_columns]

    def info(self) -> Table:
        """Return summary information about a table.

        Returns
        -------
        Table
            Summary of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> t.info()
        ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━┓
        ┃ name              ┃ type    ┃ nullable ┃ nulls ┃ non_nulls ┃ null_frac ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━┩
        │ string            │ string  │ boolean  │ int64 │ int64     │ float64   │ … │
        ├───────────────────┼─────────┼──────────┼───────┼───────────┼───────────┼───┤
        │ species           │ string  │ True     │     0 │       344 │  0.000000 │ … │
        │ island            │ string  │ True     │     0 │       344 │  0.000000 │ … │
        │ bill_length_mm    │ float64 │ True     │     2 │       342 │  0.005814 │ … │
        │ bill_depth_mm     │ float64 │ True     │     2 │       342 │  0.005814 │ … │
        │ flipper_length_mm │ int64   │ True     │     2 │       342 │  0.005814 │ … │
        │ body_mass_g       │ int64   │ True     │     2 │       342 │  0.005814 │ … │
        │ sex               │ string  │ True     │    11 │       333 │  0.031977 │ … │
        │ year              │ int64   │ True     │     0 │       344 │  0.000000 │ … │
        └───────────────────┴─────────┴──────────┴───────┴───────────┴───────────┴───┘
        """
        from ibis import literal as lit

        aggs = []

        for pos, colname in enumerate(self.columns):
            col = self[colname]
            typ = col.type()
            agg = self.select(
                isna=ibis.case().when(col.isnull(), 1).else_(0).end()
            ).agg(
                name=lit(colname),
                type=lit(str(typ)),
                nullable=lit(int(typ.nullable)).cast("bool"),
                nulls=lambda t: t.isna.sum(),
                non_nulls=lambda t: (1 - t.isna).sum(),
                null_frac=lambda t: t.isna.mean(),
                pos=lit(pos),
            )
            aggs.append(agg)
        return ibis.union(*aggs).order_by(ibis.asc("pos"))

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

        if expr.get_name() != name:
            expr = expr.name(name)

        if name not in self:
            raise KeyError(f'{name} is not in the table')

        proj_exprs = []
        for key in self.columns:
            if key == name:
                proj_exprs.append(expr)
            else:
                proj_exprs.append(self[key])

        return self.select(proj_exprs)

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
        expr = klass(left, right, predicates).to_expr()

        # semi/anti join only give access to the left table's fields, so
        # there's never overlap
        if how in ("semi", "anti"):
            return expr

        return ops.relations._dedup_join_columns(expr, suffixes=suffixes)

    def asof_join(
        left: Table,
        right: Table,
        predicates: str | ir.BooleanColumn | Sequence[str | ir.BooleanColumn] = (),
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
        op = ops.AsOfJoin(
            left=left,
            right=right,
            predicates=predicates,
            by=by,
            tolerance=tolerance,
        )
        return ops.relations._dedup_join_columns(op.to_expr(), suffixes=suffixes)

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
        >>> import ibis.expr.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> agg = t.drop("year").agg(s.across(s.numeric(), _.mean()))
        >>> expr = t.cross_join(agg)
        >>> expr
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm_x ┃ bill_depth_mm_x ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64          │ float64         │ … │
        ├─────────┼───────────┼──────────────────┼─────────────────┼───┤
        │ Adelie  │ Torgersen │             39.1 │            18.7 │ … │
        │ Adelie  │ Torgersen │             39.5 │            17.4 │ … │
        │ Adelie  │ Torgersen │             40.3 │            18.0 │ … │
        │ Adelie  │ Torgersen │              nan │             nan │ … │
        │ Adelie  │ Torgersen │             36.7 │            19.3 │ … │
        │ Adelie  │ Torgersen │             39.3 │            20.6 │ … │
        │ Adelie  │ Torgersen │             38.9 │            17.8 │ … │
        │ Adelie  │ Torgersen │             39.2 │            19.6 │ … │
        │ Adelie  │ Torgersen │             34.1 │            18.1 │ … │
        │ Adelie  │ Torgersen │             42.0 │            20.2 │ … │
        │ …       │ …         │                … │               … │ … │
        └─────────┴───────────┴──────────────────┴─────────────────┴───┘
        >>> from pprint import pprint
        >>> pprint(expr.columns)
        ['species',
         'island',
         'bill_length_mm_x',
         'bill_depth_mm_x',
         'flipper_length_mm_x',
         'body_mass_g_x',
         'sex',
         'year',
         'bill_length_mm_y',
         'bill_depth_mm_y',
         'flipper_length_mm_y',
         'body_mass_g_y']
        >>> expr.count()
        344
        >>> t.count()
        344
        """
        op = ops.CrossJoin(
            left,
            functools.reduce(Table.cross_join, rest, right),
            [],
        )
        return ops.relations._dedup_join_columns(op.to_expr(), suffixes=suffixes)

    inner_join = _regular_join_method("inner_join", "inner")
    left_join = _regular_join_method("left_join", "left")
    outer_join = _regular_join_method("outer_join", "outer")
    right_join = _regular_join_method("right_join", "right")
    semi_join = _regular_join_method("semi_join", "semi")
    anti_join = _regular_join_method("anti_join", "anti")
    any_inner_join = _regular_join_method("any_inner_join", "any_inner")
    any_left_join = _regular_join_method("any_left_join", "any_left")

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
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> expr = t.alias("pingüinos").sql('SELECT * FROM "pingüinos" LIMIT 5')
        >>> expr
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        """
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
            will run against.

            This restriction may be lifted in a future version of ibis.

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
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> expr = t.sql("SELECT island, mean(bill_length_mm) FROM penguins GROUP BY 1 ORDER BY 2 DESC")
        >>> expr
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ mean(bill_length_mm) ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ string    │ float64              │
        ├───────────┼──────────────────────┤
        │ Biscoe    │            45.257485 │
        │ Dream     │            44.167742 │
        │ Torgersen │            38.950980 │
        └───────────┴──────────────────────┘
        """
        op = ops.SQLStringView(
            child=self,
            name=next(_ALIASES),
            query=query,
        )
        return op.to_expr()

    def to_pandas(self, **kwargs) -> pd.DataFrame:
        """Convert a table expression to a pandas DataFrame.

        Parameters
        ----------
        kwargs
            Same as keyword arguments to [`execute`][ibis.expr.types.core.Expr.execute]
        """
        return self.execute(**kwargs)

    def cache(self) -> Table:
        """Cache the provided expression.

         All subsequent operations on the returned expression will be performed on the cached data.
         You may use the ``with`` statement to explicitly clean up the cached expression when it's not needed.

        !!! note "This method eagerly evaluates the expression prior to caching"

        Returns
        -------
        Table
            Cached table

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> cached_penguins = t.mutate(computation="Heavy Computation").cache()
        >>> cached_penguins
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Explicit cache cleanup

        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> with t.mutate(computation="Heavy Computation").cache() as cached_penguins:
        ...     cached_penguins
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │            nan │           nan │                 ∅ │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        """
        current_backend = self._find_backend(use_default=True)
        return current_backend._cache(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_backend = self._find_backend(use_default=True)
        return current_backend._release_cache(self)

    def __enter__(self):
        return self

    def pivot_longer(
        self,
        cols: str | s.Selector,
        *,
        names_to: str | Iterable[str] = "name",
        names_pattern: str | re.Pattern = r"(.+)",
        names_transform: Callable[[str], ir.Value]
        | Mapping[str, Callable[[str], ir.Value]]
        | None = None,
        values_to: str = "value",
        values_transform: Callable[[ir.Value], ir.Value] | Deferred | None = None,
    ) -> Table:
        """Transform a table from wider to longer.

        Parameters
        ----------
        cols
            String column names or selectors.
        names_to
            A string or iterable of strings indicating how to name the new
            pivoted columns.
        names_pattern
            Pattern to use to extract column names from the input. By default
            the entire column name is extracted.
        names_transform
            Function or mapping of a name in `names_to` to a function to
            transform a column name to a value.
        values_to
            Name of the pivoted value column.
        values_transform
            Apply a function to the value column. This can be a lambda or
            deferred expression.

        Returns
        -------
        Table
            Pivoted table

        Examples
        --------
        Basic usage

        >>> import ibis
        >>> import ibis.expr.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> relig_income = ibis.examples.relig_income_raw.fetch()
        >>> relig_income
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━┓
        ┃ religion                ┃ <$10k ┃ $10-20k ┃ $20-30k ┃ $30-40k ┃ $40-50k ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━┩
        │ string                  │ int64 │ int64   │ int64   │ int64   │ int64   │ … │
        ├─────────────────────────┼───────┼─────────┼─────────┼─────────┼─────────┼───┤
        │ Agnostic                │    27 │      34 │      60 │      81 │      76 │ … │
        │ Atheist                 │    12 │      27 │      37 │      52 │      35 │ … │
        │ Buddhist                │    27 │      21 │      30 │      34 │      33 │ … │
        │ Catholic                │   418 │     617 │     732 │     670 │     638 │ … │
        │ Don’t know/refused      │    15 │      14 │      15 │      11 │      10 │ … │
        │ Evangelical Prot        │   575 │     869 │    1064 │     982 │     881 │ … │
        │ Hindu                   │     1 │       9 │       7 │       9 │      11 │ … │
        │ Historically Black Prot │   228 │     244 │     236 │     238 │     197 │ … │
        │ Jehovah's Witness       │    20 │      27 │      24 │      24 │      21 │ … │
        │ Jewish                  │    19 │      19 │      25 │      25 │      30 │ … │
        │ …                       │     … │       … │       … │       … │       … │ … │
        └─────────────────────────┴───────┴─────────┴─────────┴─────────┴─────────┴───┘

        Here we convert column names not matching the selector for the `religion` column
        and convert those names into values

        >>> relig_income.pivot_longer(~s.c("religion"), names_to="income", values_to="count")
        ┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ religion ┃ income             ┃ count ┃
        ┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ string   │ string             │ int64 │
        ├──────────┼────────────────────┼───────┤
        │ Agnostic │ <$10k              │    27 │
        │ Agnostic │ $10-20k            │    34 │
        │ Agnostic │ $20-30k            │    60 │
        │ Agnostic │ $30-40k            │    81 │
        │ Agnostic │ $40-50k            │    76 │
        │ Agnostic │ $50-75k            │   137 │
        │ Agnostic │ $75-100k           │   122 │
        │ Agnostic │ $100-150k          │   109 │
        │ Agnostic │ >150k              │    84 │
        │ Agnostic │ Don't know/refused │    96 │
        │ …        │ …                  │     … │
        └──────────┴────────────────────┴───────┘

        Simliarly for a different example dataset, we convert names to values
        but using a different selector and the default `values_to` value.

        >>> world_bank_pop = ibis.examples.world_bank_pop_raw.fetch(header=1)
        >>> world_bank_pop.head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━┓
        ┃ country ┃ indicator   ┃ 2000         ┃ 2001         ┃ 2002         ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string      │ float64      │ float64      │ float64      │ … │
        ├─────────┼─────────────┼──────────────┼──────────────┼──────────────┼───┤
        │ ABW     │ SP.URB.TOTL │ 4.244400e+04 │ 4.304800e+04 │ 4.367000e+04 │ … │
        │ ABW     │ SP.URB.GROW │ 1.182632e+00 │ 1.413021e+00 │ 1.434560e+00 │ … │
        │ ABW     │ SP.POP.TOTL │ 9.085300e+04 │ 9.289800e+04 │ 9.499200e+04 │ … │
        │ ABW     │ SP.POP.GROW │ 2.055027e+00 │ 2.225930e+00 │ 2.229056e+00 │ … │
        │ AFG     │ SP.URB.TOTL │ 4.436299e+06 │ 4.648055e+06 │ 4.892951e+06 │ … │
        └─────────┴─────────────┴──────────────┴──────────────┴──────────────┴───┘
        >>> world_bank_pop.pivot_longer(s.matches(r"\\d{4}"), names_to="year").head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
        ┃ country ┃ indicator   ┃ year   ┃ value   ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
        │ string  │ string      │ string │ float64 │
        ├─────────┼─────────────┼────────┼─────────┤
        │ ABW     │ SP.URB.TOTL │ 2000   │ 42444.0 │
        │ ABW     │ SP.URB.TOTL │ 2001   │ 43048.0 │
        │ ABW     │ SP.URB.TOTL │ 2002   │ 43670.0 │
        │ ABW     │ SP.URB.TOTL │ 2003   │ 44246.0 │
        │ ABW     │ SP.URB.TOTL │ 2004   │ 44669.0 │
        └─────────┴─────────────┴────────┴─────────┘

        `pivot_longer` has some preprocessing capabiltiies like stripping a prefix and applying
        a function to column names

        >>> billboard = ibis.examples.billboard.fetch()
        >>> billboard
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━┓
        ┃ artist         ┃ track                   ┃ date_entered ┃ wk1   ┃ wk2   ┃ … ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━┩
        │ string         │ string                  │ date         │ int64 │ int64 │ … │
        ├────────────────┼─────────────────────────┼──────────────┼───────┼───────┼───┤
        │ 2 Pac          │ Baby Don't Cry (Keep... │ 2000-02-26   │    87 │    82 │ … │
        │ 2Ge+her        │ The Hardest Part Of ... │ 2000-09-02   │    91 │    87 │ … │
        │ 3 Doors Down   │ Kryptonite              │ 2000-04-08   │    81 │    70 │ … │
        │ 3 Doors Down   │ Loser                   │ 2000-10-21   │    76 │    76 │ … │
        │ 504 Boyz       │ Wobble Wobble           │ 2000-04-15   │    57 │    34 │ … │
        │ 98^0           │ Give Me Just One Nig... │ 2000-08-19   │    51 │    39 │ … │
        │ A*Teens        │ Dancing Queen           │ 2000-07-08   │    97 │    97 │ … │
        │ Aaliyah        │ I Don't Wanna           │ 2000-01-29   │    84 │    62 │ … │
        │ Aaliyah        │ Try Again               │ 2000-03-18   │    59 │    53 │ … │
        │ Adams, Yolanda │ Open My Heart           │ 2000-08-26   │    76 │    76 │ … │
        │ …              │ …                       │ …            │     … │     … │ … │
        └────────────────┴─────────────────────────┴──────────────┴───────┴───────┴───┘
        >>> billboard.pivot_longer(
        ...     s.startswith("wk"),
        ...     names_to="week",
        ...     names_pattern=r"wk(.+)",
        ...     names_transform=int,
        ...     values_to="rank",
        ...     values_transform=_.cast("int"),
        ... ).dropna("rank")
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┓
        ┃ artist  ┃ track                   ┃ date_entered ┃ week ┃ rank  ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━┩
        │ string  │ string                  │ date         │ int8 │ int64 │
        ├─────────┼─────────────────────────┼──────────────┼──────┼───────┤
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    1 │    87 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    2 │    82 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    3 │    72 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    4 │    77 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    5 │    87 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    6 │    94 │
        │ 2 Pac   │ Baby Don't Cry (Keep... │ 2000-02-26   │    7 │    99 │
        │ 2Ge+her │ The Hardest Part Of ... │ 2000-09-02   │    1 │    91 │
        │ 2Ge+her │ The Hardest Part Of ... │ 2000-09-02   │    2 │    87 │
        │ 2Ge+her │ The Hardest Part Of ... │ 2000-09-02   │    3 │    92 │
        │ …       │ …                       │ …            │    … │     … │
        └─────────┴─────────────────────────┴──────────────┴──────┴───────┘

        You can use regular expression capture groups to extract multiple
        variables stored in column names

        >>> who = ibis.examples.who.fetch()
        >>> who
        ┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━┓
        ┃ country     ┃ iso2   ┃ iso3   ┃ year  ┃ new_sp_m014 ┃ new_sp_m1524 ┃ … ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━┩
        │ string      │ string │ string │ int64 │ int64       │ int64        │ … │
        ├─────────────┼────────┼────────┼───────┼─────────────┼──────────────┼───┤
        │ Afghanistan │ AF     │ AFG    │  1980 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1981 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1982 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1983 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1984 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1985 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1986 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1987 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1988 │           ∅ │            ∅ │ … │
        │ Afghanistan │ AF     │ AFG    │  1989 │           ∅ │            ∅ │ … │
        │ …           │ …      │ …      │     … │           … │            … │ … │
        └─────────────┴────────┴────────┴───────┴─────────────┴──────────────┴───┘
        >>> len(who.columns)
        60
        >>> who.pivot_longer(
        ...     s.r["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_(.)(.*)",
        ...     values_to="count",
        ... )
        ┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ country     ┃ iso2   ┃ iso3   ┃ year  ┃ diagnosis ┃ gender ┃ age    ┃ count ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ string      │ string │ string │ int64 │ string    │ string │ string │ int64 │
        ├─────────────┼────────┼────────┼───────┼───────────┼────────┼────────┼───────┤
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 014    │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 1524   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 2534   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 3544   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 4554   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 5564   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 65     │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 014    │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 1524   │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 2534   │     ∅ │
        │ …           │ …      │ …      │     … │ …         │ …      │ …      │     … │
        └─────────────┴────────┴────────┴───────┴───────────┴────────┴────────┴───────┘

        `names_transform` is flexible, and can be:

            1. A mapping of one or more names in `names_to` to callable
            2. A callable that will be applied to every name

        Let's recode gender and age to numeric values using a mapping

        >>> who.pivot_longer(
        ...     s.r["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_(.)(.*)",
        ...     names_transform=dict(
        ...         gender={"m": 1, "f": 2}.get,
        ...         age=dict(zip(["014", "1524", "2534", "3544", "4554", "5564", "65"], range(7))).get,
        ...     ),
        ...     values_to="count",
        ... )
        ┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━┓
        ┃ country     ┃ iso2   ┃ iso3   ┃ year  ┃ diagnosis ┃ gender ┃ age  ┃ count ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━┩
        │ string      │ string │ string │ int64 │ string    │ int8   │ int8 │ int64 │
        ├─────────────┼────────┼────────┼───────┼───────────┼────────┼──────┼───────┤
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    0 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    1 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    2 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    3 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    4 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    5 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    6 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    0 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    1 │     ∅ │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    2 │     ∅ │
        │ …           │ …      │ …      │     … │ …         │      … │    … │     … │
        └─────────────┴────────┴────────┴───────┴───────────┴────────┴──────┴───────┘

        The number of match groups in `names_pattern` must match the length of `names_to`

        >>> who.pivot_longer(
        ...     s.r["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_.(.*)",
        ... )
        Traceback (most recent call last):
          ...
        ibis.common.exceptions.IbisInputError: Number of match groups in `names_pattern` ...

        `names_transform` must be a mapping or callable

        >>> who.pivot_longer(s.r["new_sp_m014":"newrel_f65"], names_transform="upper")
        Traceback (most recent call last):
          ...
        ibis.common.exceptions.IbisTypeError: ... Got <class 'str'>
        """
        import ibis.expr.selectors as s

        pivot_sel = s.c(cols) if isinstance(cols, str) else cols

        pivot_cols = pivot_sel.expand(self)
        if not pivot_cols:
            # TODO: improve the repr of selectors
            raise com.IbisInputError("Selector returned no columns to pivot on")

        names_to = util.promote_list(names_to)

        names_pattern = re.compile(names_pattern)
        if (ngroups := names_pattern.groups) != (nnames := len(names_to)):
            raise com.IbisInputError(
                f"Number of match groups in `names_pattern`"
                f"{names_pattern.pattern!r} ({ngroups:d} groups) doesn't "
                f"match the length of `names_to` {names_to} (length {nnames:d})"
            )

        if names_transform is None:
            names_transform = dict.fromkeys(names_to, toolz.identity)
        elif not isinstance(names_transform, Mapping):
            if callable(names_transform):
                names_transform = dict.fromkeys(names_to, names_transform)
            else:
                raise com.IbisTypeError(
                    f"`names_transform` must be a mapping or callable. Got {type(names_transform)}"
                )

        for name in names_to:
            names_transform.setdefault(name, toolz.identity)

        if values_transform is None:
            values_transform = toolz.identity
        elif isinstance(values_transform, Deferred):
            values_transform = values_transform.resolve

        names_map = {name: [] for name in names_to}
        values = []

        for pivot_col in pivot_cols:
            col_name = pivot_col.get_name()
            match_result = names_pattern.match(col_name)
            for name, value in zip(names_to, match_result.groups()):
                transformer = names_transform[name]
                names_map[name].append(transformer(value))
            values.append(values_transform(pivot_col))

        new_cols = {key: ibis.array(value).unnest() for key, value in names_map.items()}
        new_cols[values_to] = ibis.array(values).unnest()

        return self.select(~pivot_sel, **new_cols)


def _resolve_predicates(
    table: Table, predicates
) -> tuple[list[ir.BooleanValue], list[tuple[ir.BooleanValue, ir.Table]]]:
    import ibis.expr.analysis as an
    import ibis.expr.types as ir

    predicates = [
        pred.op()
        for preds in map(
            functools.partial(ir.relations.bind_expr, table),
            util.promote_list(predicates),
        )
        for pred in util.promote_list(preds)
    ]
    predicates = an.flatten_predicate(predicates)

    resolved_predicates = []
    for pred in predicates:
        if isinstance(pred, ops.logical._UnresolvedSubquery):
            resolved_predicates.append(pred._resolve(table.op()))
        else:
            resolved_predicates.append(pred)

    return resolved_predicates


def bind_expr(table, expr):
    if util.is_iterable(expr):
        return [bind_expr(table, x) for x in expr]

    return table._ensure_expr(expr)


public(TableExpr=Table)
