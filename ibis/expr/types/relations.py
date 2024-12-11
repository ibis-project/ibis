from __future__ import annotations

import itertools
import operator
import re
import warnings
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from keyword import iskeyword
from typing import TYPE_CHECKING, Any, Literal, NoReturn, overload

import toolz
from public import public

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis import util
from ibis.common.deferred import Deferred, Resolver
from ibis.common.selectors import Expandable, Selector
from ibis.expr.rewrites import DerefMap
from ibis.expr.types.core import Expr, _FixedTextJupyterMixin
from ibis.expr.types.generic import Value, literal
from ibis.expr.types.temporal import TimestampColumn
from ibis.util import deprecated

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from rich.table import Table as RichTable

    import ibis.expr.types as ir
    import ibis.selectors as s
    from ibis.expr.operations.relations import JoinKind, Set
    from ibis.expr.schema import SchemaLike
    from ibis.expr.types import Table
    from ibis.expr.types.groupby import GroupedTable
    from ibis.expr.types.temporal_windows import WindowedTable
    from ibis.formats.pandas import PandasData
    from ibis.formats.pyarrow import PyArrowData
    from ibis.selectors import IfAnyAll


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
        self: ir.Table,
        right: ir.Table,
        predicates: (
            str
            | Sequence[str | tuple[str | ir.Column, str | ir.Column] | ir.BooleanValue]
        ) = (),
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ) -> ir.Table:
        """Perform a join between two tables.

        Parameters
        ----------
        right
            Right table to join
        predicates
            Boolean or column names to join on
        lname
            A format string to use to rename overlapping columns in the left
            table (e.g. `"left_{name}"`).
        rname
            A format string to use to rename overlapping columns in the right
            table (e.g. `"right_{name}"`).

        Returns
        -------
        Table
            Joined table
        """
        return self.join(right, predicates, how=how, lname=lname, rname=rname)

    f.__name__ = name
    return f


def bind(table: Table, value) -> Iterator[ir.Value]:
    """Bind a value to a table expression."""
    if isinstance(value, str):
        # TODO(kszucs): perhaps use getattr(table, value) instead for nicer error msg
        yield ops.Field(table, value).to_expr()
    elif isinstance(value, ops.Value):
        yield value.to_expr()
    elif isinstance(value, Value):
        yield value
    elif isinstance(value, Table):
        for name in value.columns:
            yield ops.Field(value, name).to_expr()
    elif isinstance(value, Deferred):
        yield value.resolve(table)
    elif isinstance(value, Resolver):
        yield value.resolve({"_": table})
    elif isinstance(value, Expandable):
        yield from value.expand(table)
    elif callable(value):
        # rebind, otherwise the callable is required to return an expression
        # which would preclude support for expressions like lambda _: 2
        yield from bind(table, value(table))
    else:
        yield literal(value)


def unwrap_alias(node: ops.Value) -> ops.Value:
    """Unwrap an alias node."""
    if isinstance(node, ops.Alias):
        return node.arg
    else:
        return node


def unwrap_aliases(values: Iterator[ir.Value]) -> Mapping[str, ir.Value]:
    """Unwrap aliases into a mapping of {name: expression}."""
    result = {}
    for value in values:
        node = value.op()
        if node.name in result:
            raise com.IbisInputError(
                f"Duplicate column name {node.name!r} in result set"
            )
        result[node.name] = unwrap_alias(node)
    return result


@public
class Table(Expr, _FixedTextJupyterMixin):
    """An immutable and lazy dataframe.

    Analogous to a SQL table or a pandas DataFrame. A table expression contains
    an [ordered set of named columns](./schemas.qmd#ibis.expr.schema.Schema),
    each with a single known type. Unless explicitly ordered with an
    [`.order_by()`](./expression-tables.qmd#ibis.expr.types.relations.Table.order_by),
    the order of rows is undefined.

    Table immutability means that the data underlying an Ibis `Table` cannot be modified: every
    method on a Table returns a new Table with those changes. Laziness
    means that an Ibis `Table` expression does not run your computation every time you call one of its methods.
    Instead, it is a symbolic expression that represents a set of operations
    to be performed, which typically is translated into a SQL query. That
    SQL query is then executed on a backend, where the data actually lives.
    The result (now small enough to be manageable) can then be materialized back
    into python as a pandas/pyarrow/python DataFrame/Column/scalar.

    You will not create Table objects directly. Instead, you will create one

    - from a pandas DataFrame, pyarrow table, Polars table, or raw python dicts/lists
      with [`ibis.memtable(df)`](./expression-tables.qmd#ibis.memtable)
    - from an existing table in a data platform with
      [`connection.table("name")`](./expression-tables.qmd#ibis.backends.duckdb.Backend.table)
    - from a file or URL, into a specific backend with
      [`connection.read_csv/parquet/json("path/to/file")`](../backends/duckdb.qmd#ibis.backends.duckdb.Backend.read_csv)
      (only some backends, typically local ones, support this)
    - from a file or URL, into the default backend with
       [`ibis.read_csv/read_json/read_parquet("path/to/file")`](./expression-tables.qmd#ibis.read_csv)

    See the [user guide](https://ibis-project.org/how-to/input-output/basics) for more
    info.
    """

    # Higher than numpy objects
    __array_priority__ = 20

    __array_ufunc__ = None

    def get_name(self) -> str:
        """Return the fully qualified name of the table.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.duckdb.connect()
        >>> t = con.create_table("t", {"id": [1, 2, 3]})
        >>> t.get_name()
        'memory.main.t'
        """
        arg = self._arg
        namespace = getattr(arg, "namespace", ops.Namespace())
        pieces = namespace.catalog, namespace.database, arg.name
        return ".".join(filter(None, pieces))

    def __array__(self, dtype=None):
        return self.execute().__array__(dtype)

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        from ibis.expr.types.dataframe_interchange import IbisDataFrame

        return IbisDataFrame(self, nan_as_null=nan_as_null, allow_copy=allow_copy)

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        return self.to_pyarrow().__arrow_c_stream__(requested_schema)

    def __pyarrow_result__(
        self,
        table: pa.Table,
        *,
        schema: sch.Schema | None = None,
        data_mapper: type[PyArrowData] | None = None,
    ) -> pa.Table:
        if data_mapper is None:
            from ibis.formats.pyarrow import PyArrowData as data_mapper

        return data_mapper.convert_table(
            table, self.schema() if schema is None else schema
        )

    def __pandas_result__(
        self,
        df: pd.DataFrame,
        *,
        schema: sch.Schema | None = None,
        data_mapper: type[PandasData] | None = None,
    ) -> pd.DataFrame:
        if data_mapper is None:
            from ibis.formats.pandas import PandasData as data_mapper

        return data_mapper.convert_table(
            df, self.schema() if schema is None else schema
        )

    def __polars_result__(self, df: pl.DataFrame) -> Any:
        from ibis.formats.polars import PolarsData

        return PolarsData.convert_table(df, self.schema())

    def _fast_bind(self, *args, **kwargs):
        # allow the first argument to be either a dictionary or a list of values
        if len(args) == 1:
            if isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
                args = ()
            else:
                args = util.promote_list(args[0])
        # bind positional arguments
        values = []
        for arg in args:
            values.extend(bind(self, arg))

        # bind keyword arguments where each entry can produce only one value
        # which is then named with the given key
        for key, arg in kwargs.items():
            bindings = tuple(bind(self, arg))
            if len(bindings) != 1:
                raise com.IbisInputError(
                    "Keyword arguments cannot produce more than one value"
                )
            (value,) = bindings
            values.append(value.name(key))
        return values

    def bind(self, *args: Any, **kwargs: Any) -> tuple[Value, ...]:
        """Bind column values to a table expression.

        This method handles the binding of every kind of column-like value that
        Ibis handles, including strings, integers, deferred expressions and
        selectors, to a table expression.

        Parameters
        ----------
        args
            Column-like values to bind.
        kwargs
            Column-like values to bind, with names.

        Returns
        -------
        tuple[Value, ...]
            A tuple of bound values
        """
        values = self._fast_bind(*args, **kwargs)
        # dereference the values to `self`
        dm = DerefMap.from_targets(self.op())
        result = []
        for original in values:
            value = dm.dereference(original.op()).to_expr()
            value = value.name(original.get_name())
            result.append(value)
        return tuple(result)

    def as_scalar(self) -> ir.Scalar:
        """Inform ibis that the table expression should be treated as a scalar.

        Note that the table must have exactly one column and one row for this to
        work. If the table has more than one column an error will be raised in
        expression construction time. If the table has more than one row an
        error will be raised by the backend when the expression is executed.

        Returns
        -------
        Scalar
            A scalar subquery

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> heavy_gentoo = t.filter(t.species == "Gentoo", t.body_mass_g > 6200)
        >>> from_that_island = t.filter(t.island == heavy_gentoo.select("island").as_scalar())
        >>> from_that_island.species.value_counts().order_by("species")
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━┓
        ┃ species ┃ species_count ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━┩
        │ string  │ int64         │
        ├─────────┼───────────────┤
        │ Adelie  │            44 │
        │ Gentoo  │           124 │
        └─────────┴───────────────┘
        """
        return ops.ScalarSubquery(self).to_expr()

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

    def cast(self, schema: SchemaLike) -> Table:
        """Cast the columns of a table.

        Similar to `pandas.DataFrame.astype`.

        ::: {.callout-note}
        ## If you need to cast columns to a single type, use [selectors](./selectors.qmd).
        :::

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
        >>> import ibis.selectors as s
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
        │        NULL │           NULL │
        │        3450 │           36.7 │
        └─────────────┴────────────────┘

        Columns not present in the input schema will be passed through unchanged

        >>> t.columns
        ('species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year')
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
        │        NULL │           NULL │
        │      3450.0 │             37 │
        └─────────────┴────────────────┘

        Columns that are in the input `schema` but not in the table raise an error

        >>> t.cast({"foo": "string"})  # quartodoc: +EXPECTED_FAILURE
        Traceback (most recent call last):
            ...
        ibis.common.exceptions.IbisError: Cast schema has fields that are not in the table: ['foo']
        """
        return self._cast(schema, cast_method="cast")

    def try_cast(self, schema: SchemaLike) -> Table:
        """Cast the columns of a table.

        If the cast fails for a row, the value is returned
        as `NULL` or `NaN` depending on backend behavior.

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
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": ["1", "2", "3"], "b": ["2.2", "3.3", "book"]})
        >>> t.try_cast({"a": "int", "b": "float"})
        ┏━━━━━━━┳━━━━━━━━━┓
        ┃ a     ┃ b       ┃
        ┡━━━━━━━╇━━━━━━━━━┩
        │ int64 │ float64 │
        ├───────┼─────────┤
        │     1 │     2.2 │
        │     2 │     3.3 │
        │     3 │    NULL │
        └───────┴─────────┘
        """
        return self._cast(schema, cast_method="try_cast")

    def _cast(self, schema: SchemaLike, cast_method: str = "cast") -> Table:
        schema = sch.schema(schema)

        cols = []

        columns = self.columns
        if missing_fields := frozenset(schema.names).difference(columns):
            raise com.IbisError(
                f"Cast schema has fields that are not in the table: {sorted(missing_fields)}"
            )

        for col in columns:
            if (new_type := schema.get(col)) is not None:
                new_col = getattr(self[col], cast_method)(new_type).name(col)
            else:
                new_col = col
            cols.append(new_col)
        return self.select(*cols)

    def preview(
        self,
        *,
        max_rows: int | None = None,
        max_columns: int | None = None,
        max_length: int | None = None,
        max_string: int | None = None,
        max_depth: int | None = None,
        console_width: int | float | None = None,
    ) -> RichTable:
        """Return a subset as a Rich Table.

        This is an explicit version of what you get when you inspect
        this object in interactive mode, except with this version you
        can pass formatting options. The options are the same as those exposed
        in `ibis.options.interactive`.

        Parameters
        ----------
        max_rows
            Maximum number of rows to display
        max_columns
            Maximum number of columns to display
        max_length
            Maximum length for pretty-printed arrays and maps
        max_string
            Maximum length for pretty-printed strings
        max_depth
            Maximum depth for nested data types
        console_width
            Width of the console in characters. If not specified, the width
            will be inferred from the console.

        Examples
        --------
        >>> import ibis
        >>> t = ibis.examples.penguins.fetch()

        Because the console_width is too small, only 2 columns are shown even though
        we specified up to 3.

        >>> t.preview(
        ...     max_rows=3,
        ...     max_columns=3,
        ...     max_string=8,
        ...     console_width=30,
        ... )  # doctest: +SKIP
        ┏━━━━━━━━━┳━━━━━━━━━━┳━━━┓
        ┃ species ┃ island   ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━╇━━━┩
        │ string  │ string   │ … │
        ├─────────┼──────────┼───┤
        │ Adelie  │ Torgers… │ … │
        │ Adelie  │ Torgers… │ … │
        │ Adelie  │ Torgers… │ … │
        │ …       │ …        │ … │
        └─────────┴──────────┴───┘
        """
        from ibis.expr.types.pretty import to_rich

        return to_rich(
            self,
            max_columns=max_columns,
            max_rows=max_rows,
            max_length=max_length,
            max_string=max_string,
            max_depth=max_depth,
            console_width=console_width,
        )

    @overload
    def __getitem__(self, what: str | int) -> ir.Column: ...

    @overload
    def __getitem__(self, what: slice | Sequence[str | int]) -> Table: ...

    def __getitem__(self, what: str | int | slice | Sequence[str | int]):
        """Select one or more columns or rows from a table expression.

        Parameters
        ----------
        what
            What to select. Options are:
            - A `str` column name or `int` column index to select a single column.
            - A sequence of column names or indices to select multiple columns.
            - A slice to select a subset of rows.

        Returns
        -------
        Table | Column
            The return type depends on the input. For a single string or int
            input a column is returned, otherwise a table is returned.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch().head()
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
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        Select a single column by name:

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
        └───────────┘

        Select a single column by index:

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
        └───────────┘

        Select multiple columns by name:

        >>> t[["island", "bill_length_mm"]]
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │           NULL │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘

        Select a range of rows:

        >>> t[:2]
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t[2:5]
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           40.3 │          18.0 │               195 │ … │
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        """
        from ibis.expr.types.logical import BooleanValue

        if isinstance(what, str):
            return ops.Field(self.op(), what).to_expr()
        elif isinstance(what, int):
            return ops.Field(self.op(), self.columns[what]).to_expr()
        elif isinstance(what, slice):
            limit, offset = util.slice_to_limit_offset(what, self.count())
            return self.limit(limit, offset=offset)

        columns = self.columns
        args = [
            columns[arg] if isinstance(arg, int) else arg
            for arg in util.promote_list(what)
        ]
        if util.all_of(args, str):
            return self.select(args)

        # Once this deprecation is removed, we'll want to error here instead.
        warnings.warn(
            "Selecting/filtering arbitrary expressions in `Table.__getitem__` is "
            "deprecated and will be removed in version 10.0. Please use "
            "`Table.select` or `Table.filter` instead.",
            FutureWarning,
            stacklevel=2,
        )
        values = self.bind(args)

        if util.all_of(values, BooleanValue):
            return self.filter(values)
        else:
            return self.select(values)

    def __len__(self) -> NoReturn:
        raise com.ExpressionError("Use .count() instead")

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
        try:
            return ops.Field(self, key).to_expr()
        except com.IbisTypeError:
            pass

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

    @property
    def columns(self) -> tuple[str, ...]:
        """Return a [](`tuple`) of column names in this table.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.columns
        ('species',
         'island',
         'bill_length_mm',
         'bill_depth_mm',
         'flipper_length_mm',
         'body_mass_g',
         'sex',
         'year')
        """
        return self._arg.schema.names

    def schema(self) -> sch.Schema:
        """Return the [Schema](./schemas.qmd#ibis.expr.schema.Schema) for this table.

        Returns
        -------
        Schema
            The table's schema.

        Examples
        --------
        >>> import ibis
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
        """
        return self.op().schema

    def group_by(
        self,
        *by: str | ir.Value | Iterable[str] | Iterable[ir.Value] | None,
        **key_exprs: str | ir.Value | Iterable[str] | Iterable[ir.Value],
    ) -> GroupedTable:
        """Create a grouped table expression.

        Similar to SQL's GROUP BY statement, or pandas .groupby() method.

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
        >>> t = ibis.memtable(
        ...     {
        ...         "fruit": ["apple", "apple", "banana", "orange"],
        ...         "price": [0.5, 0.5, 0.25, 0.33],
        ...     }
        ... )
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
        >>> t.group_by("fruit").agg(total_cost=_.price.sum(), avg_cost=_.price.mean()).order_by(
        ...     "fruit"
        ... )
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

        by = tuple(v for v in by if v is not None)
        groups = self.bind(*by, **key_exprs)
        return GroupedTable(self, groups)

    # TODO(kszucs): shouldn't this be ibis.rowid() instead not bound to a specific table?
    def rowid(self) -> ir.IntegerValue:
        """A unique integer per row.

        ::: {.callout-note}
        ## This operation is only valid on physical tables

        Any further meaning behind this expression is backend dependent.
        Generally this corresponds to some index into the database storage
        (for example, SQLite and DuckDB's `rowid`).

        For a monotonically increasing row number, see `ibis.row_number`.
        :::

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
        if isinstance(self.op(), ops.SelfReference):
            return self
        else:
            return ops.SelfReference(self).to_expr()

    def aggregate(
        self,
        metrics: Sequence[ir.Scalar] | None = (),
        by: Sequence[ir.Value] | None = (),
        having: Sequence[ir.BooleanValue] | None = (),
        **kwargs: ir.Value,
    ) -> Table:
        """Aggregate a table with a given set of reductions grouping by `by`.

        Parameters
        ----------
        metrics
            Aggregate expressions. These can be any scalar-producing
            expression, including aggregation functions like `sum` or literal
            values like `ibis.literal(1)`.
        by
            Grouping expressions.
        having
            Post-aggregation filters. The shape requirements are the same
            `metrics`, but the output type for `having` is `boolean`.

            ::: {.callout-warning}
            ## Expressions like `x is None` return `bool` and **will not** generate a SQL comparison to `NULL`
            :::
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
        >>> t = ibis.memtable(
        ...     {
        ...         "fruit": ["apple", "apple", "banana", "orange"],
        ...         "price": [0.5, 0.5, 0.25, 0.33],
        ...     }
        ... )
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
        >>> t.aggregate(
        ...     by=["fruit"],
        ...     total_cost=_.price.sum(),
        ...     avg_cost=_.price.mean(),
        ...     having=_.price.sum() < 0.5,
        ... )
        ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ fruit  ┃ total_cost ┃ avg_cost ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
        │ string │ float64    │ float64  │
        ├────────┼────────────┼──────────┤
        │ banana │       0.25 │     0.25 │
        │ orange │       0.33 │     0.33 │
        └────────┴────────────┴──────────┘
        """
        from ibis.common.patterns import Contains, In
        from ibis.expr.rewrites import p

        node = self.op()

        groups = self.bind(by)
        metrics = self.bind(metrics, **kwargs)
        having = self.bind(having)

        groups = unwrap_aliases(groups)
        metrics = unwrap_aliases(metrics)

        # the user doesn't need to specify the metrics used in the having clause
        # explicitly, we implicitly add them to the metrics list by looking for
        # any metrics depending on self which are not specified explicitly
        pattern = p.Reduction(relations=Contains(node)) & ~In(set(metrics.values()))
        original_metrics = metrics.copy()
        for pred in having:
            for metric in pred.op().find_topmost(pattern):
                if metric.name in metrics:
                    metrics[util.get_name("metric")] = metric
                else:
                    metrics[metric.name] = metric

        # construct the aggregate node
        agg = ops.Aggregate(node, groups, metrics).to_expr()

        if having:
            # apply the having clause
            agg = agg.filter(having)
            # remove any metrics that were only used in the having clause
            if metrics != original_metrics:
                agg = agg.select(*groups.keys(), *original_metrics.keys())

        return agg

    agg = aggregate

    def distinct(
        self,
        *,
        on: str | Iterable[str] | s.Selector | None = None,
        keep: Literal["first", "last"] | None = "first",
    ) -> Table:
        """Return a Table with duplicate rows removed.

        Similar to `pandas.DataFrame.drop_duplicates()`.

        ::: {.callout-note}
        ## Some backends do not support `keep='last'`
        :::

        Parameters
        ----------
        on
            Only consider certain columns for identifying duplicates.
            By default deduplicate all of the columns.
        keep
            Determines which duplicates to keep.

            - `"first"`: Drop duplicates except for the first occurrence.
            - `"last"`: Drop duplicates except for the last occurrence.
            - `None`: Drop all duplicates

        Examples
        --------
        >>> import ibis
        >>> import ibis.examples as ex
        >>> import ibis.selectors as s
        >>> ibis.options.interactive = True
        >>> t = ex.penguins.fetch()
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

        Compute the distinct rows of a subset of columns

        >>> t[["species", "island"]].distinct().order_by(s.all())
        ┏━━━━━━━━━━━┳━━━━━━━━━━━┓
        ┃ species   ┃ island    ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ string    │ string    │
        ├───────────┼───────────┤
        │ Adelie    │ Biscoe    │
        │ Adelie    │ Dream     │
        │ Adelie    │ Torgersen │
        │ Chinstrap │ Dream     │
        │ Gentoo    │ Biscoe    │
        └───────────┴───────────┘

        Drop all duplicate rows except the first

        >>> t.distinct(on=["species", "island"], keep="first").order_by(s.all())
        ┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━┓
        ┃ species   ┃ island    ┃ bill_length_mm ┃ bill_depth_… ┃ flipper_length_mm ┃  ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━┩
        │ string    │ string    │ float64        │ float64      │ int64             │  │
        ├───────────┼───────────┼────────────────┼──────────────┼───────────────────┼──┤
        │ Adelie    │ Biscoe    │           37.8 │         18.3 │               174 │  │
        │ Adelie    │ Dream     │           39.5 │         16.7 │               178 │  │
        │ Adelie    │ Torgersen │           39.1 │         18.7 │               181 │  │
        │ Chinstrap │ Dream     │           46.5 │         17.9 │               192 │  │
        │ Gentoo    │ Biscoe    │           46.1 │         13.2 │               211 │  │
        └───────────┴───────────┴────────────────┴──────────────┴───────────────────┴──┘

        Drop all duplicate rows except the last

        >>> t.distinct(on=["species", "island"], keep="last").order_by(s.all())
        ┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━┓
        ┃ species   ┃ island    ┃ bill_length_mm ┃ bill_depth_… ┃ flipper_length_mm ┃  ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━┩
        │ string    │ string    │ float64        │ float64      │ int64             │  │
        ├───────────┼───────────┼────────────────┼──────────────┼───────────────────┼──┤
        │ Adelie    │ Biscoe    │           42.7 │         18.3 │               196 │  │
        │ Adelie    │ Dream     │           41.5 │         18.5 │               201 │  │
        │ Adelie    │ Torgersen │           43.1 │         19.2 │               197 │  │
        │ Chinstrap │ Dream     │           50.2 │         18.7 │               198 │  │
        │ Gentoo    │ Biscoe    │           49.9 │         16.1 │               213 │  │
        └───────────┴───────────┴────────────────┴──────────────┴───────────────────┴──┘

        Drop all duplicated rows

        >>> expr = t.distinct(on=["species", "island", "year", "bill_length_mm"], keep=None)
        >>> expr.count()
        ┌─────┐
        │ 273 │
        └─────┘
        >>> t.count()
        ┌─────┐
        │ 344 │
        └─────┘

        You can pass [`selectors`](./selectors.qmd) to `on`

        >>> t.distinct(on=~s.numeric())  # doctest: +SKIP
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ species ┃ island    ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string    │ float64        │ float64       │ int64             │ … │
        ├─────────┼───────────┼────────────────┼───────────────┼───────────────────┼───┤
        │ Adelie  │ Torgersen │           39.1 │          18.7 │               181 │ … │
        │ Adelie  │ Torgersen │           39.5 │          17.4 │               186 │ … │
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Biscoe    │           37.8 │          18.3 │               174 │ … │
        │ Adelie  │ Biscoe    │           37.7 │          18.7 │               180 │ … │
        │ Adelie  │ Dream     │           39.5 │          16.7 │               178 │ … │
        │ Adelie  │ Dream     │           37.2 │          18.1 │               178 │ … │
        │ Adelie  │ Dream     │           37.5 │          18.9 │               179 │ … │
        │ Gentoo  │ Biscoe    │           46.1 │          13.2 │               211 │ … │
        │ Gentoo  │ Biscoe    │           50.0 │          16.3 │               230 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘

        The only valid values of `keep` are `"first"`, `"last"` and [](`None`).

        >>> t.distinct(on="species", keep="second")  # quartodoc: +EXPECTED_FAILURE
        Traceback (most recent call last):
          ...
        ibis.common.exceptions.IbisError: Invalid value for keep: 'second' ...
        """

        import ibis.selectors as s

        if on is None:
            # dedup everything
            if keep != "first":
                raise com.IbisError(
                    f"Only keep='first' (the default) makes sense when deduplicating all columns; got keep={keep!r}"
                )
            return ops.Distinct(self).to_expr()

        on = s._to_selector(on)

        if keep is None:
            having = lambda t: t.count() == 1
            method = "first"
        elif keep in ("first", "last"):
            having = ()
            method = keep
        else:
            raise com.IbisError(
                f"Invalid value for `keep`: {keep!r}, must be 'first', 'last' or None"
            )

        aggs = {col.get_name(): getattr(col, method)() for col in (~on).expand(self)}
        res = self.aggregate(aggs, by=on, having=having)

        assert len(res.columns) == len(self.columns)
        if res.columns != self.columns:
            return res.select(self.columns)
        return res

    def sample(
        self,
        fraction: float,
        *,
        method: Literal["row", "block"] = "row",
        seed: int | None = None,
    ) -> Table:
        """Sample a fraction of rows from a table.

        ::: {.callout-note}
        ## Results may be non-repeatable

        Sampling is by definition a random operation. Some backends support
        specifying a `seed` for repeatable results, but not all backends
        support that option. And some backends (duckdb, for example) do support
        specifying a seed but may still not have repeatable results in all
        cases.

        In all cases, results are backend-specific. An execution against one
        backend is unlikely to sample the same rows when executed against a
        different backend, even with the same `seed` set.
        :::

        Parameters
        ----------
        fraction
            The percentage of rows to include in the sample, expressed as a
            float between 0 and 1.
        method
            The sampling method to use. The default is "row", which includes
            each row with a probability of `fraction`. If method is "block",
            some backends may instead sample a fraction of blocks of rows
            (where "block" is a backend dependent definition), which may be
            significantly more efficient (at the cost of a less statistically
            random sample). This is identical to "row" for backends lacking a
            blockwise sampling implementation. For those coming from SQL, "row"
            and "block" correspond to "bernoulli" and "system" respectively in
            a TABLESAMPLE clause.
        seed
            An optional random seed to use, for repeatable sampling. The range
            of possible seed values is backend specific (most support at least
            `[0, 2**31 - 1]`). Backends that never support specifying a seed
            for repeatable sampling will error appropriately. Note that some
            backends (like DuckDB) do support specifying a seed, but may still
            not have repeatable results in all cases.

        Returns
        -------
        Table
            The input table, with `fraction` of rows selected.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"x": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]})
        >>> t
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ x     ┃ y      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ a      │
        │     2 │ b      │
        │     3 │ c      │
        │     4 │ d      │
        └───────┴────────┘

        Sample approximately half the rows, with a seed specified for
        reproducibility.

        >>> t.sample(0.5, seed=1234)
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ x     ┃ y      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     2 │ b      │
        │     3 │ c      │
        └───────┴────────┘
        """
        if fraction == 1:
            return self
        elif fraction == 0:
            return self.limit(0)
        else:
            return ops.Sample(
                self, fraction=fraction, method=method, seed=seed
            ).to_expr()

    def limit(self, n: int | None, offset: int = 0) -> Table:
        """Select `n` rows from `self` starting at `offset`.

        ::: {.callout-note}
        ## The result set is not deterministic without a call to [`order_by`](#ibis.expr.types.relations.Table.order_by).
        :::

        Parameters
        ----------
        n
            Number of rows to include. If `None`, the entire table is selected
            starting from `offset`.
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

        You can use `None` with `offset` to slice starting from a particular row

        >>> t.limit(None, offset=1)
        ┏━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b      ┃
        ┡━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │
        ├───────┼────────┤
        │     1 │ a      │
        │     2 │ a      │
        └───────┴────────┘

        See Also
        --------
        [`Table.order_by`](#ibis.expr.types.relations.Table.order_by)
        """
        return ops.Limit(self, n, offset).to_expr()

    def head(self, n: int = 5) -> Table:
        """Select the first `n` rows of a table.

        ::: {.callout-note}
        ## The result set is not deterministic without a call to [`order_by`](#ibis.expr.types.relations.Table.order_by).
        :::

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
        [`Table.limit`](#ibis.expr.types.relations.Table.limit)
        [`Table.order_by`](#ibis.expr.types.relations.Table.order_by)
        """
        return self.limit(n=n)

    def order_by(
        self,
        *by: str
        | ir.Column
        | s.Selector
        | Sequence[str]
        | Sequence[ir.Column]
        | Sequence[s.Selector]
        | None,
    ) -> Table:
        """Sort a table by one or more expressions.

        Similar to `pandas.DataFrame.sort_values()`.

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
        >>> t = ibis.memtable(
        ...     {
        ...         "a": [3, 2, 1, 3],
        ...         "b": ["a", "B", "c", "D"],
        ...         "c": [4, 6, 5, 7],
        ...     }
        ... )
        >>> t
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     3 │ a      │     4 │
        │     2 │ B      │     6 │
        │     1 │ c      │     5 │
        │     3 │ D      │     7 │
        └───────┴────────┴───────┘

        Sort by b. Default is ascending. Note how capital letters come before lowercase

        >>> t.order_by("b")
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     2 │ B      │     6 │
        │     3 │ D      │     7 │
        │     3 │ a      │     4 │
        │     1 │ c      │     5 │
        └───────┴────────┴───────┘

        Sort in descending order

        >>> t.order_by(ibis.desc("b"))
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     1 │ c      │     5 │
        │     3 │ a      │     4 │
        │     3 │ D      │     7 │
        │     2 │ B      │     6 │
        └───────┴────────┴───────┘

        You can also use the deferred API to get the same result

        >>> from ibis import _
        >>> t.order_by(_.b.desc())
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     1 │ c      │     5 │
        │     3 │ a      │     4 │
        │     3 │ D      │     7 │
        │     2 │ B      │     6 │
        └───────┴────────┴───────┘

        Sort by multiple columns/expressions

        >>> t.order_by(["a", _.c.desc()])
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     1 │ c      │     5 │
        │     2 │ B      │     6 │
        │     3 │ D      │     7 │
        │     3 │ a      │     4 │
        └───────┴────────┴───────┘

        You can actually pass arbitrary expressions to use as sort keys.
        For example, to ignore the case of the strings in column `b`

        >>> t.order_by(_.b.lower())
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     3 │ a      │     4 │
        │     2 │ B      │     6 │
        │     1 │ c      │     5 │
        │     3 │ D      │     7 │
        └───────┴────────┴───────┘

        This means that shuffling a Table is super simple

        >>> t.order_by(ibis.random())  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ string │ int64 │
        ├───────┼────────┼───────┤
        │     1 │ c      │     5 │
        │     3 │ D      │     7 │
        │     3 │ a      │     4 │
        │     2 │ B      │     6 │
        └───────┴────────┴───────┘

        [Selectors](./selectors.qmd) are allowed as sort keys and are a concise way to sort by
        multiple columns matching some criteria

        >>> import ibis.selectors as s
        >>> penguins = ibis.examples.penguins.fetch()
        >>> penguins[["year", "island"]].value_counts().order_by(s.startswith("year"))
        ┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ year  ┃ island    ┃ year_island_count ┃
        ┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ int64 │ string    │ int64             │
        ├───────┼───────────┼───────────────────┤
        │  2007 │ Torgersen │                20 │
        │  2007 │ Biscoe    │                44 │
        │  2007 │ Dream     │                46 │
        │  2008 │ Torgersen │                16 │
        │  2008 │ Dream     │                34 │
        │  2008 │ Biscoe    │                64 │
        │  2009 │ Torgersen │                16 │
        │  2009 │ Dream     │                44 │
        │  2009 │ Biscoe    │                60 │
        └───────┴───────────┴───────────────────┘

        Use the [`across`](./selectors.qmd#ibis.selectors.across) selector to
        apply a specific order to multiple columns

        >>> penguins[["year", "island"]].value_counts().order_by(
        ...     s.across(s.startswith("year"), _.desc())
        ... )
        ┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ year  ┃ island    ┃ year_island_count ┃
        ┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ int64 │ string    │ int64             │
        ├───────┼───────────┼───────────────────┤
        │  2009 │ Biscoe    │                60 │
        │  2009 │ Dream     │                44 │
        │  2009 │ Torgersen │                16 │
        │  2008 │ Biscoe    │                64 │
        │  2008 │ Dream     │                34 │
        │  2008 │ Torgersen │                16 │
        │  2007 │ Dream     │                46 │
        │  2007 │ Biscoe    │                44 │
        │  2007 │ Torgersen │                20 │
        └───────┴───────────┴───────────────────┘
        """
        keys = self.bind(*by)
        keys = unwrap_aliases(keys)
        if not keys:
            raise com.IbisError("At least one sort key must be provided")

        node = ops.Sort(self, keys.values())
        return node.to_expr()

    def _assemble_set_op(
        self, opcls: type[Set], table: Table, *rest: Table, distinct: bool
    ) -> Table:
        """Assemble a set operation expression.

        This exists to workaround an issue in sqlglot where codegen blows the
        Python stack because of set operation nesting.

        The implementation here uses a queue to balance the operation tree.
        """
        queue = deque()

        queue.append(self)
        queue.append(table)
        queue.extend(rest)

        while len(queue) > 1:
            left = queue.popleft()
            right = queue.popleft()
            node = opcls(left, right, distinct=distinct)
            queue.append(node)
        result = queue.popleft()
        assert not queue, "items left in queue"
        return result.to_expr()

    def union(self, table: Table, *rest: Table, distinct: bool = False) -> Table:
        """Compute the set union of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        table
            A table expression
        *rest
            Additional table expressions
        distinct
            Only return distinct rows

        Returns
        -------
        Table
            A new table containing the union of all input tables.

        See Also
        --------
        [`ibis.union`](./expression-tables.qmd#ibis.union)

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
        return self._assemble_set_op(ops.Union, table, *rest, distinct=distinct)

    def intersect(self, table: Table, *rest: Table, distinct: bool = True) -> Table:
        """Compute the set intersection of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        table
            A table expression
        *rest
            Additional table expressions
        distinct
            Only return distinct rows

        Returns
        -------
        Table
            A new table containing the intersection of all input tables.

        See Also
        --------
        [`ibis.intersect`](./expression-tables.qmd#ibis.intersect)

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t1 = ibis.memtable({"a": [1, 2, 2]})
        >>> t1
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     1 │
        │     2 │
        │     2 │
        └───────┘
        >>> t2 = ibis.memtable({"a": [2, 2, 3]})
        >>> t2
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
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
        >>> t1.intersect(t2, distinct=False)
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        │     2 │
        └───────┘

        More than two table expressions can be intersected at once.
        >>> t3 = ibis.memtable({"a": [2, 3, 3]})
        >>> t1.intersect(t2, t3)
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     2 │
        └───────┘
        """
        return self._assemble_set_op(ops.Intersection, table, *rest, distinct=distinct)

    def difference(self, table: Table, *rest: Table, distinct: bool = True) -> Table:
        """Compute the set difference of multiple table expressions.

        The input tables must have identical schemas.

        Parameters
        ----------
        table:
            A table expression
        *rest:
            Additional table expressions
        distinct
            Only diff distinct rows not occurring in the calling table

        See Also
        --------
        [`ibis.difference`](./expression-tables.qmd#ibis.difference)

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
        node = ops.Difference(self, table, distinct=distinct)
        for expr in rest:
            node = ops.Difference(node, expr, distinct=distinct)
        return node.to_expr()

    @deprecated(as_of="9.0", instead="use table.as_scalar() instead")
    def to_array(self) -> ir.Column:
        """Deprecated - use `as_scalar` instead."""

        schema = self.schema()
        if len(schema) != 1:
            raise com.ExpressionError(
                "Table must have exactly one column when viewed as array"
            )
        return self.as_scalar()

    def mutate(self, *exprs: Sequence[ir.Expr] | None, **mutations: ir.Value) -> Table:
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
        >>> import ibis.selectors as s
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
        │ Adelie  │  2007 │           NULL │
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
        │ Adelie  │  2007 │           NULL │      2008 │
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
        │ Adelie  │        NULL │
        │ Adelie  │    -7.22193 │
        └─────────┴─────────────┘

        Mutate across multiple columns

        >>> t.mutate(s.across(s.numeric() & ~s.cols("year"), _ - _.mean())).head()
        ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ species ┃ year  ┃ bill_length_mm ┃
        ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string  │ int64 │ float64        │
        ├─────────┼───────┼────────────────┤
        │ Adelie  │  2007 │       -4.82193 │
        │ Adelie  │  2007 │       -4.42193 │
        │ Adelie  │  2007 │       -3.62193 │
        │ Adelie  │  2007 │           NULL │
        │ Adelie  │  2007 │       -7.22193 │
        └─────────┴───────┴────────────────┘
        """
        # string and integer inputs are going to be coerced to literals instead
        # of interpreted as column references like in select
        node = self.op()
        values = self.bind(*exprs, **mutations)
        values = unwrap_aliases(values)
        # allow overriding of fields, hence the mutation behavior
        values = {**node.fields, **values}
        return self.select(**values)

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
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
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
        │ Torgersen │           NULL │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘

        In that simple case, you could also just use python's indexing syntax

        >>> t[["island", "bill_length_mm"]].head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ bill_length_mm ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string    │ float64        │
        ├───────────┼────────────────┤
        │ Torgersen │           39.1 │
        │ Torgersen │           39.5 │
        │ Torgersen │           40.3 │
        │ Torgersen │           NULL │
        │ Torgersen │           36.7 │
        └───────────┴────────────────┘

        Projection by zero-indexed column position

        >>> t.select(t[0], t[4]).head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ species ┃ flipper_length_mm ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ string  │ int64             │
        ├─────────┼───────────────────┤
        │ Adelie  │               181 │
        │ Adelie  │               186 │
        │ Adelie  │               195 │
        │ Adelie  │              NULL │
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

        You can do the same thing with a named expression, and using the
        deferred API

        >>> from ibis import _
        >>> t.select((_.year + 1).name("next_year")).head()
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

        >>> import ibis.selectors as s
        >>> t.select(s.numeric() & ~s.cols("year")).head()
        ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
        ┃ bill_length_mm ┃ bill_depth_mm ┃ flipper_length_mm ┃ body_mass_g ┃
        ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
        │ float64        │ float64       │ int64             │ int64       │
        ├────────────────┼───────────────┼───────────────────┼─────────────┤
        │           39.1 │          18.7 │               181 │        3750 │
        │           39.5 │          17.4 │               186 │        3800 │
        │           40.3 │          18.0 │               195 │        3250 │
        │           NULL │          NULL │              NULL │        NULL │
        │           36.7 │          19.3 │               193 │        3450 │
        └────────────────┴───────────────┴───────────────────┴─────────────┘

        Projection + aggregation across multiple columns

        >>> from ibis import _
        >>> t.select(s.across(s.numeric() & ~s.cols("year"), _.mean())).head()
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
        from ibis.expr.rewrites import rewrite_project_input

        values = self.bind(*exprs, **named_exprs)
        values = unwrap_aliases(values)
        if not values:
            raise com.IbisTypeError(
                "You must select at least one column for a valid projection"
            )

        # we need to detect reductions which are either turned into window functions
        # or scalar subqueries depending on whether they are originating from self
        values = {
            k: rewrite_project_input(v, relation=self.op()) for k, v in values.items()
        }
        return ops.Project(self, values).to_expr()

    projection = select

    @util.deprecated(
        as_of="7.0",
        instead=(
            "use `Table.rename` instead (if passing a mapping, note the meaning "
            "of keys and values are swapped in Table.rename)."
        ),
    )
    def relabel(
        self,
        substitutions: (
            Mapping[str, str]
            | Callable[[str], str | None]
            | str
            | Literal["snake_case", "ALL_CAPS"]
        ),
    ) -> Table:
        """Deprecated in favor of `Table.rename`."""
        if isinstance(substitutions, Mapping):
            substitutions = {new: old for old, new in substitutions.items()}
        return self.rename(substitutions)

    def rename(
        self,
        method: (
            str
            | Callable[[str], str | None]
            | Literal["snake_case", "ALL_CAPS"]
            | Mapping[str, str]
            | None
        ) = None,
        /,
        **substitutions: str,
    ) -> Table:
        """Rename columns in the table.

        Parameters
        ----------
        method
            An optional method for renaming columns. May be one of:

            - A format string to use to rename all columns, like
              `"prefix_{name}"`.
            - A function from old name to new name. If the function returns
              `None` the old name is used.
            - The literal strings `"snake_case"` or `"ALL_CAPS"` to
              rename all columns using a `snake_case` or `"ALL_CAPS"``
              naming convention respectively.
            - A mapping from new name to old name. Existing columns not present
              in the mapping will passthrough with their original name.
        substitutions
            Columns to be explicitly renamed, expressed as `new_name=old_name``
            keyword arguments.

        Returns
        -------
        Table
            A renamed table expression

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> ibis.options.interactive = True
        >>> first3 = s.index[:3]  # first 3 columns
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

        Rename specific columns by passing keyword arguments like
        `new_name="old_name"``

        >>> t.rename(study_name="studyName").head(1)
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ study_name ┃ Sample Number ┃ Species                             ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string     │ int64         │ string                              │
        ├────────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708    │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └────────────┴───────────────┴─────────────────────────────────────┘

        Rename all columns using a format string

        >>> t.rename("p_{name}").head(1)
        ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ p_studyName ┃ p_Sample Number ┃ p_Species                           ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string      │ int64           │ string                              │
        ├─────────────┼─────────────────┼─────────────────────────────────────┤
        │ PAL0708     │               1 │ Adelie Penguin (Pygoscelis adeliae) │
        └─────────────┴─────────────────┴─────────────────────────────────────┘

        Rename all columns using a snake_case convention

        >>> t.rename("snake_case").head(1)
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ study_name ┃ sample_number ┃ species                             ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string     │ int64         │ string                              │
        ├────────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708    │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └────────────┴───────────────┴─────────────────────────────────────┘

        Rename all columns using an ALL_CAPS convention

        >>> t.rename("ALL_CAPS").head(1)
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ STUDY_NAME ┃ SAMPLE_NUMBER ┃ SPECIES                             ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string     │ int64         │ string                              │
        ├────────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708    │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └────────────┴───────────────┴─────────────────────────────────────┘

        Rename all columns using a callable

        >>> t.rename(str.upper).head(1)
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ STUDYNAME ┃ SAMPLE NUMBER ┃ SPECIES                             ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string    │ int64         │ string                              │
        ├───────────┼───────────────┼─────────────────────────────────────┤
        │ PAL0708   │             1 │ Adelie Penguin (Pygoscelis adeliae) │
        └───────────┴───────────────┴─────────────────────────────────────┘
        """
        if isinstance(method, Mapping):
            substitutions.update(method)
            method = None

        # A mapping from old_name -> renamed expr
        renamed = {}

        for new_name, old_name in substitutions.items():
            if old_name not in renamed:
                renamed[old_name] = (new_name, self[old_name].op())
            else:
                raise ValueError("duplicate new names passed for renaming {old_name!r}")

        if isinstance(method, str) and method in {"snake_case", "ALL_CAPS"}:

            def rename(c):
                c = c.strip()
                if " " in c:
                    # Handle "space case possibly with-hyphens"
                    if method == "snake_case":
                        return "_".join(c.lower().split()).replace("-", "_")
                    elif method == "ALL_CAPS":
                        return "_".join(c.upper().split()).replace("-", "_")
                # Handle PascalCase, camelCase, and kebab-case
                c = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", c)
                c = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", c)
                c = c.replace("-", "_")
                if method == "snake_case":
                    return c.lower()
                elif method == "ALL_CAPS":
                    return c.upper()
                else:
                    return None

        elif isinstance(method, str):

            def rename(name):
                return method.format(name=name)

            # Detect the case of missing or extra format string parameters
            try:
                dummy_name1 = "_unlikely_column_name_1_"
                dummy_name2 = "_unlikely_column_name_2_"
                invalid = rename(dummy_name1) == rename(dummy_name2)
            except KeyError:
                invalid = True
            if invalid:
                raise ValueError("Format strings must take a single parameter `name`")
        else:
            rename = method

        exprs = {}
        fields = self.op().fields
        for c in self.columns:
            if (new_name_op := renamed.get(c)) is not None:
                new_name, op = new_name_op
            else:
                op = fields[c]
                if rename is None or (new_name := rename(c)) is None:
                    new_name = c

            exprs[new_name] = op

        return ops.Project(self, exprs).to_expr()

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
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
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
        │ Torgersen │           NULL │          NULL │              NULL │ … │
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
        │ Torgersen │          NULL │              NULL │        NULL │ NULL   │ … │
        │ Torgersen │          19.3 │               193 │        3450 │ female │ … │
        └───────────┴───────────────┴───────────────────┴─────────────┴────────┴───┘

        Drop with selectors, mix and match

        >>> import ibis.selectors as s
        >>> t.drop("species", s.startswith("bill_")).head()
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ island    ┃ flipper_length_mm ┃ body_mass_g ┃ sex    ┃ year  ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ string    │ int64             │ int64       │ string │ int64 │
        ├───────────┼───────────────────┼─────────────┼────────┼───────┤
        │ Torgersen │               181 │        3750 │ male   │  2007 │
        │ Torgersen │               186 │        3800 │ female │  2007 │
        │ Torgersen │               195 │        3250 │ female │  2007 │
        │ Torgersen │              NULL │        NULL │ NULL   │  2007 │
        │ Torgersen │               193 │        3450 │ female │  2007 │
        └───────────┴───────────────────┴─────────────┴────────┴───────┘
        """
        if not fields:
            # no-op if nothing to be dropped
            return self

        columns_to_drop = frozenset(map(Expr.get_name, self._fast_bind(*fields)))
        return ops.DropColumns(parent=self, columns_to_drop=columns_to_drop).to_expr()

    def filter(
        self,
        *predicates: ir.BooleanValue | Sequence[ir.BooleanValue] | IfAnyAll,
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
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t.filter([t.species == "Adelie", t.body_mass_g > 3500]).sex.value_counts().drop_null(
        ...     "sex"
        ... ).order_by("sex")
        ┏━━━━━━━━┳━━━━━━━━━━━┓
        ┃ sex    ┃ sex_count ┃
        ┡━━━━━━━━╇━━━━━━━━━━━┩
        │ string │ int64     │
        ├────────┼───────────┤
        │ female │        22 │
        │ male   │        68 │
        └────────┴───────────┘
        """
        from ibis.expr.rewrites import flatten_predicates, rewrite_filter_input

        preds = self.bind(*predicates)

        # we can't use `unwrap_aliases` here because that function
        # deduplicates based on name alone
        #
        # it's perfectly valid to repeat a filter, even if it might be
        # useless, so enforcing uniquely named expressions here doesn't make
        # sense
        #
        # instead, compute all distinct unaliased predicates
        result = toolz.unique(
            node.arg if isinstance(node := value.op(), ops.Alias) else node
            for value in preds
        )

        preds = flatten_predicates(list(result))
        preds = list(map(rewrite_filter_input, preds))
        if not preds:
            raise com.IbisInputError("You must pass at least one predicate to filter")
        return ops.Filter(self, preds).to_expr()

    def nunique(self, where: ir.BooleanValue | None = None) -> ir.IntegerScalar:
        """Compute the number of unique rows in the table.

        Parameters
        ----------
        where
            Optional boolean expression to filter rows when counting.

        Returns
        -------
        IntegerScalar
            Number of unique rows in the table

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": ["foo", "bar", "bar"]})
        >>> t
        ┏━━━━━━━━┓
        ┃ a      ┃
        ┡━━━━━━━━┩
        │ string │
        ├────────┤
        │ foo    │
        │ bar    │
        │ bar    │
        └────────┘
        >>> t.nunique()
        ┌───┐
        │ 2 │
        └───┘
        >>> t.nunique(t.a != "foo")
        ┌───┐
        │ 1 │
        └───┘
        """
        if where is not None:
            (where,) = bind(self, where)
        return ops.CountDistinctStar(self, where=where).to_expr()

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
        ┌───┐
        │ 3 │
        └───┘
        >>> t.count(t.a != "foo")
        ┌───┐
        │ 2 │
        └───┘
        >>> type(t.count())
        <class 'ibis.expr.types.numeric.IntegerScalar'>
        """
        if where is not None:
            (where,) = bind(self, where)
        return ops.CountStar(self, where=where).to_expr()

    def drop_null(
        self,
        subset: Sequence[str] | str | None = None,
        how: Literal["any", "all"] = "any",
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
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        │ Adelie  │ Torgersen │           39.3 │          20.6 │               190 │ … │
        │ Adelie  │ Torgersen │           38.9 │          17.8 │               181 │ … │
        │ Adelie  │ Torgersen │           39.2 │          19.6 │               195 │ … │
        │ Adelie  │ Torgersen │           34.1 │          18.1 │               193 │ … │
        │ Adelie  │ Torgersen │           42.0 │          20.2 │               190 │ … │
        │ …       │ …         │              … │             … │                 … │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        >>> t.count()
        ┌─────┐
        │ 344 │
        └─────┘
        >>> t.drop_null(["bill_length_mm", "body_mass_g"]).count()
        ┌─────┐
        │ 342 │
        └─────┘
        >>> t.drop_null(how="all").count()  # no rows where all columns are null
        ┌─────┐
        │ 344 │
        └─────┘
        """
        if subset is not None:
            subset = self.bind(subset)
        return ops.DropNull(self, how, subset).to_expr()

    def fill_null(
        self,
        replacements: ir.Scalar | Mapping[str, ir.Scalar],
    ) -> Table:
        """Fill null values in a table expression.

        ::: {.callout-note}
        ## There is potential lack of type stability with the `fill_null` API

        For example, different library versions may impact whether a given
        backend promotes integer replacement values to floats.
        :::

        Parameters
        ----------
        replacements
            Value with which to fill nulls. If `replacements` is a mapping, the
            keys are column names that map to their replacement value. If
            passed as a scalar all columns are filled with that value.

        Returns
        -------
        Table
            Table expression

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
        │ NULL   │
        │ female │
        │ male   │
        │ female │
        │ male   │
        │ NULL   │
        │ NULL   │
        │ …      │
        └────────┘
        >>> t.fill_null({"sex": "unrecorded"}).sex
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
        """
        schema = self.schema()

        if isinstance(replacements, Mapping):
            for col, val in replacements.items():
                if col not in schema:
                    columns_formatted = ", ".join(map(repr, schema.names))
                    raise com.IbisTypeError(
                        f"Column {col!r} is not found in table. "
                        f"Existing columns: {columns_formatted}."
                    ) from None

                col_type = schema[col]
                val_type = val.type() if isinstance(val, Expr) else dt.infer(val)
                if not val_type.castable(col_type):
                    raise com.IbisTypeError(
                        f"Cannot fill_null on column {col!r} of type {col_type} with a "
                        f"value of type {val_type}"
                    )
        else:
            val_type = (
                replacements.type()
                if isinstance(replacements, Expr)
                else dt.infer(replacements)
            )
            for col, col_type in schema.items():
                if col_type.nullable and not val_type.castable(col_type):
                    raise com.IbisTypeError(
                        f"Cannot fill_null on column {col!r} of type {col_type} with a "
                        f"value of type {val_type} - pass in an explicit mapping "
                        f"of fill values to `fill_null` instead."
                    )
        return ops.FillNull(self, replacements).to_expr()

    @deprecated(as_of="9.1", instead="use drop_null instead")
    def dropna(
        self,
        subset: Sequence[str] | str | None = None,
        how: Literal["any", "all"] = "any",
    ) -> Table:
        """Deprecated - use `drop_null` instead."""

        return self.drop_null(subset, how)

    @deprecated(as_of="9.1", instead="use fill_null instead")
    def fillna(
        self,
        replacements: ir.Scalar | Mapping[str, ir.Scalar],
    ) -> Table:
        """Deprecated - use `fill_null` instead."""

        return self.fill_null(replacements)

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
        ...     nbytes = f.write(lines)  # nbytes is unused
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
        [`StructValue.lift`](./expression-collections.qmd#ibis.expr.types.structs.StructValue.lift)
        """
        columns_to_unpack = frozenset(columns)
        result_columns = []
        for column in self.columns:
            if column in columns_to_unpack:
                expr = self[column]
                result_columns.extend(expr[field] for field in expr.names)
            else:
                result_columns.append(column)
        return self.select(result_columns)

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
        >>> t = ibis.examples.penguins.fetch()
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
            agg = self.select(isna=ibis.cases((col.isnull(), 1), else_=0)).agg(
                name=lit(colname),
                type=lit(str(typ)),
                nullable=lit(typ.nullable),
                nulls=lambda t: t.isna.sum(),
                non_nulls=lambda t: (1 - t.isna).sum(),
                null_frac=lambda t: t.isna.mean(),
                pos=lit(pos, type=dt.int16),
            )
            aggs.append(agg)
        return ibis.union(*aggs).order_by(ibis.asc("pos"))

    def describe(
        self, quantile: Sequence[ir.NumericValue | float] = (0.25, 0.5, 0.75)
    ) -> Table:
        """Return summary information about a table.

        Parameters
        ----------
        quantile
            The quantiles to compute for numerical columns. Defaults to (0.25, 0.5, 0.75).

        Returns
        -------
        Table
            A table containing summary information about the columns of self.

        Notes
        -----
        This function computes summary statistics for each column in the table. For
        numerical columns, it computes statistics such as minimum, maximum, mean,
        standard deviation, and quantiles. For string columns, it computes the mode
        and the number of unique values.

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> ibis.options.interactive = True
        >>> p = ibis.examples.penguins.fetch()
        >>> p.describe()
        ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━┓
        ┃ name              ┃ pos   ┃ type    ┃ count ┃ nulls ┃ unique ┃ mode   ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━┩
        │ string            │ int16 │ string  │ int64 │ int64 │ int64  │ string │ … │
        ├───────────────────┼───────┼─────────┼───────┼───────┼────────┼────────┼───┤
        │ species           │     0 │ string  │   344 │     0 │      3 │ Adelie │ … │
        │ island            │     1 │ string  │   344 │     0 │      3 │ Biscoe │ … │
        │ bill_length_mm    │     2 │ float64 │   344 │     2 │    164 │ NULL   │ … │
        │ bill_depth_mm     │     3 │ float64 │   344 │     2 │     80 │ NULL   │ … │
        │ flipper_length_mm │     4 │ int64   │   344 │     2 │     55 │ NULL   │ … │
        │ body_mass_g       │     5 │ int64   │   344 │     2 │     94 │ NULL   │ … │
        │ sex               │     6 │ string  │   344 │    11 │      2 │ male   │ … │
        │ year              │     7 │ int64   │   344 │     0 │      3 │ NULL   │ … │
        └───────────────────┴───────┴─────────┴───────┴───────┴────────┴────────┴───┘
        >>> p.select(s.of_type("numeric")).describe()
        ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━┓
        ┃ name              ┃ pos   ┃ type    ┃ count ┃ nulls ┃ unique ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━┩
        │ string            │ int16 │ string  │ int64 │ int64 │ int64  │ … │
        ├───────────────────┼───────┼─────────┼───────┼───────┼────────┼───┤
        │ flipper_length_mm │     2 │ int64   │   344 │     2 │     55 │ … │
        │ body_mass_g       │     3 │ int64   │   344 │     2 │     94 │ … │
        │ year              │     4 │ int64   │   344 │     0 │      3 │ … │
        │ bill_length_mm    │     0 │ float64 │   344 │     2 │    164 │ … │
        │ bill_depth_mm     │     1 │ float64 │   344 │     2 │     80 │ … │
        └───────────────────┴───────┴─────────┴───────┴───────┴────────┴───┘
        >>> p.select(s.of_type("string")).describe()
        ┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ name    ┃ pos   ┃ type   ┃ count ┃ nulls ┃ unique ┃ mode   ┃
        ┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ string  │ int16 │ string │ int64 │ int64 │ int64  │ string │
        ├─────────┼───────┼────────┼───────┼───────┼────────┼────────┤
        │ sex     │     2 │ string │   344 │    11 │      2 │ male   │
        │ species │     0 │ string │   344 │     0 │      3 │ Adelie │
        │ island  │     1 │ string │   344 │     0 │      3 │ Biscoe │
        └─────────┴───────┴────────┴───────┴───────┴────────┴────────┘
        """
        import ibis.selectors as s
        from ibis import literal as lit

        quantile = sorted(quantile)
        aggs = []
        string_col = False
        numeric_col = False
        for pos, colname in enumerate(self.columns):
            col = self[colname]
            typ = col.type()

            # default statistics to None
            col_mean = lit(None).cast(float)
            col_std = lit(None).cast(float)
            col_min = lit(None).cast(float)
            col_max = lit(None).cast(float)
            col_mode = lit(None).cast(str)
            quantile_values = {
                f"p{100*q:.6f}".rstrip("0").rstrip("."): lit(None).cast(float)
                for q in quantile
            }

            if typ.is_numeric():
                numeric_col = True
                col_mean = col.mean()
                col_std = col.std()
                col_min = col.min().cast(float)
                col_max = col.max().cast(float)
                quantile_values = {
                    f"p{100*q:.6f}".rstrip("0").rstrip("."): col.quantile(q).cast(float)
                    for q in quantile
                }
            elif typ.is_string():
                string_col = True
                col_mode = col.mode()
            elif typ.is_boolean():
                numeric_col = True
                col_mean = col.mean()
            else:
                # Will not calculate statistics for other types
                continue

            agg = self.agg(
                name=lit(colname),
                pos=lit(pos, type=dt.int16),
                type=lit(str(typ)),
                count=col.isnull().count(),
                nulls=col.isnull().sum(),
                unique=col.nunique(),
                mode=col_mode,
                mean=col_mean,
                std=col_std,
                min=col_min,
                **quantile_values,
                max=col_max,
            )
            aggs.append(agg)

        names = aggs[0].schema().names
        new_schema = {
            name: dt.highest_precedence(types)
            for name, *types in zip(names, *(agg.schema().types for agg in aggs))
        }
        t = ibis.union(*(agg.cast(new_schema) for agg in aggs))

        # TODO(jiting): Need a better way to remove columns with all NULL
        if string_col and not numeric_col:
            t = t.select(~s.of_type("float"))
        elif numeric_col and not string_col:
            t = t.drop("mode")

        return t

    def join(
        left: Table,
        right: Table,
        predicates: (
            str
            | Sequence[
                str
                | ir.BooleanColumn
                | Literal[True]
                | Literal[False]
                | tuple[
                    str | ir.Column | ir.Deferred,
                    str | ir.Column | ir.Deferred,
                ]
            ]
        ) = (),
        how: JoinKind = "inner",
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ) -> Table:
        """Perform a join between two tables.

        Parameters
        ----------
        left
            Left table to join
        right
            Right table to join
        predicates
            Condition(s) to join on. See examples for details.
        how
            Join method, e.g. `"inner"` or `"left"`.
        lname
            A format string to use to rename overlapping columns in the left
            table (e.g. `"left_{name}"`).
        rname
            A format string to use to rename overlapping columns in the right
            table (e.g. `"right_{name}"`).

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> movies = ibis.examples.ml_latest_small_movies.fetch()
        >>> movies.head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ movieId ┃ title                            ┃ genres                          ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64   │ string                           │ string                          │
        ├─────────┼──────────────────────────────────┼─────────────────────────────────┤
        │       1 │ Toy Story (1995)                 │ Adventure|Animation|Children|C… │
        │       2 │ Jumanji (1995)                   │ Adventure|Children|Fantasy      │
        │       3 │ Grumpier Old Men (1995)          │ Comedy|Romance                  │
        │       4 │ Waiting to Exhale (1995)         │ Comedy|Drama|Romance            │
        │       5 │ Father of the Bride Part II (19… │ Comedy                          │
        └─────────┴──────────────────────────────────┴─────────────────────────────────┘
        >>> ratings = ibis.examples.ml_latest_small_ratings.fetch().drop("timestamp")
        >>> ratings.head()
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
        ┃ userId ┃ movieId ┃ rating  ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
        │ int64  │ int64   │ float64 │
        ├────────┼─────────┼─────────┤
        │      1 │       1 │     4.0 │
        │      1 │       3 │     4.0 │
        │      1 │       6 │     4.0 │
        │      1 │      47 │     5.0 │
        │      1 │      50 │     5.0 │
        └────────┴─────────┴─────────┘

        Equality left join on the shared `movieId` column.
        Note the `_right` suffix added to all overlapping
        columns from the right table
        (in this case only the "movieId" column).

        >>> ratings.join(movies, "movieId", how="left").head(5)
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━┓
        ┃ userId ┃ movieId ┃ rating  ┃ movieId_right ┃ title                       ┃ … ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━┩
        │ int64  │ int64   │ float64 │ int64         │ string                      │ … │
        ├────────┼─────────┼─────────┼───────────────┼─────────────────────────────┼───┤
        │      1 │       1 │     4.0 │             1 │ Toy Story (1995)            │ … │
        │      1 │       3 │     4.0 │             3 │ Grumpier Old Men (1995)     │ … │
        │      1 │       6 │     4.0 │             6 │ Heat (1995)                 │ … │
        │      1 │      47 │     5.0 │            47 │ Seven (a.k.a. Se7en) (1995) │ … │
        │      1 │      50 │     5.0 │            50 │ Usual Suspects, The (1995)  │ … │
        └────────┴─────────┴─────────┴───────────────┴─────────────────────────────┴───┘

        Explicit equality join using the default `how` value of `"inner"`.
        Note how there is no `_right` suffix added to the `movieId` column
        since this is an inner join and the `movieId` column is part of the
        join condition.

        >>> ratings.join(movies, ratings.movieId == movies.movieId).head(5)
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ userId ┃ movieId ┃ rating  ┃ title                  ┃ genres                 ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64  │ int64   │ float64 │ string                 │ string                 │
        ├────────┼─────────┼─────────┼────────────────────────┼────────────────────────┤
        │      1 │       1 │     4.0 │ Toy Story (1995)       │ Adventure|Animation|C… │
        │      1 │       3 │     4.0 │ Grumpier Old Men (199… │ Comedy|Romance         │
        │      1 │       6 │     4.0 │ Heat (1995)            │ Action|Crime|Thriller  │
        │      1 │      47 │     5.0 │ Seven (a.k.a. Se7en) … │ Mystery|Thriller       │
        │      1 │      50 │     5.0 │ Usual Suspects, The (… │ Crime|Mystery|Thriller │
        └────────┴─────────┴─────────┴────────────────────────┴────────────────────────┘

        >>> tags = ibis.examples.ml_latest_small_tags.fetch()
        >>> tags.head()
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
        ┃ userId ┃ movieId ┃ tag             ┃ timestamp  ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
        │ int64  │ int64   │ string          │ int64      │
        ├────────┼─────────┼─────────────────┼────────────┤
        │      2 │   60756 │ funny           │ 1445714994 │
        │      2 │   60756 │ Highly quotable │ 1445714996 │
        │      2 │   60756 │ will ferrell    │ 1445714992 │
        │      2 │   89774 │ Boxing story    │ 1445715207 │
        │      2 │   89774 │ MMA             │ 1445715200 │
        └────────┴─────────┴─────────────────┴────────────┘

        You can join on multiple columns/conditions by passing in a
        sequence. Find all instances where a user both tagged and
        rated a movie:

        >>> tags.join(ratings, ["userId", "movieId"]).head(5).order_by("userId")
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ userId ┃ movieId ┃ tag            ┃ timestamp  ┃ rating  ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━┩
        │ int64  │ int64   │ string         │ int64      │ float64 │
        ├────────┼─────────┼────────────────┼────────────┼─────────┤
        │     62 │       2 │ Robin Williams │ 1528843907 │     4.0 │
        │     62 │     110 │ sword fight    │ 1528152535 │     4.5 │
        │     62 │     410 │ gothic         │ 1525636609 │     4.5 │
        │     62 │    2023 │ mafia          │ 1525636733 │     5.0 │
        │     62 │    2124 │ quirky         │ 1525636846 │     5.0 │
        └────────┴─────────┴────────────────┴────────────┴─────────┘

        To self-join a table with itself, you need to call
        `.view()` on one of the arguments so the two tables
        are distinct from each other.

        For crafting more complex join conditions,
        a valid form of a join condition is a 2-tuple like
        `({left_key}, {right_key})`, where each key can be

        - a Column
        - Deferred expression
        - lambda of the form (Table) -> Column

        For example, to find all movies pairings that received the same
        (ignoring case) tags:

        >>> movie_tags = tags["movieId", "tag"]
        >>> view = movie_tags.view()
        >>> movie_tags.join(
        ...     view,
        ...     [
        ...         movie_tags.movieId != view.movieId,
        ...         (_.tag.lower(), lambda t: t.tag.lower()),
        ...     ],
        ... ).head().order_by(("movieId", "movieId_right"))
        ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ movieId ┃ tag               ┃ movieId_right ┃ tag_right         ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ int64   │ string            │ int64         │ string            │
        ├─────────┼───────────────────┼───────────────┼───────────────────┤
        │   60756 │ funny             │          1732 │ funny             │
        │   60756 │ Highly quotable   │          1732 │ Highly quotable   │
        │   89774 │ Tom Hardy         │        139385 │ tom hardy         │
        │  106782 │ drugs             │          1732 │ drugs             │
        │  106782 │ Leonardo DiCaprio │          5989 │ Leonardo DiCaprio │
        └─────────┴───────────────────┴───────────────┴───────────────────┘
        """
        from ibis.expr.types.joins import Join

        return Join(left.op()).join(
            right, predicates, how=how, lname=lname, rname=rname
        )

    def asof_join(
        left: Table,
        right: Table,
        on: str | ir.BooleanColumn,
        predicates: str | ir.Column | Sequence[str | ir.Column] = (),
        tolerance: str | ir.IntervalScalar | None = None,
        *,
        lname: str = "",
        rname: str = "{name}_right",
    ) -> Table:
        """Perform an "as-of" join between `left` and `right`.

        Similar to a left join except that the match is done on nearest key
        rather than equal keys.

        Parameters
        ----------
        left
            Table expression
        right
            Table expression
        on
            Closest match inequality condition
        predicates
            Additional join predicates
        tolerance
            Amount of time to look behind when joining
        lname
            A format string to use to rename overlapping columns in the left
            table (e.g. `"left_{name}"`).
        rname
            A format string to use to rename overlapping columns in the right
            table (e.g. `"right_{name}"`).

        Returns
        -------
        Table
            Table expression

        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> sensors = ibis.memtable(
        ...     {
        ...         "site": ["a", "b", "a", "b", "a"],
        ...         "humidity": [0.3, 0.4, 0.5, 0.6, 0.7],
        ...         "event_time": [
        ...             datetime(2024, 11, 16, 12, 0, 15, 500000),
        ...             datetime(2024, 11, 16, 12, 0, 15, 700000),
        ...             datetime(2024, 11, 17, 18, 12, 14, 950000),
        ...             datetime(2024, 11, 17, 18, 12, 15, 120000),
        ...             datetime(2024, 11, 18, 18, 12, 15, 100000),
        ...         ],
        ...     }
        ... )
        >>> events = ibis.memtable(
        ...     {
        ...         "site": ["a", "b", "a"],
        ...         "event_type": [
        ...             "cloud coverage",
        ...             "rain start",
        ...             "rain stop",
        ...         ],
        ...         "event_time": [
        ...             datetime(2024, 11, 16, 12, 0, 15, 400000),
        ...             datetime(2024, 11, 17, 18, 12, 15, 100000),
        ...             datetime(2024, 11, 18, 18, 12, 15, 100000),
        ...         ],
        ...     }
        ... )

        This setup simulates time-series data by pairing irregularly collected sensor
        readings with weather events, enabling analysis of environmental conditions
        before each event. We will use the `asof_join` method to match each event with
        the most recent prior sensor reading from the sensors table at the same site.

        >>> sensors
        ┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ site   ┃ humidity ┃ event_time              ┃
        ┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string │ float64  │ timestamp               │
        ├────────┼──────────┼─────────────────────────┤
        │ a      │      0.3 │ 2024-11-16 12:00:15.500 │
        │ b      │      0.4 │ 2024-11-16 12:00:15.700 │
        │ a      │      0.5 │ 2024-11-17 18:12:14.950 │
        │ b      │      0.6 │ 2024-11-17 18:12:15.120 │
        │ a      │      0.7 │ 2024-11-18 18:12:15.100 │
        └────────┴──────────┴─────────────────────────┘
        >>> events
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ site   ┃ event_type     ┃ event_time              ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string │ string         │ timestamp               │
        ├────────┼────────────────┼─────────────────────────┤
        │ a      │ cloud coverage │ 2024-11-16 12:00:15.400 │
        │ b      │ rain start     │ 2024-11-17 18:12:15.100 │
        │ a      │ rain stop      │ 2024-11-18 18:12:15.100 │
        └────────┴────────────────┴─────────────────────────┘

        We can find the closest event to each sensor reading with a 1 second tolerance.
        Using the "site" column as a join predicate ensures we only match events that
        occurred at or near the same site as the sensor reading.

        >>> tolerance = timedelta(seconds=1)
        >>> sensors.asof_join(events, on="event_time", predicates="site", tolerance=tolerance).drop(
        ...     "event_time_right"
        ... ).order_by("event_time")
        ┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
        ┃ site   ┃ humidity ┃ event_time              ┃ site_right ┃ event_type     ┃
        ┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
        │ string │ float64  │ timestamp               │ string     │ string         │
        ├────────┼──────────┼─────────────────────────┼────────────┼────────────────┤
        │ a      │      0.3 │ 2024-11-16 12:00:15.500 │ a          │ cloud coverage │
        │ b      │      0.4 │ 2024-11-16 12:00:15.700 │ NULL       │ NULL           │
        │ a      │      0.5 │ 2024-11-17 18:12:14.950 │ NULL       │ NULL           │
        │ b      │      0.6 │ 2024-11-17 18:12:15.120 │ b          │ rain start     │
        │ a      │      0.7 │ 2024-11-18 18:12:15.100 │ a          │ rain stop      │
        └────────┴──────────┴─────────────────────────┴────────────┴────────────────┘
        """
        from ibis.expr.types.joins import Join

        return Join(left.op()).asof_join(
            right, on, predicates, tolerance=tolerance, lname=lname, rname=rname
        )

    def cross_join(
        left: Table,
        right: Table,
        *rest: Table,
        lname: str = "",
        rname: str = "{name}_right",
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
        lname
            A format string to use to rename overlapping columns in the left
            table (e.g. `"left_{name}"`).
        rname
            A format string to use to rename overlapping columns in the right
            table (e.g. `"right_{name}"`).

        Returns
        -------
        Table
            Cross join of `left`, `right` and `rest`

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> t.count()
        ┌─────┐
        │ 344 │
        └─────┘
        >>> agg = t.drop("year").agg(s.across(s.numeric(), _.mean()))
        >>> expr = t.cross_join(agg)
        >>> expr
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
        >>> expr.columns
        ('species',
         'island',
         'bill_length_mm',
         'bill_depth_mm',
         'flipper_length_mm',
         'body_mass_g',
         'sex',
         'year',
         'bill_length_mm_right',
         'bill_depth_mm_right',
         'flipper_length_mm_right',
         'body_mass_g_right')
        >>> expr.count()
        ┌─────┐
        │ 344 │
        └─────┘
        """
        from ibis.expr.types.joins import Join

        return Join(left.op()).cross_join(right, *rest, lname=lname, rname=rname)

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
        [`Table.sql`](#ibis.expr.types.relations.Table.sql) method.

        ::: {.callout-note}
        ## `.alias` will create a temporary view

        `.alias` creates a temporary view in the database.

        This side effect will be removed in a future version of ibis and **is
        not part of the public API**.
        :::

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
        │ Adelie  │ Torgersen │           NULL │          NULL │              NULL │ … │
        │ Adelie  │ Torgersen │           36.7 │          19.3 │               193 │ … │
        └─────────┴───────────┴────────────────┴───────────────┴───────────────────┴───┘
        """
        return ops.View(child=self, name=alias).to_expr()

    def sql(self, query: str, dialect: str | None = None) -> ir.Table:
        '''Run a SQL query against a table expression.

        Parameters
        ----------
        query
            Query string
        dialect
            Optional string indicating the dialect of `query`. Defaults to the
            backend's native dialect.

        Returns
        -------
        Table
            An opaque table expression

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> expr = t.sql(
        ...     """
        ...     SELECT island, mean(bill_length_mm) AS avg_bill_length
        ...     FROM penguins
        ...     GROUP BY 1
        ...     ORDER BY 2 DESC
        ...     """
        ... )
        >>> expr
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ avg_bill_length ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
        │ string    │ float64         │
        ├───────────┼─────────────────┤
        │ Biscoe    │       45.257485 │
        │ Dream     │       44.167742 │
        │ Torgersen │       38.950980 │
        └───────────┴─────────────────┘

        Mix and match ibis expressions with SQL queries

        >>> t = ibis.examples.penguins.fetch(table_name="penguins")
        >>> expr = t.sql(
        ...     """
        ...     SELECT island, mean(bill_length_mm) AS avg_bill_length
        ...     FROM penguins
        ...     GROUP BY 1
        ...     ORDER BY 2 DESC
        ...     """
        ... )
        >>> expr = expr.mutate(
        ...     island=_.island.lower(),
        ...     avg_bill_length=_.avg_bill_length.round(1),
        ... )
        >>> expr
        ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        ┃ island    ┃ avg_bill_length ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
        │ string    │ float64         │
        ├───────────┼─────────────────┤
        │ biscoe    │            45.3 │
        │ dream     │            44.2 │
        │ torgersen │            39.0 │
        └───────────┴─────────────────┘

        Because ibis expressions aren't named, they aren't visible to
        subsequent `.sql` calls. Use the [`alias`](#ibis.expr.types.relations.Table.alias) method
        to assign a name to an expression.

        >>> expr.alias("b").sql("SELECT * FROM b WHERE avg_bill_length > 40")
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        ┃ island ┃ avg_bill_length ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
        │ string │ float64         │
        ├────────┼─────────────────┤
        │ biscoe │            45.3 │
        │ dream  │            44.2 │
        └────────┴─────────────────┘

        See Also
        --------
        [`Table.alias`](#ibis.expr.types.relations.Table.alias)
        '''
        op = self.op()
        backend = self._find_backend()

        if dialect is not None:
            # only transpile if dialect was passed
            query = backend._transpile_sql(query, dialect=dialect)

        if isinstance(op, ops.View):
            name = op.name
            expr = op.child.to_expr()
        else:
            name = util.gen_name("sql_query")
            expr = self

        schema = backend._get_sql_string_view_schema(name=name, table=expr, query=query)
        node = ops.SQLStringView(child=self.op(), query=query, schema=schema)
        return node.to_expr()

    def to_pandas(self, **kwargs) -> pd.DataFrame:
        """Convert a table expression to a pandas DataFrame.

        Parameters
        ----------
        kwargs
            Same as keyword arguments to [`execute`](./expression-generic.qmd#ibis.expr.types.core.Expr.execute)
        """
        return self.execute(**kwargs)

    def cache(self) -> Table:
        """Cache the provided expression.

        All subsequent operations on the returned expression will be performed
        on the cached data. The lifetime of the cached table is tied to its
        python references (ie. it is released once the last reference to it is
        garbage collected). Alternatively, use the
        [`with`](https://docs.python.org/3/reference/compound_stmts.html#with)
        statement or call the `.release()` method for more control.

        This method is idempotent: calling it multiple times in succession will
        return the same value as the first call.

        ::: {.callout-note}
        ## This method eagerly evaluates the expression prior to caching

        Subsequent evaluations will not recompute the expression so method
        chaining will not incur the overhead of caching more than once.
        :::

        Returns
        -------
        Table
            Cached table

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.examples.penguins.fetch()
        >>> heavy_computation = ibis.literal("Heavy Computation")
        >>> cached_penguins = t.mutate(computation=heavy_computation).cache()
        >>> cached_penguins
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

        Explicit cache cleanup

        >>> with t.mutate(computation=heavy_computation).cache() as cached_penguins:
        ...     cached_penguins
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
        """
        current_backend = self._find_backend(use_default=True)
        return current_backend._cached_table(self)

    def pivot_longer(
        self,
        col: str | s.Selector,
        *,
        names_to: str | Iterable[str] = "name",
        names_pattern: str | re.Pattern = r"(.+)",
        names_transform: (
            Callable[[str], ir.Value] | Mapping[str, Callable[[str], ir.Value]] | None
        ) = None,
        values_to: str = "value",
        values_transform: Callable[[ir.Value], ir.Value] | Deferred | None = None,
    ) -> Table:
        r"""Transform a table from wider to longer.

        Parameters
        ----------
        col
            String column name or selector.
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
        >>> import ibis.selectors as s
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

        >>> relig_income.pivot_longer(~s.cols("religion"), names_to="income", values_to="count")
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

        Similarly for a different example dataset, we convert names to values
        but using a different selector and the default `values_to` value.

        >>> world_bank_pop = ibis.examples.world_bank_pop_raw.fetch()
        >>> world_bank_pop.head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━┓
        ┃ country ┃ indicator   ┃ 2000         ┃ 2001         ┃ 2002         ┃ … ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━┩
        │ string  │ string      │ float64      │ float64      │ float64      │ … │
        ├─────────┼─────────────┼──────────────┼──────────────┼──────────────┼───┤
        │ ABW     │ SP.URB.TOTL │ 4.162500e+04 │ 4.202500e+04 │ 4.219400e+04 │ … │
        │ ABW     │ SP.URB.GROW │ 1.664222e+00 │ 9.563731e-01 │ 4.013352e-01 │ … │
        │ ABW     │ SP.POP.TOTL │ 8.910100e+04 │ 9.069100e+04 │ 9.178100e+04 │ … │
        │ ABW     │ SP.POP.GROW │ 2.539234e+00 │ 1.768757e+00 │ 1.194718e+00 │ … │
        │ AFE     │ SP.URB.TOTL │ 1.155517e+08 │ 1.197755e+08 │ 1.242275e+08 │ … │
        └─────────┴─────────────┴──────────────┴──────────────┴──────────────┴───┘
        >>> world_bank_pop.pivot_longer(s.matches(r"\d{4}"), names_to="year").head()
        ┏━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
        ┃ country ┃ indicator   ┃ year   ┃ value   ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
        │ string  │ string      │ string │ float64 │
        ├─────────┼─────────────┼────────┼─────────┤
        │ ABW     │ SP.URB.TOTL │ 2000   │ 41625.0 │
        │ ABW     │ SP.URB.TOTL │ 2001   │ 42025.0 │
        │ ABW     │ SP.URB.TOTL │ 2002   │ 42194.0 │
        │ ABW     │ SP.URB.TOTL │ 2003   │ 42277.0 │
        │ ABW     │ SP.URB.TOTL │ 2004   │ 42317.0 │
        └─────────┴─────────────┴────────┴─────────┘

        `pivot_longer` has some preprocessing capabilities like stripping a prefix and applying
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
        ... ).drop_null("rank")
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
        │ Afghanistan │ AF     │ AFG    │  1980 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1981 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1982 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1983 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1984 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1985 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1986 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1987 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1988 │        NULL │         NULL │ … │
        │ Afghanistan │ AF     │ AFG    │  1989 │        NULL │         NULL │ … │
        │ …           │ …      │ …      │     … │           … │            … │ … │
        └─────────────┴────────┴────────┴───────┴─────────────┴──────────────┴───┘
        >>> len(who.columns)
        60
        >>> who.pivot_longer(
        ...     s.index["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_(.)(.*)",
        ...     values_to="count",
        ... )
        ┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ country     ┃ iso2   ┃ iso3   ┃ year  ┃ diagnosis ┃ gender ┃ age    ┃ count ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ string      │ string │ string │ int64 │ string    │ string │ string │ int64 │
        ├─────────────┼────────┼────────┼───────┼───────────┼────────┼────────┼───────┤
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 014    │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 1524   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 2534   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 3544   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 4554   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 5564   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ m      │ 65     │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 014    │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 1524   │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │ f      │ 2534   │  NULL │
        │ …           │ …      │ …      │     … │ …         │ …      │ …      │     … │
        └─────────────┴────────┴────────┴───────┴───────────┴────────┴────────┴───────┘

        `names_transform` is flexible, and can be:

            1. A mapping of one or more names in `names_to` to callable
            2. A callable that will be applied to every name

        Let's recode gender and age to numeric values using a mapping

        >>> who.pivot_longer(
        ...     s.index["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_(.)(.*)",
        ...     names_transform=dict(
        ...         gender={"m": 1, "f": 2}.get,
        ...         age=dict(
        ...             zip(
        ...                 ["014", "1524", "2534", "3544", "4554", "5564", "65"],
        ...                 range(7),
        ...             )
        ...         ).get,
        ...     ),
        ...     values_to="count",
        ... )
        ┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━┓
        ┃ country     ┃ iso2   ┃ iso3   ┃ year  ┃ diagnosis ┃ gender ┃ age  ┃ count ┃
        ┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━┩
        │ string      │ string │ string │ int64 │ string    │ int8   │ int8 │ int64 │
        ├─────────────┼────────┼────────┼───────┼───────────┼────────┼──────┼───────┤
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    0 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    1 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    2 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    3 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    4 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    5 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      1 │    6 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    0 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    1 │  NULL │
        │ Afghanistan │ AF     │ AFG    │  1980 │ sp        │      2 │    2 │  NULL │
        │ …           │ …      │ …      │     … │ …         │      … │    … │     … │
        └─────────────┴────────┴────────┴───────┴───────────┴────────┴──────┴───────┘

        The number of match groups in `names_pattern` must match the length of `names_to`

        >>> who.pivot_longer(  # quartodoc: +EXPECTED_FAILURE
        ...     s.index["new_sp_m014":"newrel_f65"],
        ...     names_to=["diagnosis", "gender", "age"],
        ...     names_pattern="new_?(.*)_.(.*)",
        ... )
        Traceback (most recent call last):
          ...
        ibis.common.exceptions.IbisInputError: Number of match groups in `names_pattern` ...

        `names_transform` must be a mapping or callable

        >>> who.pivot_longer(
        ...     s.index["new_sp_m014":"newrel_f65"], names_transform="upper"
        ... )  # quartodoc: +EXPECTED_FAILURE
        Traceback (most recent call last):
          ...
        ibis.common.exceptions.IbisTypeError: ... Got <class 'str'>
        """  # noqa: RUF002
        import ibis.selectors as s

        pivot_sel = s._to_selector(col)

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

        pieces = []

        for pivot_col in pivot_cols:
            col_name = pivot_col.get_name()
            match_result = names_pattern.match(col_name)
            row = {
                name: names_transform[name](value)
                for name, value in zip(names_to, match_result.groups())
            }
            row[values_to] = values_transform(pivot_col)
            pieces.append(ibis.struct(row))

        # nest into an array of structs to zip unnests together
        pieces = ibis.array(pieces)

        return self.select(~pivot_sel, __pivoted__=pieces.unnest()).unpack(
            "__pivoted__"
        )

    @util.experimental
    def pivot_wider(
        self,
        *,
        id_cols: s.Selector | None = None,
        names_from: str | Iterable[str] | s.Selector = "name",
        names_prefix: str = "",
        names_sep: str = "_",
        names_sort: bool = False,
        names: Iterable[str] | None = None,
        values_from: str | Iterable[str] | s.Selector = "value",
        values_fill: int | float | str | ir.Scalar | None = None,
        values_agg: str | Callable[[ir.Value], ir.Scalar] | Deferred = "arbitrary",
    ) -> Table:
        """Pivot a table to a wider format.

        Parameters
        ----------
        id_cols
            A set of columns that uniquely identify each observation.
        names_from
            An argument describing which column or columns to use to get the
            name of the output columns.
        names_prefix
            String added to the start of every column name.
        names_sep
            If `names_from` or `values_from` contains multiple columns, this
            argument will be used to join their values together into a single
            string to use as a column name.
        names_sort
            If [](`True`) columns are sorted. If [](`False`) column names are
            ordered by appearance.
        names
            An explicit sequence of values to look for in columns matching
            `names_from`.

            * When this value is `None`, the values will be computed from
              `names_from`.
            * When this value is not `None`, each element's length must match
              the length of `names_from`.

            See examples below for more detail.
        values_from
            An argument describing which column or columns to get the cell
            values from.
        values_fill
            A scalar value that specifies what each value should be filled with
            when missing.
        values_agg
            A function applied to the value in each cell in the output.

        Returns
        -------
        Table
            Wider pivoted table

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True

        Basic usage

        >>> fish_encounters = ibis.examples.fish_encounters.fetch()
        >>> fish_encounters
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
        ┃ fish  ┃ station ┃ seen  ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
        │ int64 │ string  │ int64 │
        ├───────┼─────────┼───────┤
        │  4842 │ Release │     1 │
        │  4842 │ I80_1   │     1 │
        │  4842 │ Lisbon  │     1 │
        │  4842 │ Rstr    │     1 │
        │  4842 │ Base_TD │     1 │
        │  4842 │ BCE     │     1 │
        │  4842 │ BCW     │     1 │
        │  4842 │ BCE2    │     1 │
        │  4842 │ BCW2    │     1 │
        │  4842 │ MAE     │     1 │
        │     … │ …       │     … │
        └───────┴─────────┴───────┘
        >>> fish_encounters.pivot_wider(names_from="station", values_from="seen")  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━┓
        ┃ fish  ┃ Release ┃ I80_1 ┃ Lisbon ┃ Rstr  ┃ Base_TD ┃ BCE   ┃ BCW   ┃ … ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━┩
        │ int64 │ int64   │ int64 │ int64  │ int64 │ int64   │ int64 │ int64 │ … │
        ├───────┼─────────┼───────┼────────┼───────┼─────────┼───────┼───────┼───┤
        │  4842 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4843 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4844 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4845 │       1 │     1 │      1 │     1 │       1 │  NULL │  NULL │ … │
        │  4847 │       1 │     1 │      1 │  NULL │    NULL │  NULL │  NULL │ … │
        │  4848 │       1 │     1 │      1 │     1 │    NULL │  NULL │  NULL │ … │
        │  4849 │       1 │     1 │   NULL │  NULL │    NULL │  NULL │  NULL │ … │
        │  4850 │       1 │     1 │   NULL │     1 │       1 │     1 │     1 │ … │
        │  4851 │       1 │     1 │   NULL │  NULL │    NULL │  NULL │  NULL │ … │
        │  4854 │       1 │     1 │   NULL │  NULL │    NULL │  NULL │  NULL │ … │
        │     … │       … │     … │      … │     … │       … │     … │     … │ … │
        └───────┴─────────┴───────┴────────┴───────┴─────────┴───────┴───────┴───┘

        You can do simple transpose-like operations using `pivot_wider`

        >>> t = ibis.memtable(dict(outcome=["yes", "no"], counted=[3, 4]))
        >>> t
        ┏━━━━━━━━━┳━━━━━━━━━┓
        ┃ outcome ┃ counted ┃
        ┡━━━━━━━━━╇━━━━━━━━━┩
        │ string  │ int64   │
        ├─────────┼─────────┤
        │ yes     │       3 │
        │ no      │       4 │
        └─────────┴─────────┘
        >>> t.pivot_wider(names_from="outcome", values_from="counted", names_sort=True)
        ┏━━━━━━━┳━━━━━━━┓
        ┃ no    ┃ yes   ┃
        ┡━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │
        ├───────┼───────┤
        │     4 │     3 │
        └───────┴───────┘

        Fill missing pivoted values using `values_fill`

        >>> fish_encounters.pivot_wider(
        ...     names_from="station", values_from="seen", values_fill=0
        ... )  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━┓
        ┃ fish  ┃ Release ┃ I80_1 ┃ Lisbon ┃ Rstr  ┃ Base_TD ┃ BCE   ┃ BCW   ┃ … ┃
        ┡━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━┩
        │ int64 │ int64   │ int64 │ int64  │ int64 │ int64   │ int64 │ int64 │ … │
        ├───────┼─────────┼───────┼────────┼───────┼─────────┼───────┼───────┼───┤
        │  4842 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4843 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4844 │       1 │     1 │      1 │     1 │       1 │     1 │     1 │ … │
        │  4845 │       1 │     1 │      1 │     1 │       1 │     0 │     0 │ … │
        │  4847 │       1 │     1 │      1 │     0 │       0 │     0 │     0 │ … │
        │  4848 │       1 │     1 │      1 │     1 │       0 │     0 │     0 │ … │
        │  4849 │       1 │     1 │      0 │     0 │       0 │     0 │     0 │ … │
        │  4850 │       1 │     1 │      0 │     1 │       1 │     1 │     1 │ … │
        │  4851 │       1 │     1 │      0 │     0 │       0 │     0 │     0 │ … │
        │  4854 │       1 │     1 │      0 │     0 │       0 │     0 │     0 │ … │
        │     … │       … │     … │      … │     … │       … │     … │     … │ … │
        └───────┴─────────┴───────┴────────┴───────┴─────────┴───────┴───────┴───┘

        Compute multiple values columns

        >>> us_rent_income = ibis.examples.us_rent_income.fetch()
        >>> us_rent_income
        ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
        ┃ geoid  ┃ name       ┃ variable ┃ estimate ┃ moe   ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
        │ string │ string     │ string   │ int64    │ int64 │
        ├────────┼────────────┼──────────┼──────────┼───────┤
        │ 01     │ Alabama    │ income   │    24476 │   136 │
        │ 01     │ Alabama    │ rent     │      747 │     3 │
        │ 02     │ Alaska     │ income   │    32940 │   508 │
        │ 02     │ Alaska     │ rent     │     1200 │    13 │
        │ 04     │ Arizona    │ income   │    27517 │   148 │
        │ 04     │ Arizona    │ rent     │      972 │     4 │
        │ 05     │ Arkansas   │ income   │    23789 │   165 │
        │ 05     │ Arkansas   │ rent     │      709 │     5 │
        │ 06     │ California │ income   │    29454 │   109 │
        │ 06     │ California │ rent     │     1358 │     3 │
        │ …      │ …          │ …        │        … │     … │
        └────────┴────────────┴──────────┴──────────┴───────┘
        >>> us_rent_income.pivot_wider(
        ...     names_from="variable", values_from=["estimate", "moe"]
        ... )  # doctest: +SKIP
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━┓
        ┃ geoid  ┃ name                 ┃ estimate_income ┃ moe_income ┃ … ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━┩
        │ string │ string               │ int64           │ int64      │ … │
        ├────────┼──────────────────────┼─────────────────┼────────────┼───┤
        │ 01     │ Alabama              │           24476 │        136 │ … │
        │ 02     │ Alaska               │           32940 │        508 │ … │
        │ 04     │ Arizona              │           27517 │        148 │ … │
        │ 05     │ Arkansas             │           23789 │        165 │ … │
        │ 06     │ California           │           29454 │        109 │ … │
        │ 08     │ Colorado             │           32401 │        109 │ … │
        │ 09     │ Connecticut          │           35326 │        195 │ … │
        │ 10     │ Delaware             │           31560 │        247 │ … │
        │ 11     │ District of Columbia │           43198 │        681 │ … │
        │ 12     │ Florida              │           25952 │         70 │ … │
        │ …      │ …                    │               … │          … │ … │
        └────────┴──────────────────────┴─────────────────┴────────────┴───┘

        The column name separator can be changed using the `names_sep` parameter

        >>> us_rent_income.pivot_wider(
        ...     names_from="variable",
        ...     names_sep=".",
        ...     values_from=("estimate", "moe"),
        ... )  # doctest: +SKIP
        ┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━┓
        ┃ geoid  ┃ name                 ┃ estimate.income ┃ moe.income ┃ … ┃
        ┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━┩
        │ string │ string               │ int64           │ int64      │ … │
        ├────────┼──────────────────────┼─────────────────┼────────────┼───┤
        │ 01     │ Alabama              │           24476 │        136 │ … │
        │ 02     │ Alaska               │           32940 │        508 │ … │
        │ 04     │ Arizona              │           27517 │        148 │ … │
        │ 05     │ Arkansas             │           23789 │        165 │ … │
        │ 06     │ California           │           29454 │        109 │ … │
        │ 08     │ Colorado             │           32401 │        109 │ … │
        │ 09     │ Connecticut          │           35326 │        195 │ … │
        │ 10     │ Delaware             │           31560 │        247 │ … │
        │ 11     │ District of Columbia │           43198 │        681 │ … │
        │ 12     │ Florida              │           25952 │         70 │ … │
        │ …      │ …                    │               … │          … │ … │
        └────────┴──────────────────────┴─────────────────┴────────────┴───┘

        Supply an alternative function to summarize values

        >>> warpbreaks = ibis.examples.warpbreaks.fetch().select("wool", "tension", "breaks")
        >>> warpbreaks
        ┏━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
        ┃ wool   ┃ tension ┃ breaks ┃
        ┡━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
        │ string │ string  │ int64  │
        ├────────┼─────────┼────────┤
        │ A      │ L       │     26 │
        │ A      │ L       │     30 │
        │ A      │ L       │     54 │
        │ A      │ L       │     25 │
        │ A      │ L       │     70 │
        │ A      │ L       │     52 │
        │ A      │ L       │     51 │
        │ A      │ L       │     26 │
        │ A      │ L       │     67 │
        │ A      │ M       │     18 │
        │ …      │ …       │      … │
        └────────┴─────────┴────────┘
        >>> warpbreaks.pivot_wider(
        ...     names_from="wool", values_from="breaks", values_agg="mean"
        ... ).select("tension", "A", "B").order_by("tension")
        ┏━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
        ┃ tension ┃ A         ┃ B         ┃
        ┡━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
        │ string  │ float64   │ float64   │
        ├─────────┼───────────┼───────────┤
        │ H       │ 24.555556 │ 18.777778 │
        │ L       │ 44.555556 │ 28.222222 │
        │ M       │ 24.000000 │ 28.777778 │
        └─────────┴───────────┴───────────┘

        Passing `Deferred` objects to `values_agg` is supported

        >>> warpbreaks.pivot_wider(
        ...     names_from="tension",
        ...     values_from="breaks",
        ...     values_agg=_.sum(),
        ... ).select("wool", "H", "L", "M").order_by(s.all())
        ┏━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ wool   ┃ H     ┃ L     ┃ M     ┃
        ┡━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ string │ int64 │ int64 │ int64 │
        ├────────┼───────┼───────┼───────┤
        │ A      │   221 │   401 │   216 │
        │ B      │   169 │   254 │   259 │
        └────────┴───────┴───────┴───────┘

        Use a custom aggregate function

        >>> warpbreaks.pivot_wider(
        ...     names_from="wool",
        ...     values_from="breaks",
        ...     values_agg=lambda col: col.std() / col.mean(),
        ... ).select("tension", "A", "B").order_by("tension")
        ┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ tension ┃ A        ┃ B        ┃
        ┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
        │ string  │ float64  │ float64  │
        ├─────────┼──────────┼──────────┤
        │ H       │ 0.418344 │ 0.260590 │
        │ L       │ 0.406183 │ 0.349325 │
        │ M       │ 0.360844 │ 0.327719 │
        └─────────┴──────────┴──────────┘

        Generate some random data, setting the random seed for reproducibility

        >>> import random
        >>> random.seed(0)
        >>> raw = ibis.memtable(
        ...     [
        ...         dict(
        ...             product=product,
        ...             country=country,
        ...             year=year,
        ...             production=random.random(),
        ...         )
        ...         for product in "AB"
        ...         for country in ["AI", "EI"]
        ...         for year in range(2000, 2015)
        ...     ]
        ... )
        >>> production = raw.filter(((_.product == "A") & (_.country == "AI")) | (_.product == "B"))
        >>> production.order_by(s.all())
        ┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
        ┃ product ┃ country ┃ year  ┃ production ┃
        ┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
        │ string  │ string  │ int64 │ float64    │
        ├─────────┼─────────┼───────┼────────────┤
        │ A       │ AI      │  2000 │   0.844422 │
        │ A       │ AI      │  2001 │   0.757954 │
        │ A       │ AI      │  2002 │   0.420572 │
        │ A       │ AI      │  2003 │   0.258917 │
        │ A       │ AI      │  2004 │   0.511275 │
        │ A       │ AI      │  2005 │   0.404934 │
        │ A       │ AI      │  2006 │   0.783799 │
        │ A       │ AI      │  2007 │   0.303313 │
        │ A       │ AI      │  2008 │   0.476597 │
        │ A       │ AI      │  2009 │   0.583382 │
        │ …       │ …       │     … │          … │
        └─────────┴─────────┴───────┴────────────┘

        Pivoting with multiple name columns

        >>> production.pivot_wider(
        ...     names_from=["product", "country"],
        ...     values_from="production",
        ... )  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ year  ┃ B_AI     ┃ B_EI     ┃ A_AI     ┃
        ┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ float64  │ float64  │ float64  │
        ├───────┼──────────┼──────────┼──────────┤
        │  2000 │ 0.477010 │ 0.870471 │ 0.844422 │
        │  2001 │ 0.865310 │ 0.191067 │ 0.757954 │
        │  2002 │ 0.260492 │ 0.567511 │ 0.420572 │
        │  2003 │ 0.805028 │ 0.238616 │ 0.258917 │
        │  2004 │ 0.548699 │ 0.967540 │ 0.511275 │
        │  2005 │ 0.014042 │ 0.803179 │ 0.404934 │
        │  2006 │ 0.719705 │ 0.447970 │ 0.783799 │
        │  2007 │ 0.398824 │ 0.080446 │ 0.303313 │
        │  2008 │ 0.824845 │ 0.320055 │ 0.476597 │
        │  2009 │ 0.668153 │ 0.507941 │ 0.583382 │
        │     … │        … │        … │        … │
        └───────┴──────────┴──────────┴──────────┘

        Select a subset of names. This call incurs no computation when
        constructing the expression.

        >>> production.pivot_wider(
        ...     names_from=["product", "country"],
        ...     names=[("A", "AI"), ("B", "AI")],
        ...     values_from="production",
        ... )  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ year  ┃ A_AI     ┃ B_AI     ┃
        ┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ float64  │ float64  │
        ├───────┼──────────┼──────────┤
        │  2000 │ 0.844422 │ 0.477010 │
        │  2001 │ 0.757954 │ 0.865310 │
        │  2002 │ 0.420572 │ 0.260492 │
        │  2003 │ 0.258917 │ 0.805028 │
        │  2004 │ 0.511275 │ 0.548699 │
        │  2005 │ 0.404934 │ 0.014042 │
        │  2006 │ 0.783799 │ 0.719705 │
        │  2007 │ 0.303313 │ 0.398824 │
        │  2008 │ 0.476597 │ 0.824845 │
        │  2009 │ 0.583382 │ 0.668153 │
        │     … │        … │        … │
        └───────┴──────────┴──────────┘

        Sort the new columns' names

        >>> production.pivot_wider(
        ...     names_from=["product", "country"],
        ...     values_from="production",
        ...     names_sort=True,
        ... )  # doctest: +SKIP
        ┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
        ┃ year  ┃ A_AI     ┃ B_AI     ┃ B_EI     ┃
        ┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
        │ int64 │ float64  │ float64  │ float64  │
        ├───────┼──────────┼──────────┼──────────┤
        │  2000 │ 0.844422 │ 0.477010 │ 0.870471 │
        │  2001 │ 0.757954 │ 0.865310 │ 0.191067 │
        │  2002 │ 0.420572 │ 0.260492 │ 0.567511 │
        │  2003 │ 0.258917 │ 0.805028 │ 0.238616 │
        │  2004 │ 0.511275 │ 0.548699 │ 0.967540 │
        │  2005 │ 0.404934 │ 0.014042 │ 0.803179 │
        │  2006 │ 0.783799 │ 0.719705 │ 0.447970 │
        │  2007 │ 0.303313 │ 0.398824 │ 0.080446 │
        │  2008 │ 0.476597 │ 0.824845 │ 0.320055 │
        │  2009 │ 0.583382 │ 0.668153 │ 0.507941 │
        │     … │        … │        … │        … │
        └───────┴──────────┴──────────┴──────────┘
        """
        import ibis.selectors as s
        from ibis.expr.rewrites import _, p, x

        orig_names_from = util.promote_list(names_from)

        names_from = s._to_selector(orig_names_from)
        values_from = s._to_selector(values_from)

        if id_cols is None:
            id_cols = ~(names_from | values_from)
        else:
            id_cols = s._to_selector(id_cols)

        if isinstance(values_agg, str):
            values_agg = operator.methodcaller(values_agg)
        elif isinstance(values_agg, Deferred):
            values_agg = values_agg.resolve

        if names is None:
            # no names provided, compute them from the data
            names = self.select(names_from).distinct().execute()
            columns = names.columns.tolist()
            names = list(names.itertuples(index=False))
        else:
            if not (columns := [col.get_name() for col in names_from.expand(self)]):
                raise com.IbisInputError(
                    f"No matching names columns in `names_from`: {orig_names_from}"
                )
            names = list(map(tuple, map(util.promote_list, names)))

        if names_sort:
            names.sort()

        values_cols = values_from.expand(self)
        more_than_one_value = len(values_cols) > 1
        aggs = {}

        names_cols_exprs = [self[col] for col in columns]

        for keys in names:
            where = ibis.and_(*map(operator.eq, names_cols_exprs, keys))

            for values_col in values_cols:
                arg = values_agg(values_col)

                # this allows users to write the aggregate without having to deal with
                # the filter themselves
                rules = (
                    # add in the where clause to filter the appropriate values
                    p.Reduction(where=None) >> _.copy(where=where)
                    | p.Reduction(where=x) >> _.copy(where=where & x)
                )
                arg = arg.op().replace(rules, filter=p.Value).to_expr()

                # build the components of the group by key
                key_components = (
                    # user provided prefix
                    names_prefix,
                    # include the `values` column name if there's more than one
                    # `values` column
                    values_col.get_name() * more_than_one_value,
                    # values computed from `names`/`names_from`
                    *keys,
                )
                key = names_sep.join(filter(None, key_components))
                aggs[key] = arg if values_fill is None else arg.coalesce(values_fill)

        grouping_keys = id_cols.expand(self)

        # no id columns, so do an ungrouped aggregation
        if not grouping_keys:
            return self.aggregate(**aggs)

        return self.group_by(*grouping_keys).aggregate(**aggs)

    def relocate(
        self,
        *columns: str | s.Selector,
        before: str | s.Selector | None = None,
        after: str | s.Selector | None = None,
        **kwargs: str,
    ) -> Table:
        """Relocate `columns` before or after other specified columns.

        Parameters
        ----------
        columns
            Columns to relocate. Selectors are accepted.
        before
            A column name or selector to insert the new columns before.
        after
            A column name or selector. Columns in `columns` are relocated after the last
            column selected in `after`.
        kwargs
            Additional column names to relocate, renaming argument values to
            keyword argument names.

        Returns
        -------
        Table
            A table with the columns relocated.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> import ibis.selectors as s
        >>> t = ibis.memtable(dict(a=[1], b=[1], c=[1], d=["a"], e=["a"], f=["a"]))
        >>> t
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ b     ┃ c     ┃ d      ┃ e      ┃ f      ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ int64 │ int64 │ string │ string │ string │
        ├───────┼───────┼───────┼────────┼────────┼────────┤
        │     1 │     1 │     1 │ a      │ a      │ a      │
        └───────┴───────┴───────┴────────┴────────┴────────┘
        >>> t.relocate("f")
        ┏━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ f      ┃ a     ┃ b     ┃ c     ┃ d      ┃ e      ┃
        ┡━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ string │ int64 │ int64 │ int64 │ string │ string │
        ├────────┼───────┼───────┼───────┼────────┼────────┤
        │ a      │     1 │     1 │     1 │ a      │ a      │
        └────────┴───────┴───────┴───────┴────────┴────────┘
        >>> t.relocate("a", after="c")
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ b     ┃ c     ┃ a     ┃ d      ┃ e      ┃ f      ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ int64 │ int64 │ string │ string │ string │
        ├───────┼───────┼───────┼────────┼────────┼────────┤
        │     1 │     1 │     1 │ a      │ a      │ a      │
        └───────┴───────┴───────┴────────┴────────┴────────┘
        >>> t.relocate("f", before="b")
        ┏━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ f      ┃ b     ┃ c     ┃ d      ┃ e      ┃
        ┡━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ string │ int64 │ int64 │ string │ string │
        ├───────┼────────┼───────┼───────┼────────┼────────┤
        │     1 │ a      │     1 │     1 │ a      │ a      │
        └───────┴────────┴───────┴───────┴────────┴────────┘
        >>> t.relocate("a", after=s.last())
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ b     ┃ c     ┃ d      ┃ e      ┃ f      ┃ a     ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │ string │ string │ string │ int64 │
        ├───────┼───────┼────────┼────────┼────────┼───────┤
        │     1 │     1 │ a      │ a      │ a      │     1 │
        └───────┴───────┴────────┴────────┴────────┴───────┘

        Relocate allows renaming

        >>> t.relocate(ff="f")
        ┏━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ ff     ┃ a     ┃ b     ┃ c     ┃ d      ┃ e      ┃
        ┡━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ string │ int64 │ int64 │ int64 │ string │ string │
        ├────────┼───────┼───────┼───────┼────────┼────────┤
        │ a      │     1 │     1 │     1 │ a      │ a      │
        └────────┴───────┴───────┴───────┴────────┴────────┘

        You can relocate based on any predicate selector, such as
        [`of_type`](./selectors.qmd#ibis.selectors.of_type)

        >>> t.relocate(s.of_type("string"))
        ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ d      ┃ e      ┃ f      ┃ a     ┃ b     ┃ c     ┃
        ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ string │ string │ string │ int64 │ int64 │ int64 │
        ├────────┼────────┼────────┼───────┼───────┼───────┤
        │ a      │ a      │ a      │     1 │     1 │     1 │
        └────────┴────────┴────────┴───────┴───────┴───────┘
        >>> t.relocate(s.numeric(), after=s.last())
        ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ d      ┃ e      ┃ f      ┃ a     ┃ b     ┃ c     ┃
        ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ string │ string │ string │ int64 │ int64 │ int64 │
        ├────────┼────────┼────────┼───────┼───────┼───────┤
        │ a      │ a      │ a      │     1 │     1 │     1 │
        └────────┴────────┴────────┴───────┴───────┴───────┘

        When multiple columns are selected with `before` or `after`, those
        selected columns are moved before and after the `selectors` input

        >>> t = ibis.memtable(dict(a=[1], b=["a"], c=[1], d=["a"]))
        >>> t.relocate(s.numeric(), after=s.of_type("string"))
        ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ b      ┃ d      ┃ a     ┃ c     ┃
        ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ string │ string │ int64 │ int64 │
        ├────────┼────────┼───────┼───────┤
        │ a      │ a      │     1 │     1 │
        └────────┴────────┴───────┴───────┘
        >>> t.relocate(s.numeric(), before=s.of_type("string"))
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
        ┃ a     ┃ c     ┃ b      ┃ d      ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
        │ int64 │ int64 │ string │ string │
        ├───────┼───────┼────────┼────────┤
        │     1 │     1 │ a      │ a      │
        └───────┴───────┴────────┴────────┘

        When there are duplicate **renames** in a call to relocate, the
        last one is preserved

        >>> t.relocate(e="d", f="d")
        ┏━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃ f      ┃ a     ┃ b      ┃ c     ┃
        ┡━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ string │ int64 │ string │ int64 │
        ├────────┼───────┼────────┼───────┤
        │ a      │     1 │ a      │     1 │
        └────────┴───────┴────────┴───────┘

        However, if there are duplicates that are **not** part of a rename, the
        order specified in the relocate call is preserved

        >>> t.relocate(
        ...     "b",
        ...     s.of_type("string"),  # "b" is a string column, so the selector matches
        ... )
        ┏━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ b      ┃ d      ┃ a     ┃ c     ┃
        ┡━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ string │ string │ int64 │ int64 │
        ├────────┼────────┼───────┼───────┤
        │ a      │ a      │     1 │     1 │
        └────────┴────────┴───────┴───────┘
        """
        if not columns and before is None and after is None and not kwargs:
            raise com.IbisInputError(
                "At least one selector or `before` or `after` must be provided"
            )

        if before is not None and after is not None:
            raise com.IbisInputError("Cannot specify both `before` and `after`")

        sels = {}

        schema = self.schema()
        positions = schema._name_locs

        for new_name, expr in itertools.zip_longest(
            kwargs.keys(), self._fast_bind(*kwargs.values(), *columns)
        ):
            expr_name = expr.get_name()
            pos = positions[expr_name]
            renamed = new_name is not None
            if renamed and pos in sels:
                # **only when renaming**: make sure the last duplicate
                # column wins by reinserting the position if it already
                # exists
                #
                # to do that, we first delete the existing one, which causes
                # the subsequent insertion to be at the end
                del sels[pos]
            sels[pos] = new_name if renamed else expr_name

        ncols = len(schema)

        if before is not None:
            where = min(
                (positions[expr.get_name()] for expr in self._fast_bind(before)),
                default=0,
            )
        elif after is not None:
            where = (
                max(
                    (positions[expr.get_name()] for expr in self._fast_bind(after)),
                    default=ncols - 1,
                )
                + 1
            )
        else:
            assert before is None and after is None
            where = 0

        columns = schema.names

        fields = self.op().fields

        # all columns that should come BEFORE the matched selectors
        exprs = {
            name: fields[name]
            for name in (columns[left] for left in range(where) if left not in sels)
        }

        # selected columns
        exprs.update((name, fields[columns[i]]) for i, name in sels.items())

        # all columns that should come AFTER the matched selectors
        exprs.update(
            (name, fields[name])
            for name in (
                columns[right] for right in range(where, ncols) if right not in sels
            )
        )

        return ops.Project(self, exprs).to_expr()

    def window_by(
        self,
        time_col: str | ir.Value,
    ) -> WindowedTable:
        from ibis.expr.types.temporal_windows import WindowedTable

        time_col = next(iter(self.bind(time_col)))

        # validate time_col is a timestamp column
        if not isinstance(time_col, TimestampColumn):
            raise com.IbisInputError(
                f"`time_col` must be a timestamp column, got {time_col.type()}"
            )

        return WindowedTable(self, time_col)

    def value_counts(self, *, name: str | None = None) -> ir.Table:
        """Compute a frequency table of this table's values.

        ::: {.callout-note title="Changed in version 10.0.0"}
        Added `name` parameter.
        :::

        Parameters
        ----------
        name
            The name to use for the frequency column. A suitable name will be
            automatically generated if not provided.

        Returns
        -------
        Table
            Frequency table of this table's values.

        Examples
        --------
        >>> from ibis import examples
        >>> ibis.options.interactive = True
        >>> t = examples.penguins.fetch()
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
        >>> t.year.value_counts(name="n").order_by("year")
        ┏━━━━━━━┳━━━━━━━┓
        ┃ year  ┃ n     ┃
        ┡━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │
        ├───────┼───────┤
        │  2007 │   110 │
        │  2008 │   114 │
        │  2009 │   120 │
        └───────┴───────┘
        >>> t[["year", "island"]].value_counts().order_by("year", "island")
        ┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
        ┃ year  ┃ island    ┃ year_island_count ┃
        ┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
        │ int64 │ string    │ int64             │
        ├───────┼───────────┼───────────────────┤
        │  2007 │ Biscoe    │                44 │
        │  2007 │ Dream     │                46 │
        │  2007 │ Torgersen │                20 │
        │  2008 │ Biscoe    │                64 │
        │  2008 │ Dream     │                34 │
        │  2008 │ Torgersen │                16 │
        │  2009 │ Biscoe    │                60 │
        │  2009 │ Dream     │                44 │
        │  2009 │ Torgersen │                16 │
        └───────┴───────────┴───────────────────┘
        """
        columns = self.columns
        if name is None:
            name = "_".join(columns) + "_count"
        return self.group_by(columns).agg(lambda t: t.count().name(name))

    def unnest(
        self, column, offset: str | None = None, keep_empty: bool = False
    ) -> Table:
        """Unnest an array `column` from a table.

        When unnesting an existing column the newly unnested column replaces
        the existing column.

        Parameters
        ----------
        column
            Array column to unnest.
        offset
            Name of the resulting index column.
        keep_empty
            Keep empty array values as `NULL` in the output table, as well as
            existing `NULL` values.

        Returns
        -------
        Table
            Table with the array column `column` unnested.

        See Also
        --------
        [`ArrayValue.unnest`](./expression-collections.qmd#ibis.expr.types.arrays.ArrayValue.unnest)

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True

        Construct a table expression with an array column.

        >>> t = ibis.memtable({"x": [[1, 2], [], None, [3, 4, 5]], "y": [1, 2, 3, 4]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ x                    ┃ y     ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ array<int64>         │ int64 │
        ├──────────────────────┼───────┤
        │ [1, 2]               │     1 │
        │ []                   │     2 │
        │ NULL                 │     3 │
        │ [3, 4, ... +1]       │     4 │
        └──────────────────────┴───────┘

        Unnest the array column `x`, replacing the **existing** `x` column.

        >>> t.unnest("x")
        ┏━━━━━━━┳━━━━━━━┓
        ┃ x     ┃ y     ┃
        ┡━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │
        ├───────┼───────┤
        │     1 │     1 │
        │     2 │     1 │
        │     3 │     4 │
        │     4 │     4 │
        │     5 │     4 │
        └───────┴───────┘

        Unnest the array column `x` with an offset. The `offset` parameter is
        the name of the resulting index column.

        >>> t.unnest(t.x, offset="idx")
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ x     ┃ y     ┃ idx   ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │ int64 │
        ├───────┼───────┼───────┤
        │     1 │     1 │     0 │
        │     2 │     1 │     1 │
        │     3 │     4 │     0 │
        │     4 │     4 │     1 │
        │     5 │     4 │     2 │
        └───────┴───────┴───────┘

        Unnest the array column `x` keep empty array values as `NULL` in the
        output table.

        >>> t.unnest(_.x, offset="idx", keep_empty=True)
        ┏━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ x     ┃ y     ┃ idx   ┃
        ┡━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ int64 │ int64 │ int64 │
        ├───────┼───────┼───────┤
        │     1 │     1 │     0 │
        │     2 │     1 │     1 │
        │     3 │     4 │     0 │
        │     4 │     4 │     1 │
        │     5 │     4 │     2 │
        │  NULL │     2 │  NULL │
        │  NULL │     3 │  NULL │
        └───────┴───────┴───────┘

        If you need to preserve the row order of the preserved empty arrays or
        null values use
        [`row_number`](./expression-tables.qmd#ibis.row_number) to
        create an index column before calling `unnest`.

        >>> (
        ...     t.mutate(original_row=ibis.row_number())
        ...     .unnest("x", offset="idx", keep_empty=True)
        ...     .relocate("original_row")
        ...     .order_by("original_row")
        ... )
        ┏━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
        ┃ original_row ┃ x     ┃ y     ┃ idx   ┃
        ┡━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
        │ int64        │ int64 │ int64 │ int64 │
        ├──────────────┼───────┼───────┼───────┤
        │            0 │     1 │     1 │     0 │
        │            0 │     2 │     1 │     1 │
        │            1 │  NULL │     2 │  NULL │
        │            2 │  NULL │     3 │  NULL │
        │            3 │     3 │     4 │     0 │
        │            3 │     4 │     4 │     1 │
        │            3 │     5 │     4 │     2 │
        └──────────────┴───────┴───────┴───────┘

        You can also unnest more complex expressions, and the resulting column
        will be projected as the last expression in the result.

        >>> t.unnest(_.x.map(lambda v: v + 1).name("plus_one"))
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┓
        ┃ x                    ┃ y     ┃ plus_one ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━┩
        │ array<int64>         │ int64 │ int64    │
        ├──────────────────────┼───────┼──────────┤
        │ [1, 2]               │     1 │        2 │
        │ [1, 2]               │     1 │        3 │
        │ [3, 4, ... +1]       │     4 │        4 │
        │ [3, 4, ... +1]       │     4 │        5 │
        │ [3, 4, ... +1]       │     4 │        6 │
        └──────────────────────┴───────┴──────────┘
        """
        (column,) = self.bind(column)
        return ops.TableUnnest(
            parent=self,
            column=column,
            column_name=column.get_name(),
            offset=offset,
            keep_empty=keep_empty,
        ).to_expr()


@public
class CachedTable(Table):
    def __exit__(self, *_) -> None:
        self.release()

    def __enter__(self) -> CachedTable:
        return self

    def release(self) -> None:
        """Release the underlying expression from the cache."""
        current_backend = self._find_backend(use_default=True)
        return current_backend._finalize_cached_table(self.op().name)


public(Table=Table, CachedTable=CachedTable)
