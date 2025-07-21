from __future__ import annotations

import contextlib
import os
import webbrowser
from typing import TYPE_CHECKING, Any, NoReturn

from public import public

import ibis
import ibis.expr.operations as ops
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import IbisError, TranslationError
from ibis.common.grounds import Immutable
from ibis.common.patterns import Coercible, CoercionError
from ibis.common.typing import get_defining_scope
from ibis.config import _default_backend
from ibis.config import options as opts
from ibis.expr.format import pretty
from ibis.expr.types.rich import capture_rich_renderable, to_rich
from ibis.util import experimental

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import torch
    from rich.console import Console

    import ibis.expr.types as ir
    from ibis.backends import BaseBackend
    from ibis.expr.sql import SQLString
    from ibis.expr.visualize import EdgeAttributeGetter, NodeAttributeGetter


@public
class Expr(Immutable, Coercible):
    """Base expression class."""

    __slots__ = ("_arg",)
    _arg: ops.Node

    def _noninteractive_repr(self) -> str:
        if ibis.options.repr.show_variables:
            scope = get_defining_scope(self, types=Expr)
        else:
            scope = None
        return pretty(self.op(), scope=scope)

    def __repr__(self) -> str:
        if ibis.options.interactive:
            return capture_rich_renderable(self)
        else:
            return self._noninteractive_repr()

    def __rich_console__(self, console: Console, options):
        from rich.text import Text

        if console.is_jupyter:
            # Rich infers a console width in jupyter notebooks, but since
            # notebooks can use horizontal scroll bars we don't want to apply a
            # limit here. Since rich requires an integer for max_width, we
            # choose an arbitrarily large integer bound. Note that we need to
            # handle this here rather than in `to_rich`, as this setting
            # also needs to be forwarded to `console.render`.
            options = options.update(max_width=1_000_000)
            console_width = None
        else:
            console_width = options.max_width

        try:
            if opts.interactive:
                rich_object = to_rich(self, console_width=console_width)
            else:
                rich_object = Text(self._noninteractive_repr())
        except TranslationError as e:
            lines = [
                "Translation to backend failed",
                f"Error message: {e!r}",
                "Expression repr follows:",
                self._noninteractive_repr(),
            ]
            return Text("\n".join(lines))
        return console.render(rich_object, options=options)

    def __init__(self, arg: ops.Node) -> None:
        object.__setattr__(self, "_arg", arg)

    def __iter__(self) -> NoReturn:
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, ops.Node):
            return value.to_expr()
        else:
            raise CoercionError("Unable to coerce value to an expression")

    def __reduce__(self):
        return (self.__class__, (self._arg,))

    def __hash__(self):
        return hash((self.__class__, self._arg))

    def equals(self, other, /) -> bool:
        """Return whether this expression is _structurally_ equivalent to `other`.

        If you want to produce an equality expression, use `==` syntax.

        Parameters
        ----------
        other
            Another expression

        Examples
        --------
        >>> import ibis
        >>> t1 = ibis.table(dict(a="int"), name="t")
        >>> t2 = ibis.table(dict(a="int"), name="t")
        >>> t1.equals(t2)
        True
        >>> v = ibis.table(dict(a="string"), name="v")
        >>> t1.equals(v)
        False
        """
        if not isinstance(other, Expr):
            raise TypeError(
                f"invalid equality comparison between Expr and {type(other)}"
            )
        return self._arg.equals(other._arg)

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an Ibis expression is not defined")

    __nonzero__ = __bool__

    def get_name(self):
        """Return the name of this expression."""
        return self._arg.name

    def _repr_png_(self) -> bytes | None:
        if opts.interactive or not opts.graphviz_repr:
            return None
        try:
            import ibis.expr.visualize as viz
        except ImportError:
            return None
        else:
            # Something may go wrong, and we can't error in the notebook
            # so fallback to the default text representation.
            with contextlib.suppress(Exception):
                return viz.to_graph(self).pipe(format="png")

    def visualize(
        self,
        format: str = "svg",
        *,
        label_edges: bool = False,
        verbose: bool = False,
        node_attr: Mapping[str, str] | None = None,
        node_attr_getter: NodeAttributeGetter | None = None,
        edge_attr: Mapping[str, str] | None = None,
        edge_attr_getter: EdgeAttributeGetter | None = None,
    ) -> None:
        """Visualize an expression as a GraphViz graph in the browser.

        Parameters
        ----------
        format
            Image output format. These are specified by the `graphviz` Python
            library.
        label_edges
            Show operation input names as edge labels
        verbose
            Print the graphviz DOT code to stderr if [](`True`)
        node_attr
            Mapping of `(attribute, value)` pairs set for all nodes.
            Options are specified by the `graphviz` Python library.
        node_attr_getter
            Callback taking a node and returning a mapping of `(attribute, value)` pairs
            for that node. Options are specified by the `graphviz` Python library.
        edge_attr
            Mapping of `(attribute, value)` pairs set for all edges.
            Options are specified by the `graphviz` Python library.
        edge_attr_getter
            Callback taking two adjacent nodes and returning a mapping of `(attribute, value)` pairs
            for the edge between those nodes. Options are specified by the `graphviz` Python library.

        Examples
        --------
        Open the visualization of an expression in default browser:

        >>> import ibis
        >>> import ibis.expr.operations as ops
        >>> left = ibis.table(dict(a="int64", b="string"), name="left")
        >>> right = ibis.table(dict(b="string", c="int64", d="string"), name="right")
        >>> expr = left.inner_join(right, "b").select(left.a, b=right.c, c=right.d)
        >>> expr.visualize(
        ...     format="svg",
        ...     label_edges=True,
        ...     node_attr={"fontname": "Roboto Mono", "fontsize": "10"},
        ...     node_attr_getter=lambda node: isinstance(node, ops.Field) and {"shape": "oval"},
        ...     edge_attr={"fontsize": "8"},
        ...     edge_attr_getter=lambda u, v: isinstance(u, ops.Field) and {"color": "red"},
        ... )  # quartodoc: +SKIP # doctest: +SKIP

        Raises
        ------
        ImportError
            If `graphviz` is not installed.
        """
        import ibis.expr.visualize as viz

        path = viz.draw(
            viz.to_graph(
                self,
                node_attr=node_attr,
                node_attr_getter=node_attr_getter,
                edge_attr=edge_attr,
                edge_attr_getter=edge_attr_getter,
                label_edges=label_edges,
            ),
            format=format,
            verbose=verbose,
        )
        webbrowser.open(f"file://{os.path.abspath(path)}")

    def pipe(self, f, /, *args: Any, **kwargs: Any) -> Expr:
        """Compose `f` with `self`.

        Parameters
        ----------
        f
            If the expression needs to be passed as anything other than the
            first argument to the function, pass a tuple with the argument
            name. For example, (f, 'data') if the function f expects a 'data'
            keyword
        args
            Positional arguments to `f`
        kwargs
            Keyword arguments to `f`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.memtable(
        ...     {
        ...         "a": [5, 10, 15],
        ...         "b": ["a", "b", "c"],
        ...     }
        ... )
        >>> f = lambda a: (a + 1).name("a")
        >>> g = lambda a: (a * 2).name("a")
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1
        r0 := InMemoryTable
        data:
            PandasDataFrameProxy:
                a  b
            0   5  a
            1  10  b
            2  15  c
        a: r0.a + 1 * 2

        >>> result2 = g(f(t.a))  # equivalent to the above
        >>> result1.equals(result2)
        True

        Returns
        -------
        Expr
            Result type of passed function
        """
        if isinstance(f, tuple):
            f, data_keyword = f
            kwargs = kwargs.copy()
            kwargs[data_keyword] = self
            return f(*args, **kwargs)
        else:
            return f(self, *args, **kwargs)

    def op(self) -> ops.Node:
        return self._arg

    def _find_backends(self) -> tuple[list[BaseBackend], bool]:
        """Return the possible backends for an expression.

        Returns
        -------
        list[BaseBackend]
            A list of the backends found.
        """

        backends = set()
        has_unbound = False
        node_types = (ops.UnboundTable, ops.DatabaseTable, ops.SQLQueryResult)
        for table in self.op().find(node_types):
            if isinstance(table, ops.UnboundTable):
                has_unbound = True
            else:
                backends.add(table.source)

        return list(backends), has_unbound

    def _find_backend(self, *, use_default: bool = False) -> BaseBackend:
        """Find the backend attached to an expression.

        Parameters
        ----------
        use_default
            If [](`True`) and the default backend isn't set, initialize the
            default backend and use that. This should only be set to `True` for
            `.execute()`. For other contexts such as compilation, this option
            doesn't make sense so the default value is [](`False`).

        Returns
        -------
        BaseBackend
            A backend that is attached to the expression
        """
        backends, has_unbound = self._find_backends()

        if not backends:
            if has_unbound:
                raise IbisError(
                    "Expression contains unbound tables and therefore cannot "
                    "be executed. Use `<backend>.execute(expr)` to execute "
                    "against an explicit backend, or rebuild the expression "
                    "using bound tables instead."
                )
            default = _default_backend() if use_default else None
            if default is None:
                raise IbisError(
                    "Expression depends on no backends, and found no default"
                )
            return default

        if len(backends) > 1:
            raise IbisError("Multiple backends found for this expression")

        return backends[0]

    def get_backend(self) -> BaseBackend:
        """Get the current Ibis backend of the expression.

        Returns
        -------
        BaseBackend
            The Ibis backend.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.duckdb.connect()
        >>> t = con.create_table("t", {"id": [1, 2, 3]})
        >>> t.get_backend()  # doctest: +ELLIPSIS
        <ibis.backends.duckdb.Backend object at 0x...>

        See Also
        --------
        [`ibis.get_backend()`](./connection.qmd#ibis.get_backend)
        """
        return self._find_backend(use_default=True)

    def execute(
        self,
        *,
        limit: int | str | None = "default",
        params: Mapping[ir.Value, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an expression against its backend if one exists.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value
        kwargs
            Keyword arguments

        Examples
        --------
        >>> import ibis
        >>> t = ibis.examples.penguins.fetch()
        >>> t.execute()
               species     island  bill_length_mm  ...  body_mass_g     sex  year
        0       Adelie  Torgersen            39.1  ...       3750.0    male  2007
        1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
        2       Adelie  Torgersen            40.3  ...       3250.0  female  2007
        3       Adelie  Torgersen             NaN  ...          NaN    None  2007
        4       Adelie  Torgersen            36.7  ...       3450.0  female  2007
        ..         ...        ...             ...  ...          ...     ...   ...
        339  Chinstrap      Dream            55.8  ...       4000.0    male  2009
        340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
        341  Chinstrap      Dream            49.6  ...       3775.0    male  2009
        342  Chinstrap      Dream            50.8  ...       4100.0    male  2009
        343  Chinstrap      Dream            50.2  ...       3775.0  female  2009
        [344 rows x 8 columns]

        Scalar parameters can be supplied dynamically during execution.
        >>> species = ibis.param("string")
        >>> expr = t.filter(t.species == species).order_by(t.bill_length_mm)
        >>> expr.execute(limit=3, params={species: "Gentoo"})
          species  island  bill_length_mm  ...  body_mass_g     sex  year
        0  Gentoo  Biscoe            40.9  ...         4650  female  2007
        1  Gentoo  Biscoe            41.7  ...         4700  female  2009
        2  Gentoo  Biscoe            42.0  ...         4150  female  2007
        <BLANKLINE>
        [3 rows x 8 columns]

        See Also
        --------
        [`Table.to_pandas()`](./expression-tables.qmd#ibis.expr.types.relations.Table.to_pandas)
        [`Value.to_pandas()`](./expression-generic.qmd#ibis.expr.types.generic.Value.to_pandas)
        """
        return self._find_backend(use_default=True).execute(
            self, limit=limit, params=params, **kwargs
        )

    def to_sql(
        self, dialect: str | None = None, pretty: bool = True, **kwargs
    ) -> SQLString:
        """Compile to a formatted SQL string.

        Parameters
        ----------
        dialect
            SQL dialect to use for compilation.
            Uses the dialect bound to self if not specified,
            or the default dialect if no dialect is bound.
        pretty
            Whether to use pretty formatting.
        kwargs
            Scalar parameters

        Returns
        -------
        str
            Formatted SQL string

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table({"a": "int", "b": "int"}, name="t")
        >>> expr = t.mutate(c=t.a + t.b)
        >>> expr.to_sql()  # doctest: +SKIP
        SELECT
          "t0"."a",
          "t0"."b",
          "t0"."a" + "t0"."b" AS "c"
        FROM "t" AS "t0"

        You can also specify the SQL dialect to use for compilation:
        >>> expr.to_sql(dialect="mysql")  # doctest: +SKIP
        SELECT
          `t0`.`a`,
          `t0`.`b`,
          `t0`.`a` + `t0`.`b` AS `c`
        FROM `t` AS `t0`

        See Also
        --------
        [`Value.to_sql()`](./expression-generic.qmd#ibis.expr.types.generic.Value.to_sql)
        [`Table.to_sql()`](./expression-tables.qmd#ibis.expr.types.relations.Table.to_sql)
        [`ibis.to_sql()`](./expression-generic.qmd#ibis.to_sql)
        [`Value.compile()`](./expression-generic.qmd#ibis.expr.types.generic.Value.compile)
        [`Table.compile()`](./expression-tables.qmd#ibis.expr.types.relations.Table.compile)
        """
        return ibis.to_sql(self, dialect=dialect, pretty=pretty, **kwargs)

    def compile(
        self,
        *,
        limit: int | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        pretty: bool = False,
    ) -> str | pl.LazyFrame:
        r"""Compile `expr` to a SQL string (for SQL backends) or a LazyFrame (for the polars backend).

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value
        pretty
            In case of SQL backends, return a pretty formatted SQL query.

        Returns
        -------
        str | pl.LazyFrame
            A SQL string or a LazyFrame object, depending on the backend of self.

        Examples
        --------
        >>> import ibis
        >>> d = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> con = ibis.duckdb.connect()
        >>> t = con.create_table("t", d)
        >>> expr = t.mutate(c=t.a + t.b)
        >>> expr.compile()
        'SELECT "t0"."a", "t0"."b", "t0"."a" + "t0"."b" AS "c" FROM "memory"."main"."t" AS "t0"'

        If you want to see the pretty formatted SQL query, set `pretty` to `True`.
        >>> expr.compile(pretty=True)
        'SELECT\n  "t0"."a",\n  "t0"."b",\n  "t0"."a" + "t0"."b" AS "c"\nFROM "memory"."main"."t" AS "t0"'

        If the expression does not have a backend, an error will be raised.
        >>> t = ibis.memtable(d)
        >>> expr = t.mutate(c=t.a + t.b)
        >>> expr.compile()  # quartodoc: +EXPECTED_FAILURE
        Traceback (most recent call last):
        ...
        ibis.common.exceptions.IbisError: Expression depends on no backends, and ...

        See Also
        --------
        [`Value.compile()`](./expression-generic.qmd#ibis.expr.types.generic.Value.compile)
        [`Table.compile()`](./expression-tables.qmd#ibis.expr.types.relations.Table.compile)
        [`Value.to_sql()`](./expression-generic.qmd#ibis.expr.types.generic.Value.to_sql)
        [`Table.to_sql()`](./expression-tables.qmd#ibis.expr.types.relations.Table.to_sql)
        """
        return self._find_backend().compile(
            self, limit=limit, params=params, pretty=pretty
        )

    @experimental
    def to_pyarrow_batches(
        self,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return a RecordBatchReader.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.
        kwargs
            Keyword arguments

        Returns
        -------
        results
            RecordBatchReader
        """
        return self._find_backend(use_default=True).to_pyarrow_batches(
            self,
            params=params,
            limit=limit,
            chunk_size=chunk_size,
            **kwargs,
        )

    @experimental
    def to_pyarrow(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table | pa.Array | pa.Scalar:
        """Execute expression to a pyarrow object.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        result
            If the passed expression is a Table, a pyarrow table is returned.
            If the passed expression is a Column, a pyarrow array is returned.
            If the passed expression is a Scalar, a pyarrow scalar is returned.
        """
        return self._find_backend(use_default=True).to_pyarrow(
            self, params=params, limit=limit, **kwargs
        )

    @experimental
    def to_polars(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """Execute expression and return results as a polars dataframe.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        kwargs
            Keyword arguments

        Returns
        -------
        DataFrame
            A polars dataframe holding the results of the executed expression.
        """
        return self._find_backend(use_default=True).to_polars(
            self, params=params, limit=limit, **kwargs
        )

    @experimental
    def to_pandas_batches(
        self,
        *,
        limit: int | str | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame | pd.Series | Any]:
        """Execute expression and return an iterator of pandas DataFrames.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned `DataFrame`.
        kwargs
            Keyword arguments

        Returns
        -------
        Iterator[pd.DataFrame]
        """
        return self._find_backend(use_default=True).to_pandas_batches(
            self,
            params=params,
            limit=limit,
            chunk_size=chunk_size,
            **kwargs,
        )

    @experimental
    def to_parquet(
        self,
        path: str | Path,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file.

        This method is eager and will execute the associated expression
        immediately.

        See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html for details.

        Parameters
        ----------
        path
            A string or Path where the Parquet file will be written.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

        Examples
        --------
        Write out an expression to a single parquet file.

        >>> import ibis
        >>> import tempfile
        >>> penguins = ibis.examples.penguins.fetch()
        >>> penguins.to_parquet(tempfile.mktemp())

        Partition on a single column.

        >>> penguins.to_parquet(tempfile.mkdtemp(), partition_by="year")

        Partition on multiple columns.

        >>> penguins.to_parquet(tempfile.mkdtemp(), partition_by=("year", "island"))

        ::: {.callout-note}
        ## Hive-partitioned output is currently only supported when using DuckDB
        :::
        """
        self._find_backend(use_default=True).to_parquet(
            self, path, params=params, **kwargs
        )

    def to_xlsx(
        self,
        path: str | Path,
        /,
        *,
        sheet: str = "Sheet1",
        header: bool = False,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ):
        """Write a table to an Excel file.

        Parameters
        ----------
        path
            Excel output path.
        sheet
            The name of the sheet to write to, eg 'Sheet3'.
        header
            Whether to include the column names as the first row.
        params
            Additional Ibis expression parameters to pass to the backend's
            write function.
        kwargs
            Additional arguments passed to the backend's write function.

        Notes
        -----
        Requires DuckDB >= 1.2.0.

        See Also
        --------
        [DuckDB's `excel` extension docs for writing](https://duckdb.org/docs/stable/extensions/excel.html#writing-xlsx-files)

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> con = ibis.duckdb.connect()
        >>> t = con.create_table(
        ...     "t",
        ...     ibis.memtable({"a": [1, 2, 3], "b": ["a", "b", "c"]}),
        ...     temp=True,
        ... )
        >>> t.to_xlsx("/tmp/test.xlsx")
        >>> os.path.exists("/tmp/test.xlsx")
        True
        """
        self._find_backend(use_default=True).to_xlsx(
            self, path, sheet=sheet, header=header, params=params, **kwargs
        )

    @experimental
    def to_parquet_dir(
        self,
        directory: str | Path,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file in a directory.

        This method is eager and will execute the associated expression
        immediately.

        See https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html for details.

        Parameters
        ----------
        directory
            The data target. A string or Path to the directory where the parquet file will be written.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.dataset.write_dataset
        """
        self._find_backend(use_default=True).to_parquet_dir(
            self, directory, params=params, **kwargs
        )

    @experimental
    def to_csv(
        self,
        path: str | Path,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file.

        This method is eager and will execute the associated expression
        immediately.

        See https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html for details.

        Parameters
        ----------
        path
            The data target. A string or Path where the CSV file will be written.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter
        """
        self._find_backend(use_default=True).to_csv(self, path, params=params, **kwargs)

    @experimental
    def to_delta(
        self,
        path: str | Path,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a Delta Lake table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data target. A string or Path to the Delta Lake table directory.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to deltalake.writer.write_deltalake method
        """
        self._find_backend(use_default=True).to_delta(
            self, path, params=params, **kwargs
        )

    @experimental
    def to_json(self, path: str | Path, /, **kwargs: Any) -> None:
        """Write the results of `expr` to a json file of [{column -> value}, ...] objects.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data target. A string or Path where the JSON file will be written.
        kwargs
            Additional, backend-specific keyword arguments.
        """
        self._find_backend(use_default=True).to_json(self, path, **kwargs)

    @experimental
    def to_torch(
        self,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Execute an expression and return results as a dictionary of torch tensors.

        Parameters
        ----------
        params
            Parameters to substitute into the expression.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        kwargs
            Keyword arguments passed into the backend's `to_torch` implementation.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of torch tensors, keyed by column name.
        """
        return self._find_backend(use_default=True).to_torch(
            self, params=params, limit=limit, **kwargs
        )

    def unbind(self) -> ir.Table:
        """Return an expression built on `UnboundTable` instead of backend-specific objects.

        Examples
        --------
        >>> import ibis
        >>> import pandas as pd
        >>> duckdb_con = ibis.duckdb.connect()
        >>> polars_con = ibis.polars.connect()
        >>> for backend in (duckdb_con, polars_con):
        ...     t = backend.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
        >>> bound_table = duckdb_con.table("t")
        >>> bound_table.get_backend().name
        'duckdb'
        >>> unbound_table = bound_table.unbind()
        >>> polars_con.execute(unbound_table)
           a
        0  1
        1  2
        2  3
        """
        from ibis.expr.rewrites import _, d, p

        rule = p.DatabaseTable >> d.UnboundTable(
            name=_.name, schema=_.schema, namespace=_.namespace
        )
        return self.op().replace(rule).to_expr()

    def as_table(self) -> ir.Table:
        """Convert an expression to a table."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into tables"
        )

    def as_scalar(self) -> ir.Scalar:
        """Convert an expression to a scalar."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into scalars"
        )


def _binop(op_class: type[ops.Binary], left: ir.Value, right: ir.Value) -> ir.Value:
    """Try to construct a binary operation.

    Parameters
    ----------
    op_class
        The `ops.Binary` subclass for the operation
    left
        Left operand
    right
        Right operand

    Returns
    -------
    ir.Value
        A value expression

    Examples
    --------
    >>> import ibis
    >>> import ibis.expr.operations as ops
    >>> expr = _binop(ops.TimeAdd, ibis.time("01:00"), ibis.interval(hours=1))
    >>> expr
    TimeAdd(datetime.time(1, 0), 1h): datetime.time(1, 0) + 1 h
    >>> _binop(ops.TimeAdd, 1, ibis.interval(hours=1))
    TimeAdd(datetime.time(0, 0, 1), 1h): datetime.time(0, 0, 1) + 1 h
    """
    try:
        node = op_class(left, right)
    except (ValidationError, NotImplementedError):
        return NotImplemented
    else:
        return node.to_expr()
