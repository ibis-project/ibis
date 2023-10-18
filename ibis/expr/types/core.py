from __future__ import annotations

import contextlib
import os
import webbrowser
from typing import TYPE_CHECKING, Any, Mapping, NoReturn, Tuple, Iterator

from public import public
from rich.jupyter import JupyterMixin

import ibis.expr.operations as ops
from ibis.common.annotations import ValidationError
from ibis.common.exceptions import IbisError, TranslationError
from ibis.common.grounds import Immutable
from ibis.common.patterns import Coercible, CoercionError
from ibis.config import _default_backend, options as opts
from ibis.util import experimental

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    import torch

    import ibis.expr.types as ir
    from ibis.backends.base import BaseBackend

    TimeContext = Tuple[pd.Timestamp, pd.Timestamp]


class _FixedTextJupyterMixin(JupyterMixin):
    """JupyterMixin adds a spurious newline to text, this fixes the issue."""

    def _repr_mimebundle_(self, *args, **kwargs):
        bundle = super()._repr_mimebundle_(*args, **kwargs)
        bundle["text/plain"] = bundle["text/plain"].rstrip()
        return bundle


# TODO(kszucs): consider to subclass from Annotable with a single _arg field
@public
class Expr(Immutable, Coercible):
    """Base expression class."""

    __slots__ = ("_arg",)
    _arg: ops.Node

    def __rich_console__(self, console, options):
        if not opts.interactive:
            from rich.text import Text

            return console.render(Text(self._repr()), options=options)
        return self.__interactive_rich_console__(console, options)

    def __interactive_rich_console__(self, console, options):
        raise NotImplementedError()

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

    def __repr__(self) -> str:
        if not opts.interactive:
            return self._repr()

        from ibis.expr.types.pretty import simple_console

        with simple_console.capture() as capture:
            try:
                simple_console.print(self)
            except TranslationError as e:
                lines = [
                    "Translation to backend failed",
                    f"Error message: {e.args[0]}",
                    "Expression repr follows:",
                    self._repr(),
                ]
                return "\n".join(lines)
        return capture.get().rstrip()

    def __reduce__(self):
        return (self.__class__, (self._arg,))

    def __hash__(self):
        return hash((self.__class__, self._arg))

    def _repr(self) -> str:
        from ibis.expr.format import pretty

        return pretty(self)

    def equals(self, other):
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

    def has_name(self):
        """Check whether this expression has an explicit name."""
        return isinstance(self._arg, ops.Named)

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
    ) -> None:
        """Visualize an expression as a GraphViz graph in the browser.

        Parameters
        ----------
        format
            Image output format. These are specified by the ``graphviz`` Python
            library.
        label_edges
            Show operation input names as edge labels
        verbose
            Print the graphviz DOT code to stderr if [](`True`)

        Raises
        ------
        ImportError
            If ``graphviz`` is not installed.
        """
        import ibis.expr.visualize as viz

        path = viz.draw(
            viz.to_graph(self, label_edges=label_edges),
            format=format,
            verbose=verbose,
        )
        webbrowser.open(f"file://{os.path.abspath(path)}")

    def pipe(self, f, *args: Any, **kwargs: Any) -> Expr:
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
        >>> t = ibis.table([("a", "int64"), ("b", "string")], name="t")
        >>> f = lambda a: (a + 1).name("a")
        >>> g = lambda a: (a * 2).name("a")
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1
        r0 := UnboundTable: t
          a int64
          b string
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

    def op(self) -> ops.Node:  # noqa: D102
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
        node_types = (ops.DatabaseTable, ops.SQLQueryResult, ops.UnboundTable)
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
                    "be executed. Use ibis.<backend>.execute(expr) or "
                    "assign a backend instance to "
                    "`ibis.options.default_backend`."
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

    def execute(
        self,
        limit: int | str | None = "default",
        timecontext: TimeContext | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        **kwargs: Any,
    ):
        """Execute an expression against its backend if one exists.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        timecontext
            Defines a time range of `(begin, end)`. When defined, the execution
            will only compute result for data inside the time range. The time
            range is inclusive of both endpoints. This is conceptually same as
            a time filter.
            The time column must be named `'time'` and should preserve
            across the expression. For example, if that column is dropped then
            execute will result in an error.
        params
            Mapping of scalar parameter expressions to value
        kwargs
            Keyword arguments
        """
        return self._find_backend(use_default=True).execute(
            self, limit=limit, timecontext=timecontext, params=params, **kwargs
        )

    def compile(
        self,
        limit: int | None = None,
        timecontext: TimeContext | None = None,
        params: Mapping[ir.Value, Any] | None = None,
    ):
        """Compile to an execution target.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        timecontext
            Defines a time range of `(begin, end)`. When defined, the execution
            will only compute result for data inside the time range. The time
            range is inclusive of both endpoints. This is conceptually same as
            a time filter.
            The time column must be named `'time'` and should preserve
            across the expression. For example, if that column is dropped then
            execute will result in an error.
        params
            Mapping of scalar parameter expressions to value
        """
        return self._find_backend().compile(
            self, limit=limit, timecontext=timecontext, params=params
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
    ) -> pa.Table:
        """Execute expression and return results in as a pyarrow table.

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
        Table
            A pyarrow table holding the results of the executed expression.
        """
        return self._find_backend(use_default=True).to_pyarrow(
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
            Maximum number of rows in each returned `DataFrame``.
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
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a parquet file

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the parquet file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html

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
        self._find_backend(use_default=True).to_parquet(self, path, **kwargs)

    @experimental
    def to_csv(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html
        """
        self._find_backend(use_default=True).to_csv(self, path, **kwargs)

    @experimental
    def to_delta(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a Delta Lake table

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the Delta Lake table directory.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html
        """
        self._find_backend(use_default=True).to_delta(self, path, **kwargs)

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
        """Return an expression built on `UnboundTable` instead of backend-specific objects."""
        from ibis.expr.analysis import p, c, _

        rule = p.DatabaseTable >> c.UnboundTable(name=_.name, schema=_.schema)
        return self.op().replace(rule).to_expr()

    def as_table(self) -> ir.Table:
        """Convert an expression to a table."""
        raise NotImplementedError(type(self))


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
