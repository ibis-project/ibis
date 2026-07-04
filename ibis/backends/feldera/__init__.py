"""Feldera backend for Ibis.

Feldera is an incremental SQL query engine.  An Ibis backend for Feldera lets
you use the Ibis dataframe API to define and query views that live inside a
running Feldera *pipeline* and read their current state as a (Arrow / pandas)
DataFrame.

The mapping to Ibis's connection model is:

* ``connect(...)``  -> a :class:`FelderaClient` plus the name of an *existing*
  pipeline.  (Following the convention adopted by ``dbt-feldera``, we treat a
  pipeline as a "schema" and the tables/views inside it as "relations".)
* ``table(name)``    -> introspect the table/view schema via the pipeline API.
* ``execute(expr)``  -> compile the expression to a ``SELECT`` and run it as an
  ad-hoc query against the pipeline via ``Pipeline.query_arrow``.  This returns
  a *snapshot* of the view at the current moment; Feldera keeps maintaining the
  view incrementally underneath.

Streaming consumption (listening to a view's changelog) is out of scope for the
standard Ibis ``execute()`` contract, just as it is for the Flink backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.feldera import compiler as _feldera_compiler
from ibis.backends.sql.compilers.feldera import FelderaCompiler as _FelderaCompiler

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from feldera import FelderaClient
    from feldera.pipeline import Pipeline


class Backend(SQLBackend):
    name = "feldera"
    compiler = _feldera_compiler

    # Keep a handle to the compiler class for schema conversion helpers.

    # Feldera (Calcite) has no catalogs and no nested "databases"; the pipeline
    # is the only container, and it is fixed at connect() time.  We therefore
    # don't mix in CanCreateDatabase / HasCurrentCatalog / HasCurrentDatabase.

    def _from_url(self, url, **kwargs):  # noqa: D401
        """Feldera does not use connection URLs."""
        raise NotImplementedError(
            "The Feldera backend does not support connection URLs; use "
            "ibis.feldera.connect(host=..., pipeline=...) instead."
        )

    def do_connect(
        self,
        *,
        host: str | None = None,
        pipeline: str,
        api_key: str | None = None,
        client: "FelderaClient | None" = None,
    ) -> None:
        """Connect to a Feldera pipeline.

        Parameters
        ----------
        host
            URL of the Feldera API (e.g. ``"http://localhost:8080"``).
            Ignored if ``client`` is supplied.
        pipeline
            Name of the *existing* pipeline to query.  The pipeline must
            already be created (and typically running) on the Feldera instance.
        api_key
            Optional API key.  Ignored if ``client`` is supplied.
        client
            An existing :class:`feldera.FelderaClient` to reuse, e.g. for
            tests or when integrating with code that already holds a client.

        Examples
        --------
        >>> import ibis
        >>> con = ibis.feldera.connect(host="http://localhost:8080", pipeline="penguins")
        >>> con.table("penguins").head(3)  # doctest: +SKIP
        """
        from feldera import FelderaClient

        if client is None:
            client = FelderaClient(url=host, api_key=api_key)
        self._client: FelderaClient = client
        self._pipeline_name = pipeline

    @util.experimental
    @classmethod
    def from_connection(cls, client: "FelderaClient", /, *, pipeline: str) -> "Backend":
        """Create a Feldera :class:`Backend` from an existing client.

        Parameters
        ----------
        client
            A :class:`feldera.FelderaClient`.
        pipeline
            Name of the existing pipeline to query.
        """
        return ibis.feldera.connect(client=client, pipeline=pipeline)

    def disconnect(self) -> None:
        # The Feldera client holds no persistent resources that need closing.
        pass

    def version(self) -> str:
        """Return the connected Feldera server version."""
        try:
            return self._client.get_config().version
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _pipeline(self) -> "Pipeline":
        from feldera.pipeline import Pipeline

        return Pipeline.get(self._pipeline_name, self._client)

    @property
    def dialect(self) -> sg.Dialect:
        return self.compiler.dialect

    # ------------------------------------------------------------------ #
    # SQL execution
    # ------------------------------------------------------------------ #
    def raw_sql(self, query: str) -> Any:
        """Run an ad-hoc SQL query against the pipeline.

        Returns a generator of ``pyarrow.RecordBatch``.
        """
        return self._pipeline().query_arrow(query)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        import pyarrow as pa

        # Run the query with a LIMIT 0 to fetch just the schema.  Feldera's
        # query_arrow returns PyArrow RecordBatches; the first batch (even if
        # empty) carries the schema.
        schema_sql = f"SELECT * FROM ({query}) AS __ibis_schema__ LIMIT 0"
        batches = self._pipeline().query_arrow(schema_sql)
        first = next(batches, None)
        if first is None:
            # Fall back to running the full query if LIMIT 0 yields nothing.
            return sch.Schema({})
        pa_schema: pa.Schema = first.schema
        return sch.Schema.from_pyarrow(pa_schema)

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> "pd.DataFrame":
        import pandas as pd
        import pyarrow as pa

        batches = list(cursor)
        if not batches:
            # Build an empty DataFrame with the right columns/dtypes.
            data = {name: pd.Series(dtype=dtype.to_pandas()) for name, dtype in schema.items()}
            return pd.DataFrame(data)
        table = pa.Table.from_batches(batches, batches[0].schema)
        return table.to_pandas()

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: dict | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> "pd.DataFrame | pd.Series | Any":
        """Execute an Ibis expression and return a pandas DataFrame.

        Compiles the expression to a ``SELECT`` and runs it as an ad-hoc query
        against the connected Feldera pipeline.  Returns a snapshot of the
        view at the current moment.
        """
        self._run_pre_execute_hooks(expr)

        sql = self.compile(expr.as_table(), params=params, limit=limit, **kwargs)
        cursor = self._pipeline().query_arrow(sql)
        df = self._fetch_from_cursor(cursor, expr.as_table().schema())
        return expr.__pandas_result__(df)

    def to_pyarrow(
        self,
        expr: ir.Expr,
        /,
        *,
        params: dict | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> "pa.Table":
        import pyarrow as pa

        self._run_pre_execute_hooks(expr)
        sql = self.compile(expr.as_table(), params=params, limit=limit, **kwargs)
        batches = list(self._pipeline().query_arrow(sql))
        if not batches:
            return schema_to_empty_pa_table(expr.as_table().schema())
        return pa.Table.from_batches(batches, batches[0].schema)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def list_tables(self, *, like: str | None = None, **kwargs: Any) -> list[str]:
        p = self._pipeline()
        names = [t.name for t in p.tables()] + [v.name for v in p.views()]
        return self._filter_with_like(names, like)

    def get_schema(
        self,
        name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        if catalog is not None or database is not None:
            raise NotImplementedError(
                "Feldera has no catalogs/databases; the pipeline is fixed at "
                "connect() time."
            )
        p = self._pipeline()
        # tables first, then views
        candidates = list(p.tables()) + list(p.views())
        for rel in candidates:
            if rel.name.lower() == name.lower():
                return _schema_from_feldera_fields(rel.fields)
        raise KeyError(f"Table or view {name!r} not found in pipeline {self._pipeline_name!r}")

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        # Feldera tables live in the pipeline SQL program; there is no notion
        # of an ad-hoc in-memory table outside the program.  In-memory data is
        # pushed via create_table / input_pandas instead.
        raise NotImplementedError(
            "In-memory tables are not supported by the Feldera backend; use "
            "create_table(obj=df) to push data into a pipeline table instead."
        )

    def create_table(
        self,
        name: str,
        /,
        obj: "pd.DataFrame | pa.Table | ir.Table | None" = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> ir.Table:
        """Create a table or push data into an existing pipeline table.

        Feldera tables are declared in the pipeline's SQL program; they cannot
        be created ad hoc outside the program.  This method therefore supports
        two modes:

        * If ``obj`` is a pandas DataFrame and a table named ``name`` already
          exists in the pipeline, the rows are pushed into it via
          ``Pipeline.input_pandas`` (the standard Feldera data-ingress path).
        * Otherwise (e.g. ``obj is None`` with a fresh schema), this raises
          ``NotImplementedError``: defining new tables requires editing the
          pipeline SQL program, which is out of scope for this backend.
        """
        if database is not None:
            raise NotImplementedError(
                "Feldera has no databases; the pipeline is fixed at connect() time."
            )
        import pandas as pd

        if obj is None:
            raise NotImplementedError(
                "Creating a new table from a schema alone is not supported; define "
                "the table in the pipeline SQL program instead."
            )
        if isinstance(obj, ir.Table):
            obj = obj.to_pandas()
        try:
            import pyarrow as pa
            if isinstance(obj, pa.Table):
                obj = obj.to_pandas()
        except ImportError:
            pass
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"Unsupported obj type: {type(obj)!r}")

        self._pipeline().input_pandas(name, obj, force=overwrite)
        return self.table(name)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        if database is not None:
            raise NotImplementedError(
                "Feldera has no databases; the pipeline is fixed at connect() time."
            )
        raise NotImplementedError(
            "Feldera does not support dropping tables at runtime; tables are "
            "declared in the pipeline SQL program."
        )


def _schema_from_feldera_fields(fields: list[dict]) -> sch.Schema:
    """Convert Feldera's field dicts (``{"name", "columntype"}``) to an ibis schema."""
    type_mapper = _feldera_compiler.type_mapper
    out = {}
    for f in fields:
        name = f.get("name", "")
        ctype = f.get("columntype", {})
        type_str = ctype.get("type", "VARCHAR") if isinstance(ctype, dict) else str(ctype)
        # nullable info isn't reliably exposed; default to nullable=True
        out[name] = type_mapper.from_string(type_str, nullable=True)
    return sch.Schema(out)


def schema_to_empty_pa_table(schema: sch.Schema):
    import pyarrow as pa

    fields = [
        pa.field(name, dtype.to_pyarrow(), nullable=dtype.nullable)
        for name, dtype in ((n, schema[n]) for n in schema.names)
    ]
    return pa.table({f.name: pa.array([], type=f.type) for f in fields}, schema=pa.schema(fields))
