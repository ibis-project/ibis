"""The MySQL backend."""

from __future__ import annotations

import contextlib
import getpass
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge
from adbc_driver_manager import dbapi as adbc_dbapi

import ibis
import ibis.backends.sql.compilers as sc
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.base import TRUE, C, RenameTable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from urllib.parse import ParseResult

    import pandas as pd
    import polars as pl
    import pyarrow as pa


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "mysql"
    compiler = sc.mysql.compiler
    supports_create_or_replace = False

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        kwargs = {}
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database := url.path[1:].split("/", 1)[0]:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    @cached_property
    def version(self):
        with self.con.cursor() as cur:
            cur.execute("SELECT VERSION()")
            result = cur.fetch_arrow_table()
        return result.column(0)[0].as_py()

    def do_connect(
        self,
        host: str = "localhost",
        user: str | None = None,
        password: str | None = None,
        port: int = 3306,
        database: str | None = None,
        **kwargs,
    ) -> None:
        """Create an Ibis client using the passed connection parameters.

        Parameters
        ----------
        host
            Hostname
        user
            Username
        password
            Password
        port
            Port
        database
            Database to connect to
        kwargs
            Additional keyword arguments

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_MYSQL_HOST", "localhost")
        >>> user = os.environ.get("IBIS_TEST_MYSQL_USER", "ibis")
        >>> password = os.environ.get("IBIS_TEST_MYSQL_PASSWORD", "ibis")
        >>> database = os.environ.get("IBIS_TEST_MYSQL_DATABASE", "ibis-testing")
        >>> con = ibis.mysql.connect(database=database, host=host, user=user, password=password)
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int32
          bool_col        int8
          tinyint_col     int8
          smallint_col    int16
          int_col         int32
          bigint_col      int64
          float_col       float32
          double_col      float64
          date_string_col string
          string_col      string
          timestamp_col   timestamp
          year            int32
          month           int32
        """
        user = user or getpass.getuser()
        host = "127.0.0.1" if host == "localhost" else host
        password = password or ""

        # Also accept database/db from kwargs for backwards compat
        if database is None:
            database = kwargs.pop("database", kwargs.pop("db", None))

        uri = f"{user}:{password}@tcp({host}:{port})/{database or ''}"
        self.con = adbc_dbapi.connect(
            driver="mysql", db_kwargs={"uri": uri}, autocommit=True
        )

        self._post_connect()

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @classmethod
    def from_connection(cls, con, /, **kwargs):
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        new_backend._post_connect()
        return new_backend

    def disconnect(self) -> None:
        self.con.close()

    @property
    def current_database(self) -> str:
        sql = sg.select(self.compiler.f.database()).sql(self.dialect)
        with self.con.cursor() as cur:
            cur.execute(sql)
            table = cur.fetch_arrow_table()
        return table.column(0)[0].as_py()

    def list_databases(self, *, like: str | None = None) -> list[str]:
        # In MySQL, "database" and "schema" are synonymous
        with self.con.cursor() as cur:
            cur.execute("SHOW DATABASES")
            table = cur.fetch_arrow_table()
        databases = table.column(0).to_pylist()
        return self._filter_with_like(databases, like)

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        tmp_name = util.gen_name("mysql_schema")
        quoted_tmp = sg.to_identifier(tmp_name, quoted=self.compiler.quoted).sql(
            self.dialect
        )
        create_sql = f"CREATE TEMPORARY TABLE {quoted_tmp} AS SELECT * FROM ({query}) AS _t LIMIT 0"
        describe_sql = f"DESCRIBE {quoted_tmp}"
        drop_sql = f"DROP TEMPORARY TABLE IF EXISTS {quoted_tmp}"

        type_mapper = self.compiler.type_mapper
        with self.con.cursor() as cur:
            try:
                cur.execute(create_sql)
                cur.execute(describe_sql)
                result = cur.fetch_arrow_table()
            finally:
                cur.execute(drop_sql)

        fields = {}
        for i in range(result.num_rows):
            col_name = result.column(0)[i].as_py()
            type_string = result.column(1)[i].as_py()
            is_nullable = result.column(2)[i].as_py()
            fields[col_name] = type_mapper.from_string(
                type_string, nullable=is_nullable == "YES"
            )

        return sch.Schema(fields)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)

        describe_sql = sge.Describe(this=table).sql(self.dialect)
        with self.con.cursor() as cur:
            try:
                cur.execute(describe_sql)
                result = cur.fetch_arrow_table()
            except Exception as e:
                if "doesn't exist" in str(e):
                    raise com.TableNotFound(name) from e
                raise

        type_mapper = self.compiler.type_mapper
        fields = {}
        for i in range(result.num_rows):
            col_name = result.column(0)[i].as_py()
            type_string = result.column(1)[i].as_py()
            is_nullable = result.column(2)[i].as_py()
            fields[col_name] = type_mapper.from_string(
                type_string, nullable=is_nullable == "YES"
            )

        return sch.Schema(fields)

    def create_database(self, name: str, force: bool = False) -> None:
        sql = sge.Create(
            kind="DATABASE", exists=force, this=sg.to_identifier(name)
        ).sql(self.name)
        with self.con.cursor() as cur:
            cur.execute(sql)

    def drop_database(
        self, name: str, *, catalog: str | None = None, force: bool = False
    ) -> None:
        sql = sge.Drop(
            kind="DATABASE", exists=force, this=sg.table(name, catalog=catalog)
        ).sql(self.name)
        with self.con.cursor() as cur:
            cur.execute(sql)

    @contextlib.contextmanager
    def begin(self):
        cur = self.con.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.raw_sql(*args, **kwargs) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.name)

        cursor = self.con.cursor()
        try:
            cursor.execute(query, **kwargs)
        except Exception:
            cursor.close()
            raise
        else:
            return cursor

    # TODO: disable positional arguments
    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        if database is not None:
            table_loc = self._to_sqlglot_table(database)
        else:
            table_loc = sge.Table(
                db=sg.to_identifier(self.current_database, quoted=self.compiler.quoted),
                catalog=None,
            )

        conditions = [TRUE]

        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
        if table_loc.catalog or table_loc.db:
            conditions = [C.table_schema.eq(sge.convert(table_loc.sql(self.name)))]

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql(self.name)
        )

        with self.con.cursor() as cur:
            cur.execute(sql)
            table = cur.fetch_arrow_table()

        return self._filter_with_like(table.column(0).to_pylist(), like)

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        """Execute an Ibis expression and return a pandas `DataFrame`, `Series`, or scalar.

        Parameters
        ----------
        expr
            Ibis expression to execute.
        params
            Mapping of scalar parameter expressions to value.
        limit
            An integer to effect a specific row limit. A value of `None` means
            no limit. The default is in `ibis/config.py`.
        kwargs
            Keyword arguments
        """

        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, limit=limit, params=params, **kwargs)

        schema = table.schema()
        target_schema = schema.to_pyarrow()

        with self.con.cursor() as cur:
            cur.execute(sql)
            arrow_table = cur.fetch_arrow_table()

        arrow_table = self._cast_adbc_table(arrow_table, target_schema)

        import pandas as pd

        from ibis.formats.pandas import PandasData

        df = arrow_table.to_pandas(timestamp_as_object=False)
        if df.empty:
            df = pd.DataFrame(columns=schema.names)
        result = PandasData.convert_table(df, schema)
        return expr.__pandas_result__(result)

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table
        | pd.DataFrame
        | pa.Table
        | pl.DataFrame
        | pl.LazyFrame
        | None = None,
        *,
        schema: sch.IntoSchema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = []

        if temp:
            properties.append(sge.TemporaryProperty())

        if obj is not None:
            if not isinstance(obj, ir.Expr):
                table = ibis.memtable(obj)
            else:
                table = obj

            self._run_pre_execute_hooks(table)

            query = self.compiler.to_sqlglot(table)
        else:
            query = None

        if overwrite:
            temp_name = util.gen_name(f"{self.name}_table")
        else:
            temp_name = name

        if not schema:
            schema = table.schema()

        quoted = self.compiler.quoted
        dialect = self.dialect

        table_expr = sg.table(temp_name, catalog=database, quoted=quoted)
        target = sge.Schema(
            this=table_expr, expressions=schema.to_sqlglot_column_defs(dialect)
        )

        create_stmt = sge.Create(
            kind="TABLE", this=target, properties=sge.Properties(expressions=properties)
        )

        this = sg.table(name, catalog=database, quoted=quoted)
        with self._safe_raw_sql(create_stmt) as cur:
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite:
                cur.execute(sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect))
                cur.execute(
                    sge.Alter(
                        kind="TABLE",
                        this=table_expr,
                        exists=True,
                        actions=[RenameTable(this=this)],
                    ).sql(dialect)
                )

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := schema.null_fields:
            schema = ibis.schema(
                {
                    name: dt.string if name in null_columns else typ
                    for name, typ in schema.items()
                }
            )

        arrow_table = op.data.to_pyarrow(schema)

        name = op.name
        quoted = self.compiler.quoted
        dialect = self.dialect

        # ADBC's adbc_ingest temporary=True doesn't actually create
        # TEMPORARY tables in MySQL. Create the temp table with DDL first,
        # then use adbc_ingest in append mode to insert data.
        create_stmt = sge.Create(
            kind="TABLE",
            this=sge.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(dialect),
            ),
            properties=sge.Properties(expressions=[sge.TemporaryProperty()]),
        )

        ncols = len(schema)
        # MySQL has a 65535 prepared statement placeholder limit.
        # Set batch size to stay under it.
        batch_size = max(1, 65535 // max(ncols, 1) - 1)

        with self.con.cursor() as cur:
            cur.execute(create_stmt.sql(dialect))
            if arrow_table.num_rows > 0:
                cur.adbc_statement.set_options(
                    **{"adbc.statement.ingest.batch_size": str(batch_size)}
                )
                cur.adbc_ingest(name, arrow_table, mode="append")

    @staticmethod
    def _decode_opaque_storage(storage):
        """Decode ADBC opaque extension type storage to a string array.

        The ADBC MySQL driver stores some values (e.g., UNSIGNED BIGINT) as
        bracket-delimited ASCII byte sequences like ``"[52 50]"`` for ``"42"``.
        Other values are stored as plain strings.  This normalizes both forms
        into a plain string array.
        """
        import pyarrow as pa

        decoded = []
        for val in storage:
            raw = val.as_py()
            if raw is None:
                decoded.append(None)
            elif raw.startswith("[") and raw.endswith("]"):
                byte_values = [int(x) for x in raw[1:-1].split()]
                decoded.append(bytes(byte_values).decode("ascii"))
            else:
                decoded.append(raw)
        return pa.array(decoded, type=pa.string())

    @staticmethod
    def _cast_adbc_column(col, target_type):
        """Cast a single ADBC-returned Arrow column to the target type.

        ADBC MySQL returns opaque extension types for some MySQL types (NULL,
        unsigned integers, etc.) that PyArrow cannot cast directly. This method
        handles those by extracting the storage array first.
        """
        import pyarrow as pa
        import pyarrow.compute as pc

        if col.type == target_type:
            return col
        elif target_type == pa.null():
            return pa.nulls(len(col))
        elif isinstance(col.type, pa.BaseExtensionType):
            storage = (
                col.storage
                if isinstance(col, pa.Array)
                else col.combine_chunks().storage
            )
            # All-null opaque columns (e.g., type_name=NULL) can't be cast
            # meaningfully; return typed nulls directly.
            if storage.null_count == len(storage):
                return pa.nulls(len(storage), type=target_type)
            if storage.type in (pa.string(), pa.utf8()):
                decoded = Backend._decode_opaque_storage(storage)
                # For unsigned integer types that overflow the target signed
                # type (e.g., MySQL ~x returns UNSIGNED BIGINT), parse as
                # uint64 first and let the overflow wrap via two's complement.
                if pa.types.is_integer(target_type):
                    arr = decoded.cast(pa.uint64())
                    return arr.cast(target_type, safe=False)
                return decoded.cast(target_type)
            return storage.cast(target_type)
        else:
            try:
                return col.cast(target_type)
            except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
                # Some casts aren't directly supported (e.g., decimal ->
                # float16); try going through float64 as an intermediate.
                try:
                    return col.cast(pa.float64()).cast(target_type)
                except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
                    # If that also fails (e.g., double -> interval), leave
                    # as-is and let PandasData.convert_table handle it.
                    return col

    @classmethod
    def _cast_adbc_table(cls, table, target_schema):
        """Cast an ADBC-returned Arrow Table to match the target schema."""
        import pyarrow as pa

        columns = [
            cls._cast_adbc_column(table.column(i), field.type)
            for i, field in enumerate(target_schema)
        ]
        return pa.table(
            dict(zip(target_schema.names, columns)),
        )

    @classmethod
    def _cast_adbc_batch(cls, batch, target_schema):
        """Cast an ADBC-returned Arrow RecordBatch to match the target schema."""
        import pyarrow as pa

        columns = [
            cls._cast_adbc_column(batch.column(i), field.type)
            for i, field in enumerate(target_schema)
        ]
        return pa.record_batch(columns, schema=target_schema)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        import pyarrow as pa

        self._run_pre_execute_hooks(expr)

        table_expr = expr.as_table()
        sql = self.compile(table_expr, limit=limit, params=params)
        target_schema = table_expr.schema().to_pyarrow()

        cur = self.raw_sql(sql)
        reader = cur.fetch_record_batch()

        def batch_producer():
            try:
                for batch in reader:
                    yield self._cast_adbc_batch(
                        batch.rename_columns(target_schema.names), target_schema
                    )
            finally:
                cur.close()

        return pa.ipc.RecordBatchReader.from_batches(
            target_schema, batch_producer()
        )
