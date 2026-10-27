"""IBM Db2 backend for Ibis."""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import CanListDatabase, HasCurrentDatabase, PyArrowExampleLoader
from ibis.backends.db2.converter import Db2PandasData
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.db2 import Db2Compiler

if TYPE_CHECKING:
    from urllib.parse import ParseResult

    import ibm_db_dbi
    import pandas as pd
    import pyarrow as pa

# The driver returns Python datetimes, which hold microseconds, so report no
# more than 6 fractional digits even if the column is declared with more.
_MAX_TIMESTAMP_SCALE = 6

# Result-set column kinds reported by ibm_db.field_type that map to a single
# Ibis type, regardless of precision or scale.
_SIMPLE_FIELD_TYPES = {
    "bigint": dt.Int64,
    "blob": dt.Binary,
    "clob": dt.String,
    "date": dt.Date,
    "time": dt.Time,
}


def _temporary_properties():
    # Db2 deletes the rows of a GLOBAL TEMPORARY table at every commit by
    # default; keep them for the life of the session.
    return [
        sge.GlobalProperty(),
        sge.TemporaryProperty(),
        sge.OnCommitProperty(delete=False),
    ]


class Backend(SQLBackend, CanListDatabase, HasCurrentDatabase, PyArrowExampleLoader):
    """IBM Db2 backend for Ibis."""

    name = "db2"
    compiler = Db2Compiler()
    supports_temporary_tables = True
    supports_python_udfs = False

    @property
    def version(self) -> str:
        """Return the version of the Db2 server."""
        with self._safe_raw_sql(
            "SELECT SERVICE_LEVEL FROM SYSIBMADM.ENV_INST_INFO"
        ) as cur:
            result = cur.fetchone()
            return result[0] if result else "unknown"

    def do_connect(
        self,
        *,
        database: str,
        host: str = "localhost",
        port: int = 50000,
        user: str | None = None,
        password: str | None = None,
        schema: str | None = None,
        ssl: bool = False,
        ssl_server_certificate: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Connect to a Db2 database.

        Parameters
        ----------
        database : str
            Database name
        host : str, default "localhost"
            Hostname of Db2 server
        port : int, default 50000
            Port number
        user : str, optional
            Username for authentication
        password : str, optional
            Password for authentication
        schema : str, optional
            Default schema
        ssl : bool, default False
            Enable SSL/TLS encrypted connection. When ``True``, the connection
            string includes ``SECURITY=SSL``.
        ssl_server_certificate : str or Path, optional
            Path to the server's SSL certificate (PEM or ARM format). Passed
            as ``SSLServerCertificate=<path>`` in the connection string. If
            ``ssl`` is ``True`` but this is ``None``, server certificate
            validation is skipped (useful for self-signed certificates in dev
            environments).
        **kwargs
            Additional IBM Db2 connection string key=value parameters.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> host = os.environ.get("IBIS_TEST_DB2_HOST", "localhost")
        >>> port = int(os.environ.get("IBIS_TEST_DB2_PORT", "50000"))
        >>> user = os.environ.get("IBIS_TEST_DB2_USER", "db2inst1")
        >>> password = os.environ.get("IBIS_TEST_DB2_PASSWORD", "ibistest")
        >>> database = os.environ.get("IBIS_TEST_DB2_DATABASE", "ibistest")
        >>> con = ibis.db2.connect(
        ...     database=database, host=host, port=port, user=user, password=password
        ... )
        >>> con.list_tables()  # doctest: +ELLIPSIS
        [...]
        >>> t = con.table("functional_alltypes")
        >>> t
        DatabaseTable: functional_alltypes
          id              int32
          bool_col        int16
          tinyint_col     int16
          smallint_col    int16
          int_col         int32
          bigint_col      int64
          float_col       float32
          double_col      float64
          date_string_col string(255)
          string_col      string(255)
          timestamp_col   timestamp
          year            int32
          month           int32
        """
        import ibm_db_dbi

        # Build connection string
        conn_str_parts = [
            f"DATABASE={database}",
            f"HOSTNAME={host}",
            f"PORT={port}",
            "PROTOCOL=TCPIP",
        ]

        if user:
            conn_str_parts.append(f"UID={user}")
        if password:
            conn_str_parts.append(f"PWD={password}")

        # SSL parameters — ibm_db uses SECURITY=SSL in the connection string
        if ssl:
            conn_str_parts.append("SECURITY=SSL")
            if ssl_server_certificate is not None:
                conn_str_parts.append(
                    f"SSLServerCertificate={Path(ssl_server_certificate)}"
                )

        # Add any additional connection parameters
        _reserved = {
            "DATABASE",
            "HOSTNAME",
            "PORT",
            "UID",
            "PWD",
            "SECURITY",
            "SSLSERVERCERTIFICATE",
        }
        for key, value in kwargs.items():
            if key.upper() not in _reserved:
                conn_str_parts.append(f"{key.upper()}={value}")

        if schema:
            conn_str_parts.append(f"CURRENTSCHEMA={schema}")

        # ibm_db_dbi.connect() defaults to autocommit OFF, which makes every
        # read hold locks until the next commit. Every statement here is
        # expected to take effect immediately, so autocommit is set explicitly.
        self.con = ibm_db_dbi.connect(
            ";".join(conn_str_parts),
            conn_options={ibm_db_dbi.SQL_ATTR_AUTOCOMMIT: ibm_db_dbi.SQL_AUTOCOMMIT_ON},
        )

    @classmethod
    def from_connection(cls, con: ibm_db_dbi.Connection, /) -> Backend:
        """Create an Ibis client from an existing connection to a Db2 database.

        Parameters
        ----------
        con
            An existing `ibm_db_dbi` connection. The connection is used as
            given: it is not modified. Ibis expects every statement to take
            effect immediately, so turn autocommit on before passing it in
            (`con.set_autocommit(True)`); otherwise a warning is issued.
        """
        import ibm_db

        if not ibm_db.autocommit(con.conn_handler):
            warnings.warn(
                "autocommit is off; reads may hold locks and changes stay "
                "uncommitted until you call con.commit()",
                stacklevel=2,
            )
        new_backend = cls()
        new_backend._can_reconnect = False
        new_backend.con = con
        return new_backend

    def _convert_kwargs(self, kwargs):
        with contextlib.suppress(KeyError):
            kwargs["ssl"] = str(kwargs["ssl"]).lower() in ("1", "true", "yes")

    def _from_url(self, url: ParseResult, **kwarg_overrides):
        """Create a Db2 backend from a URL.

        Parameters
        ----------
        url : ParseResult
            Parsed URL object
        **kwarg_overrides
            Additional keyword arguments to override URL parameters

        Returns
        -------
        Self
            Connected Db2 backend instance

        Notes
        -----
        SSL can be enabled via query parameters in the URL::

            db2://user:pass@host:50001/SAMPLE?ssl=true
            db2://user:pass@host:50001/SAMPLE?ssl=true&ssl_server_certificate=/path/to/cert.pem
        """
        kwargs = {}
        database, *schema = url.path[1:].split("/", 1)
        if url.username:
            kwargs["user"] = url.username
        if url.password:
            kwargs["password"] = unquote_plus(url.password)
        if url.hostname:
            kwargs["host"] = url.hostname
        if database:
            kwargs["database"] = database
        if url.port:
            kwargs["port"] = url.port
        if schema:
            kwargs["schema"] = schema[0]

        # `ibis.connect` has already turned the URL query string into keyword
        # arguments and passed them in `kwarg_overrides`.
        kwargs.update(kwarg_overrides)
        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str | sg.Expression, **kwargs: Any):
        """Execute raw SQL safely with cursor management."""
        if isinstance(query, sg.exp.Expression):
            query = query.sql(dialect=self.compiler.dialect)

        cursor = self.con.cursor()
        try:
            cursor.execute(query, **kwargs)
            yield cursor
        finally:
            cursor.close()

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        """Execute a raw SQL query.

        Parameters
        ----------
        query : str | sg.Expression
            SQL query to execute
        **kwargs
            Additional parameters

        Returns
        -------
        Any
            Query results (cursor)
        """
        if isinstance(query, sg.exp.Expression):
            query = query.sql(dialect=self.compiler.dialect)

        # Don't use context manager as it closes the cursor
        cursor = self.con.cursor()
        try:
            cursor.execute(query, **kwargs)
        except Exception:
            cursor.close()
            raise
        else:
            return cursor

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        df = pd.DataFrame.from_records(
            cursor.fetchall(), columns=schema.names, coerce_float=True
        )
        return Db2PandasData.convert_table(df, schema)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        """List tables in the database.

        Parameters
        ----------
        like : str, optional
            Pattern to filter table names
        database : str, optional
            Database/schema name

        Returns
        -------
        list[str]
            List of table names
        """
        query = """
            SELECT TABNAME
            FROM SYSCAT.TABLES
            WHERE TABSCHEMA = COALESCE(?, CURRENT SCHEMA)
            AND TYPE IN ('T', 'V', 'G')
        """

        cursor = self.con.cursor()
        try:
            cursor.execute(query, (database,))
            tables = [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

        return self._filter_with_like(tables, like)

    def list_databases(
        self, *, like: str | None = None, catalog: str | None = None
    ) -> list[str]:
        """List schemas in the database.

        Parameters
        ----------
        like : str, optional
            Pattern to filter schema names
        catalog : str, optional
            Not supported: Db2 has no cross-catalog schema access.

        Returns
        -------
        list[str]
            List of schema names
        """
        if catalog is not None:
            raise exc.UnsupportedArgumentError("No cross-catalog schema access in Db2")

        query = """
            SELECT RTRIM(SCHEMANAME)
            FROM SYSCAT.SCHEMATA
            WHERE SCHEMANAME NOT LIKE 'SYS%'
        """

        with self._safe_raw_sql(query) as cursor:
            databases = [row[0] for row in cursor.fetchall()]

        return self._filter_with_like(databases, like)

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        """Get the schema of a table.

        Parameters
        ----------
        table_name : str
            Name of the table
        catalog : str, optional
            Catalog name (unused in Db2)
        database : str, optional
            Schema name

        Returns
        -------
        sch.Schema
            Table schema
        """
        query = """
            SELECT COLNAME, TYPENAME, LENGTH, SCALE, NULLS, CODEPAGE
            FROM SYSCAT.COLUMNS
            WHERE TABNAME = ?
            AND TABSCHEMA = COALESCE(?, CURRENT SCHEMA)
            ORDER BY COLNO
        """

        schema_name = database or self.current_database

        cursor = self.con.cursor()
        try:
            cursor.execute(query, (table_name, schema_name))
            rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            raise exc.TableNotFound(table_name)

        type_mapper = self.compiler.type_mapper
        fields = {}
        for col_name, type_name, length, scale, nulls, codepage in rows:
            # Reconstruct the full type string so SQLGlot's Db2 dialect parser
            # can handle it — this delegates all type parsing to the same
            # type_mapper (Db2Type / SqlglotType.from_string) that the rest of
            # the compiler uses, rather than duplicating the mapping by hand.
            if type_name in ("DECIMAL", "NUMERIC"):
                type_str = f"{type_name}({length},{scale})"
            elif type_name == "TIMESTAMP":
                type_str = f"TIMESTAMP({min(scale, _MAX_TIMESTAMP_SCALE)})"
            elif type_name in ("VARCHAR", "CHARACTER", "VARBINARY"):
                type_str = f"{type_name}({length})"
            else:
                type_str = type_name

            if type_name in ("VARCHAR", "CHARACTER") and codepage == 0:
                # FOR BIT DATA columns have codepage 0 and the driver returns bytes
                ibis_type = dt.Binary(nullable=(nulls == "Y"))
            else:
                ibis_type = type_mapper.from_string(type_str, nullable=(nulls == "Y"))
            # Return column names verbatim — SYSCAT stores them in the case they
            # were defined with (uppercase for unquoted DDL, exact case for quoted).
            fields[col_name] = ibis_type

        return sch.Schema(fields)

    @property
    def current_database(self) -> str:
        """Return the current schema."""
        with self._safe_raw_sql("VALUES CURRENT SCHEMA") as cursor:
            result = cursor.fetchone()
            return result[0] if result else None

    def create_table(
        self,
        name: str,
        /,
        obj: ir.Table | pd.DataFrame | pa.Table | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a table.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression, pandas DataFrame or PyArrow table that
            is used to populate the new table.
        schema
            The schema for the new table. Only one of `schema` or `obj` is
            required.
        database
            Name of the schema in which to create the table.
        temp
            Create a `GLOBAL TEMPORARY` table. Its rows are kept until the
            end of the session and it needs a `USER TEMPORARY` tablespace.
        overwrite
            Replace the table if it exists.

        Notes
        -----
        With `overwrite=True` the new table is built under a temporary name
        first, so a failing query leaves the existing table untouched. The
        old table is then dropped and replaced. **The replacement is a new
        table: primary keys, other constraints, indexes, triggers and
        grants on the old table are not carried over.** Recreate them
        after the overwrite, for example with `con.raw_sql("ALTER TABLE ... ADD
        PRIMARY KEY ...")`. Use `con.insert(name, obj, overwrite=True)` to
        replace the rows and keep the table definition.

        Db2 cannot rename a temporary table, so a temporary table is
        overwritten by copying the new table into a re-created one. The
        query is still evaluated before the old table is touched, but the
        old table is gone if that final copy fails.

        Returns
        -------
        Table
            The table that was just created
        """
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")
        if schema is not None:
            schema = ibis.schema(schema)

        properties = _temporary_properties() if temp else []

        if obj is not None:
            table = obj if isinstance(obj, ir.Expr) else ibis.memtable(obj)
            self._run_pre_execute_hooks(table)
            query = self.compiler.to_sqlglot(table)
            column_defs = (schema or table.schema()).to_sqlglot_column_defs(
                self.dialect
            )
        else:
            query = None
            column_defs = schema.to_sqlglot_column_defs(self.dialect)

        quoted = self.compiler.quoted
        # An overwrite is built under a temporary name so that the existing
        # table survives a failure.
        staging = util.gen_name(f"{self.name}_table") if overwrite else name
        staging_table = sg.table(staging, db=database, quoted=quoted)

        def create(table_name):
            with self._safe_raw_sql(
                sge.Create(
                    kind="TABLE",
                    this=sge.Schema(this=table_name, expressions=column_defs),
                    properties=sge.Properties(expressions=properties),
                )
            ):
                pass

        try:
            create(staging_table)
            if query is not None:
                with self._safe_raw_sql(
                    sge.Insert(this=staging_table, expression=query)
                ):
                    pass
        except BaseException:
            if overwrite:
                self.drop_table(staging, database=database, force=True)
            raise

        if overwrite:
            self.drop_table(name, database=database, force=True)
            final_table = sg.table(name, db=database, quoted=quoted)
            if temp:
                # Db2 cannot rename a temporary table (SQL0526N).
                create(final_table)
                with self._safe_raw_sql(
                    sge.Insert(
                        this=final_table,
                        expression=sg.select(sge.Star()).from_(staging_table),
                    )
                ):
                    pass
                self.drop_table(staging, database=database, force=True)
            else:
                new_name = sg.to_identifier(name, quoted=quoted).sql(self.dialect)
                with self._safe_raw_sql(
                    f"RENAME TABLE {staging_table.sql(self.dialect)} TO {new_name}"
                ):
                    pass

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def drop_view(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a view.

        DB2 does not support ``DROP VIEW IF EXISTS``, so when *force* is True
        we check SYSCAT.VIEWS first and skip the DROP if the view is absent.
        """
        full_name = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            self.dialect
        )

        if force:
            cursor = self.con.cursor()
            try:
                if database:
                    cursor.execute(
                        "SELECT COUNT(*) FROM SYSCAT.VIEWS WHERE VIEWNAME = ? AND VIEWSCHEMA = ?",
                        (name, database),
                    )
                else:
                    cursor.execute(
                        "SELECT COUNT(*) FROM SYSCAT.VIEWS WHERE VIEWNAME = ? AND VIEWSCHEMA = CURRENT SCHEMA",
                        (name,),
                    )
                exists = cursor.fetchone()[0] > 0
            finally:
                cursor.close()

            if not exists:
                return

        with self._safe_raw_sql(f"DROP VIEW {full_name}"):
            pass

    def truncate_table(
        self, name: str, /, *, database: str | tuple[str, str] | None = None
    ) -> None:
        """Delete all rows from a table.

        Parameters
        ----------
        name
            Table name
        database
            Schema name, or a `(catalog, schema)` tuple
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)
        full_name = sg.table(
            name, db=db, catalog=catalog, quoted=self.compiler.quoted
        ).sql(self.dialect)
        # Db2 does not accept a bare TRUNCATE TABLE; IMMEDIATE is required, and
        # it must be the first statement in its unit of work.
        with self._safe_raw_sql(f"TRUNCATE TABLE {full_name} IMMEDIATE"):
            pass

    @staticmethod
    def _convert_dataframe_to_rows(df: pd.DataFrame) -> list[tuple]:
        """Convert DataFrame to list of tuples, replacing NaN/NaT/pd.NA with None.

        This is necessary because Db2's ibm_db_dbi driver expects SQL NULL values
        to be represented as Python None.
        """
        import pandas as pd

        df = df.astype(object).where(pd.notnull(df), None)
        return list(df.itertuples(index=False, name=None))

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get schema from a SQL query.

        Parameters
        ----------
        query : str
            SQL query

        Returns
        -------
        sch.Schema
            Query result schema
        """
        import ibm_db
        import ibm_db_dbi

        # Preparing a statement returns its result-set metadata without running
        # it. `conn_handler` is not part of the DB-API 2.0 interface: it is the
        # native handle that ibm_db_dbi.Connection keeps, and the driver offers
        # no other accessor. Verified against ibm_db / ibm_db_dbi 3.2.9.
        #
        # By default the driver defers the prepare until the first execute, so
        # a syntax error or an undefined name in a non-SELECT statement goes
        # unnoticed and the result looks like an empty schema. Turning
        # deferred prepare off makes the server validate the statement now.
        try:
            stmt = ibm_db.prepare(
                self.con.conn_handler,
                query,
                {ibm_db.SQL_ATTR_DEFERRED_PREPARE: 0},
            )
            num_fields = ibm_db.num_fields(stmt)
        except Exception as e:
            # ibm_db raises a bare Exception; raise the DB-API error that
            # cursor.execute() raises for the same bad query.
            raise ibm_db_dbi.ProgrammingError(str(e)) from e
        try:
            fields = {}
            for i in range(num_fields):
                kind = ibm_db.field_type(stmt, i)
                precision = ibm_db.field_precision(stmt, i)
                scale = ibm_db.field_scale(stmt, i)
                nullable = bool(ibm_db.field_nullable(stmt, i))

                if kind in _SIMPLE_FIELD_TYPES:
                    ibis_type = _SIMPLE_FIELD_TYPES[kind](nullable=nullable)
                elif kind == "int":
                    # The driver reports BOOLEAN as a 1-digit int; no other Db2
                    # integer type has precision 1 (SMALLINT is 5, INTEGER is 10).
                    if precision == 1:
                        ibis_type = dt.Boolean(nullable=nullable)
                    elif precision <= 5:
                        ibis_type = dt.Int16(nullable=nullable)
                    else:
                        ibis_type = dt.Int32(nullable=nullable)
                elif kind == "timestamp":
                    ibis_type = dt.Timestamp(
                        scale=min(scale, _MAX_TIMESTAMP_SCALE), nullable=nullable
                    )
                elif kind == "decimal":
                    ibis_type = dt.Decimal(precision, scale, nullable=nullable)
                elif kind == "real":
                    # REAL has 7 digits of precision; DOUBLE and DECFLOAT have more
                    float_type = dt.Float32 if precision <= 7 else dt.Float64
                    ibis_type = float_type(nullable=nullable)
                elif kind == "string":
                    ibis_type = dt.String(length=precision, nullable=nullable)
                else:
                    raise exc.UnsupportedBackendType(
                        f"Unsupported Db2 result column type: {kind!r}"
                    )
                fields[ibm_db.field_name(stmt, i)] = ibis_type
            return sch.Schema(fields)
        finally:
            ibm_db.free_stmt(stmt)

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        """Register an in-memory table as a GLOBAL TEMPORARY table.

        Parameters
        ----------
        op : ops.InMemoryTable
            In-memory table operation
        """
        schema = op.schema
        if null_columns := schema.null_fields:
            raise exc.IbisTypeError(
                f"{self.name} cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        name = op.name
        create_stmt = sge.Create(
            kind="TABLE",
            this=sge.Schema(
                this=sg.to_identifier(name, quoted=self.compiler.quoted),
                expressions=schema.to_sqlglot_column_defs(self.dialect),
            ),
            properties=sge.Properties(expressions=_temporary_properties()),
        )
        insert_stmt = self._build_insert_template(name, schema=schema, columns=True)

        data = op.data.to_frame()
        with self._safe_raw_sql(create_stmt):
            pass
        cursor = self.con.cursor()
        try:
            for start, end in util.chunks(len(data), chunk_size=1000):
                cursor.executemany(
                    insert_stmt, self._convert_dataframe_to_rows(data.iloc[start:end])
                )
        finally:
            cursor.close()
