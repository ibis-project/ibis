"""The SingleStoreDB backend."""

# ruff: noqa: BLE001, S110, S608, PERF203, SIM105 - Performance optimization methods require comprehensive exception handling

from __future__ import annotations

import contextlib
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import unquote_plus

import sqlglot as sg
import sqlglot.expressions as sge
from singlestoredb.connection import build_params

import ibis.common.exceptions as com
import ibis.expr.schema as sch
from ibis import util
from ibis.backends import (
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
    SupportsTempTables,
)
from ibis.backends.sql import SQLBackend
from ibis.backends.sql.compilers.singlestoredb import compiler

if TYPE_CHECKING:
    from collections.abc import Generator
    from urllib.parse import ParseResult


class Backend(
    SupportsTempTables,
    SQLBackend,
    CanCreateDatabase,
    HasCurrentDatabase,
    PyArrowExampleLoader,
):
    name = "singlestoredb"
    supports_create_or_replace = False
    supports_temporary_tables = True

    _connect_string_template = (
        "singlestoredb://{{user}}:{{password}}@{{host}}:{{port}}/{{database}}"
    )

    compiler = compiler

    def _fetch_from_cursor(self, cursor, schema):
        """Fetch data from cursor using SingleStoreDB-specific data converter."""
        import pandas as pd

        from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            cursor.close()
            raise

        return SingleStoreDBPandasData.convert_table(df, schema)

    @util.experimental
    def to_pyarrow_batches(
        self,
        expr,
        /,
        *,
        params=None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ):
        """Convert expression to PyArrow record batches.

        This method ensures proper data type conversion, particularly for
        boolean values that come from TINYINT(1) columns.
        """
        import pyarrow as pa

        self._run_pre_execute_hooks(expr)

        # Get the expected schema and compile the query
        schema = expr.as_table().schema()
        sql = self.compile(expr, limit=limit, params=params)

        # Fetch data using our converter
        with self.begin() as cursor:
            cursor.execute(sql)
            df = self._fetch_from_cursor(cursor, schema)

        # Convert to PyArrow table with proper type conversion
        table = pa.Table.from_pandas(
            df, schema=schema.to_pyarrow(), preserve_index=False
        )
        return table.to_reader(max_chunksize=chunk_size)

    @property
    def con(self):
        """Return the database connection for compatibility with base class."""
        return self._client

    def _post_connect(self) -> None:
        with self.con.cursor() as cur:
            try:
                cur.execute("SET @@session.time_zone = 'UTC'")
            except Exception as e:
                warnings.warn(f"Unable to set session timezone to UTC: {e}")

    @property
    def show(self) -> Any:
        """Access to SHOW commands on the server."""
        return self.con.show

    @property
    def globals(self) -> Any:
        """Accessor for global variables in the server."""
        return self.con.globals

    @property
    def locals(self) -> Any:
        """Accessor for local variables in the server."""
        return self.con.locals

    @property
    def cluster_globals(self) -> Any:
        """Accessor for cluster global variables in the server."""
        return self.con.cluster_globals

    @property
    def cluster_locals(self) -> Any:
        """Accessor for cluster local variables in the server."""
        return self.con.cluster_locals

    @property
    def vars(self) -> Any:
        """Accessor for variables in the server."""
        return self.con.vars

    @property
    def cluster_vars(self) -> Any:
        """Accessor for cluster variables in the server."""
        return self.con.cluster_vars

    @property
    def current_database(self) -> str:
        """Return the current database name."""
        with self.begin() as cur:
            cur.execute("SELECT DATABASE()")
            (database,) = cur.fetchone()
        return database

    @property
    def database(self) -> str:
        """Return the current database name (alias for current_database)."""
        return self.current_database

    @property
    def dialect(self) -> str:
        """Return the SQLGlot dialect name."""
        return "singlestore"

    @classmethod
    def _from_url(cls, url: ParseResult, **kwargs) -> Backend:
        """Create a SingleStoreDB backend from a connection URL."""
        database = url.path[1:] if url.path and len(url.path) > 1 else ""

        backend = cls()
        backend.do_connect(
            host=url.hostname or "localhost",
            port=url.port or 3306,
            user=url.username or "root",
            password=unquote_plus(url.password or ""),
            database=database,
            **kwargs,
        )
        return backend

    def create_database(
        self,
        name: str,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new database in SingleStoreDB.

        Parameters
        ----------
        name
            Name of the database to create.
        force
            If True, create the database with IF NOT EXISTS clause.
            If False (default), raise an error if the database already exists.
        **kwargs
            Additional keyword arguments (for compatibility with base class).
        """
        if_not_exists = "IF NOT EXISTS " * force
        with self.begin() as cur:
            cur.execute(f"CREATE DATABASE {if_not_exists}{name}")

    def drop_database(
        self,
        name: str,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """Drop a database in SingleStoreDB.

        Parameters
        ----------
        name
            Name of the database to drop.
        force
            If True, drop the database with IF EXISTS clause.
            If False (default), raise an error if the database does not exist.
        **kwargs
            Additional keyword arguments (for compatibility with base class).
        """
        if_exists = "IF EXISTS " * force
        with self.begin() as cur:
            cur.execute(f"DROP DATABASE {if_exists}{name}")

    def list_databases(self, *, like: str | None = None) -> list[str]:
        """Return the list of databases.

        Parameters
        ----------
        like
            A pattern in Python's regex format to filter returned database names.

        Returns
        -------
        list[str]
            The database names that match the pattern `like`.
        """
        # In SingleStoreDB, "database" is the preferred terminology
        # though "schema" is also supported for MySQL compatibility
        query = "SHOW DATABASES"

        with self.begin() as cur:
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]

    def list_tables(
        self,
        like: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> list[str]:
        """List tables in SingleStoreDB database.

        Parameters
        ----------
        like
            SQL LIKE pattern to filter table names.
            Use '%' as wildcard, e.g., 'user_%' for tables starting with 'user_'
        database
            Database to list tables from. If None, uses current database.
            Can be a string database name or tuple (catalog, database)

        Returns
        -------
        list[str]
            List of table names in the specified database

        Examples
        --------
        >>> con.list_tables()
        ['users', 'orders', 'products']
        >>> con.list_tables(like="user_%")
        ['users', 'user_profiles']
        >>> con.list_tables(database="other_db")
        ['table1', 'table2']
        """
        from operator import itemgetter

        import sqlglot as sg
        import sqlglot.expressions as sge

        from ibis.backends.sql.compilers.base import TRUE, C

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
            conditions = [C.table_schema.eq(sge.convert(table_loc.sql("singlestore")))]

        col = "table_name"
        sql = (
            sg.select(col)
            .from_(sg.table("tables", db="information_schema"))
            .distinct()
            .where(*conditions)
            .sql("singlestore")
        )

        with self.begin() as cur:
            cur.execute(sql)
            out = cur.fetchall()

        return self._filter_with_like(map(itemgetter(0), out), like)

    def get_schema(
        self, name: str, *, catalog: str | None = None, database: str | None = None
    ) -> sch.Schema:
        """Get schema for a table in SingleStoreDB.

        Parameters
        ----------
        name
            Table name to get schema for
        catalog
            Catalog name (usually not used in SingleStoreDB)
        database
            Database name. If None, uses current database

        Returns
        -------
        Schema
            Ibis schema object with column names and types

        Examples
        --------
        >>> schema = con.get_schema("users")
        >>> print(schema)
        Schema:
          id: int64
          name: string
          email: string
          created_at: timestamp
        """
        import sqlglot as sg
        import sqlglot.expressions as sge

        table = sg.table(
            name, db=database, catalog=catalog, quoted=self.compiler.quoted
        ).sql("singlestore")  # Use singlestore dialect

        with self.begin() as cur:
            try:
                cur.execute(sge.Describe(this=table).sql("singlestore"))
            except Exception as e:
                # Handle table not found
                if "doesn't exist" in str(e) or "Table" in str(e):
                    raise com.TableNotFound(name) from e
                raise
            else:
                result = cur.fetchall()

        type_mapper = self.compiler.type_mapper
        fields = {
            name: type_mapper.from_string(type_string, nullable=is_nullable == "YES")
            for name, type_string, is_nullable, *_ in result
        }

        return sch.Schema(fields)

    @contextlib.contextmanager
    def begin(self) -> Generator[Any, None, None]:
        """Begin a transaction context for executing SQL commands.

        This method provides a cursor context manager that automatically
        handles transaction lifecycle including rollback on exceptions
        and proper cleanup.

        Yields
        ------
        Cursor
            SingleStoreDB cursor for executing SQL commands

        Examples
        --------
        >>> with con.begin() as cur:
        ...     cur.execute("SELECT COUNT(*) FROM users")
        ...     result = cur.fetchone()
        """
        con = self._client
        cur = con.cursor()
        autocommit = getattr(con, "autocommit", True)

        if not autocommit:
            con.begin()

        try:
            yield cur
        except Exception:
            if not autocommit and hasattr(con, "rollback"):
                con.rollback()
            raise
        else:
            if not autocommit and hasattr(con, "commit"):
                con.commit()
        finally:
            cur.close()

    def create_table(
        self,
        name: str,
        /,
        obj: Any | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a table in SingleStoreDB."""
        import sqlglot as sg
        import sqlglot.expressions as sge

        import ibis
        import ibis.expr.operations as ops
        import ibis.expr.types as ir
        from ibis import util

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
        # Convert SQLGlot object to SQL string before execution
        with self.begin() as cur:
            cur.execute(create_stmt.sql(dialect))
            if query is not None:
                cur.execute(sge.Insert(this=table_expr, expression=query).sql(dialect))

            if overwrite:
                cur.execute(sge.Drop(kind="TABLE", this=this, exists=True).sql(dialect))
                # Use ALTER TABLE ... RENAME TO syntax supported by SingleStoreDB
                # Extract just the table name (removing catalog/database prefixes and quotes)
                temp_table_name = temp_name
                if quoted:
                    temp_table_name = f"`{temp_name}`"
                final_table_name = name
                if quoted:
                    final_table_name = f"`{name}`"

                rename_sql = (
                    f"ALTER TABLE {temp_table_name} RENAME TO {final_table_name}"
                )
                cur.execute(rename_sql)

        if schema is None:
            return self.table(name, database=database)

        # preserve the input schema if it was provided
        return ops.DatabaseTable(
            name, schema=schema, source=self, namespace=ops.Namespace(database=database)
        ).to_expr()

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table from the database.

        Parameters
        ----------
        name
            Table name to drop
        database
            Database name
        force
            Use IF EXISTS clause when dropping
        """
        drop_stmt = sge.Drop(
            kind="TABLE",
            this=sg.table(name, db=database, quoted=self.compiler.quoted),
            exists=force,
        )
        # Convert SQLGlot object to SQL string before execution
        with self.begin() as cur:
            cur.execute(drop_stmt.sql(self.dialect))

    def _register_in_memory_table(self, op: Any) -> None:
        """Register an in-memory table in SingleStoreDB."""
        import sqlglot as sg
        import sqlglot.expressions as sge

        schema = op.schema
        if null_columns := schema.null_fields:
            raise com.IbisTypeError(
                "SingleStoreDB cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # Check for unsupported complex types
        for field_name, field_type in schema.items():
            if field_type.is_array() or field_type.is_struct() or field_type.is_map():
                raise com.UnsupportedBackendType(
                    f"SingleStoreDB does not support complex types like arrays, structs, or maps. "
                    f"Column '{field_name}' has type '{field_type}'"
                )

        name = op.name
        quoted = self.compiler.quoted
        dialect = self.dialect

        create_stmt = sg.exp.Create(
            kind="TABLE",
            this=sg.exp.Schema(
                this=sg.to_identifier(name, quoted=quoted),
                expressions=schema.to_sqlglot_column_defs(dialect),
            ),
            properties=sg.exp.Properties(expressions=[sge.TemporaryProperty()]),
        )
        create_stmt_sql = create_stmt.sql(dialect)

        df = op.data.to_frame()
        df = df.replace(float("nan"), None)

        # Fix: Convert itertuples result to list for SingleStoreDB compatibility
        data = list(df.itertuples(index=False))
        sql = self._build_insert_template(
            name, schema=schema, columns=True, placeholder="%s"
        )
        with self.begin() as cur:
            cur.execute(create_stmt_sql)

            if not df.empty:
                cur.executemany(sql, data)

    # TODO(kszucs): should make it an abstract method or remove the use of it
    # from .execute()
    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with self.raw_sql(*args, **kwargs) as result:
            yield result

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        with contextlib.suppress(AttributeError):
            query = query.sql(dialect=self.dialect)

        con = self.con
        autocommit = con.get_autocommit()

        cursor = con.cursor()

        if not autocommit:
            con.begin()

        try:
            cursor.execute(query, **kwargs)
        except Exception:
            if not autocommit:
                con.rollback()
            cursor.close()
            raise
        else:
            if not autocommit:
                con.commit()
            return cursor

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Get the schema of a query result."""
        import sqlglot as sg
        from sqlglot import expressions as sge

        from ibis import util
        from ibis.backends.singlestoredb.converter import SingleStoreDBPandasData
        from ibis.backends.singlestoredb.datatypes import _type_from_cursor_info

        # Try to parse with different dialects to see if it's a dialect issue
        try:
            # First try with SingleStore dialect
            parsed = sg.parse_one(query, dialect=self.dialect)
        except Exception:
            try:
                # Fallback to MySQL dialect which SingleStore is based on
                parsed = sg.parse_one(query, dialect="mysql")
            except Exception:
                # Last resort - use generic SQL dialect
                parsed = sg.parse_one(query, dialect="")

        # Use SQLGlot to properly construct the query
        sql = (
            sg.select(sge.Star())
            .from_(
                parsed.subquery(
                    sg.to_identifier(
                        util.gen_name("query_schema"), quoted=self.compiler.quoted
                    )
                )
            )
            .limit(0)
            .sql(self.dialect)
        )

        with self.begin() as cur:
            cur.execute(sql)
            description = cur.description

        names = []
        ibis_types = []
        for col_info in description:
            name = col_info[0]
            names.append(name)

            # Use the detailed cursor info for type conversion
            if len(col_info) >= 7:
                # Full cursor description available
                # SingleStoreDB uses 4-byte character encoding by default
                ibis_type = _type_from_cursor_info(
                    flags=col_info[7] if len(col_info) > 7 else 0,
                    type_code=col_info[1],
                    field_length=col_info[3],
                    scale=col_info[5],
                    multi_byte_maximum_length=4,  # Use 4 for SingleStoreDB's UTF8MB4 encoding
                )
            else:
                # Fallback for limited cursor info
                typename = SingleStoreDBPandasData._get_type_name(col_info[1])
                ibis_type = SingleStoreDBPandasData.convert_SingleStoreDB_type(typename)

            ibis_types.append(ibis_type)

        return sch.Schema(dict(zip(names, ibis_types)))

    @property
    def version(self) -> str:
        """Return the version of the SingleStoreDB server.

        Returns
        -------
        str
            The version string of the connected SingleStoreDB server.

        Examples
        --------
        >>> con.version  # doctest: +SKIP
        '8.7.10-bf633c1a54'
        """
        with self.begin() as cur:
            cur.execute("SELECT @@version")
            (version_string,) = cur.fetchone()
        return version_string

    def create_columnstore_table(
        self,
        name: str,
        /,
        obj: Any | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        shard_key: str | None = None,
    ):
        """Create a columnstore table in SingleStore.

        Parameters
        ----------
        name
            Table name to create
        obj
            Data to insert into the table
        schema
            Table schema
        database
            Database to create table in
        temp
            Create temporary table
        overwrite
            Overwrite existing table
        shard_key
            Shard key column for distributed storage

        Returns
        -------
        Table
            The created table expression
        """
        # Create the table using standard method first
        table_expr = self.create_table(
            name, obj, schema=schema, database=database, temp=temp, overwrite=overwrite
        )

        # If this SingleStore version supports columnstore, we would add:
        # ALTER TABLE to convert to columnstore format
        # For now, just return the standard table since our test instance
        # doesn't support the USING COLUMNSTORE syntax

        return table_expr

    def create_reference_table(
        self,
        name: str,
        /,
        obj: Any | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ):
        """Create a reference table in SingleStore.

        Reference tables are replicated across all nodes for fast lookups.

        Parameters
        ----------
        name
            Table name to create
        obj
            Data to insert into the table
        schema
            Table schema
        database
            Database to create table in
        temp
            Create temporary table
        overwrite
            Overwrite existing table

        Returns
        -------
        Table
            The created table expression
        """
        # For reference tables, we create a regular table
        # In full SingleStore, this would include REFERENCE TABLE syntax
        return self.create_table(
            name, obj, schema=schema, database=database, temp=temp, overwrite=overwrite
        )

    def execute_with_hint(self, query: str, hint: str) -> Any:
        """Execute a query with SingleStore-specific optimization hints.

        Parameters
        ----------
        query
            SQL query to execute
        hint
            Optimization hint (e.g., 'MEMORY', 'USE_COLUMNSTORE_STRATEGY')

        Returns
        -------
        Results from query execution
        """
        # Add hint to query
        hinted_query = (
            f"SELECT /*+ {hint} */" + query[6:]
            if query.strip().upper().startswith("SELECT")
            else query
        )

        with self.begin() as cur:
            cur.execute(hinted_query)
            return cur.fetchall()

    def get_partition_info(self, table_name: str) -> list[dict]:
        """Get partition information for a SingleStore table.

        Parameters
        ----------
        table_name
            Name of the table to get partition info for

        Returns
        -------
        list[dict]
            List of partition information dictionaries
        """
        try:
            with self.begin() as cur:
                # Use parameterized query to avoid SQL injection
                cur.execute(
                    """
                    SELECT
                        PARTITION_ORDINAL_POSITION as position,
                        PARTITION_METHOD as method,
                        PARTITION_EXPRESSION as expression
                    FROM INFORMATION_SCHEMA.PARTITIONS
                    WHERE TABLE_NAME = %s
                    AND TABLE_SCHEMA = DATABASE()
                """,
                    (table_name,),
                )
                results = cur.fetchall()

                return [
                    {"position": row[0], "method": row[1], "expression": row[2]}
                    for row in results
                ]
        except (KeyError, IndexError, ValueError):
            # Fallback if information_schema doesn't have expected columns
            return []

    def get_cluster_info(self) -> dict:
        """Get SingleStore cluster information.

        Returns
        -------
        dict
            Cluster information including leaves and partitions
        """
        cluster_info = {"leaves": [], "partitions": 0, "version": self.version}

        try:
            with self.begin() as cur:
                # Get leaf node information
                cur.execute("SHOW LEAVES")
                leaves = cur.fetchall()
                cluster_info["leaves"] = [
                    {
                        "host": leaf[0],
                        "port": leaf[1],
                        "state": leaf[5] if len(leaf) > 5 else "unknown",
                    }
                    for leaf in leaves
                ]

                # Get partition count
                cur.execute("SHOW PARTITIONS")
                partitions = cur.fetchall()
                cluster_info["partitions"] = len(partitions)

        except (KeyError, IndexError, ValueError, OSError) as e:
            cluster_info["error"] = str(e)

        return cluster_info

    def explain_query(self, query: str) -> dict:
        """Get execution plan for a query.

        Parameters
        ----------
        query
            SQL query to analyze

        Returns
        -------
        dict
            Query execution plan information
        """
        try:
            with self.begin() as cur:
                # Get detailed execution plan
                cur.execute(f"EXPLAIN EXTENDED {query}")
                plan_rows = cur.fetchall()

                # Get JSON format plan if available
                json_plan = None
                try:
                    cur.execute(f"EXPLAIN FORMAT=JSON {query}")
                    json_result = cur.fetchone()
                    if json_result:
                        import json

                        json_plan = json.loads(json_result[0])
                except Exception:
                    # JSON format may not be available in all versions
                    pass

                return {
                    "text_plan": [
                        {
                            "id": row[0],
                            "select_type": row[1],
                            "table": row[2],
                            "partitions": row[3],
                            "type": row[4],
                            "possible_keys": row[5],
                            "key": row[6],
                            "key_len": row[7],
                            "ref": row[8],
                            "rows": row[9],
                            "filtered": row[10] if len(row) > 10 else None,
                            "extra": row[11]
                            if len(row) > 11
                            else row[10]
                            if len(row) > 10
                            else None,
                        }
                        for row in plan_rows
                    ],
                    "json_plan": json_plan,
                    "query": query,
                }
        except Exception as e:
            return {"error": str(e), "query": query}

    def analyze_query_performance(self, query: str) -> dict:
        """Analyze query performance characteristics.

        Parameters
        ----------
        query
            SQL query to analyze

        Returns
        -------
        dict
            Performance analysis including execution plan, statistics, and recommendations
        """
        import time

        analysis = {
            "query": query,
            "execution_plan": self.explain_query(query),
            "timing": {},
            "statistics": {},
            "recommendations": [],
        }

        try:
            with self.begin() as cur:
                # Get query timing
                start_time = time.time()
                cur.execute(query)
                results = cur.fetchall()
                end_time = time.time()

                analysis["timing"] = {
                    "execution_time": end_time - start_time,
                    "rows_returned": len(results),
                }

                # Get query statistics if available
                try:
                    cur.execute("SHOW SESSION STATUS LIKE 'Handler_%'")
                    stats = cur.fetchall()
                    analysis["statistics"] = {row[0]: int(row[1]) for row in stats}
                except Exception:
                    pass

                # Generate basic recommendations
                plan = analysis["execution_plan"]
                if "text_plan" in plan:
                    for step in plan["text_plan"]:
                        # Check for full table scans
                        if step.get("type") == "ALL":
                            analysis["recommendations"].append(
                                {
                                    "type": "INDEX_RECOMMENDATION",
                                    "message": f"Consider adding an index to table '{step['table']}' to avoid full table scan",
                                    "table": step["table"],
                                    "severity": "medium",
                                }
                            )

                        # Check for temporary table usage
                        if (
                            step.get("extra")
                            and "temporary" in str(step["extra"]).lower()
                        ):
                            analysis["recommendations"].append(
                                {
                                    "type": "MEMORY_OPTIMIZATION",
                                    "message": "Query uses temporary tables which may impact performance",
                                    "severity": "low",
                                }
                            )

                        # Check for filesort
                        if (
                            step.get("extra")
                            and "filesort" in str(step["extra"]).lower()
                        ):
                            analysis["recommendations"].append(
                                {
                                    "type": "SORT_OPTIMIZATION",
                                    "message": "Query requires filesort - consider adding appropriate index for ORDER BY",
                                    "severity": "medium",
                                }
                            )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def get_table_statistics(
        self, table_name: str, database: Optional[str] = None
    ) -> dict:
        """Get detailed statistics for a table.

        Parameters
        ----------
        table_name
            Name of the table
        database
            Database name (optional, uses current database if None)

        Returns
        -------
        dict
            Table statistics including row count, size, and index information
        """
        if database is None:
            database = self.current_database

        stats = {
            "table_name": table_name,
            "database": database,
            "row_count": 0,
            "data_size": 0,
            "index_size": 0,
            "indexes": [],
            "columns": [],
        }

        try:
            with self.begin() as cur:
                # Get basic table statistics
                cur.execute(
                    """
                    SELECT
                        TABLE_ROWS,
                        DATA_LENGTH,
                        INDEX_LENGTH,
                        AUTO_INCREMENT,
                        CREATE_TIME,
                        UPDATE_TIME
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
                """,
                    (table_name, database),
                )

                result = cur.fetchone()
                if result:
                    stats.update(
                        {
                            "row_count": result[0] or 0,
                            "data_size": result[1] or 0,
                            "index_size": result[2] or 0,
                            "auto_increment": result[3],
                            "created": result[4],
                            "updated": result[5],
                        }
                    )

                # Get index information
                cur.execute(
                    """
                    SELECT
                        INDEX_NAME,
                        COLUMN_NAME,
                        SEQ_IN_INDEX,
                        NON_UNIQUE,
                        CARDINALITY
                    FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
                    ORDER BY INDEX_NAME, SEQ_IN_INDEX
                """,
                    (table_name, database),
                )

                index_rows = cur.fetchall()
                indexes_dict = {}
                for row in index_rows:
                    idx_name = row[0]
                    if idx_name not in indexes_dict:
                        indexes_dict[idx_name] = {
                            "name": idx_name,
                            "columns": [],
                            "unique": row[3] == 0,
                            "cardinality": row[4] or 0,
                        }
                    indexes_dict[idx_name]["columns"].append(row[1])

                stats["indexes"] = list(indexes_dict.values())

                # Get column information
                cur.execute(
                    """
                    SELECT
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        COLUMN_DEFAULT,
                        COLUMN_KEY
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
                    ORDER BY ORDINAL_POSITION
                """,
                    (table_name, database),
                )

                columns = cur.fetchall()
                stats["columns"] = [
                    {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == "YES",
                        "default": col[3],
                        "key": col[4],
                    }
                    for col in columns
                ]

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def suggest_indexes(self, query: str) -> list[dict]:
        """Suggest indexes that could improve query performance.

        Parameters
        ----------
        query
            SQL query to analyze for index suggestions

        Returns
        -------
        list[dict]
            List of index suggestions with rationale
        """
        suggestions = []

        try:
            # Analyze the execution plan
            plan = self.explain_query(query)

            if "text_plan" in plan:
                for step in plan["text_plan"]:
                    table = step.get("table")
                    if not table or table.startswith("derived"):
                        continue

                    # Suggest index for full table scans
                    if step.get("type") == "ALL":
                        suggestions.append(
                            {
                                "table": table,
                                "type": "COVERING_INDEX",
                                "rationale": "Full table scan detected",
                                "priority": "high",
                                "estimated_benefit": "high",
                            }
                        )

                    # Suggest index for range scans without optimal key
                    elif step.get("type") in ["range", "ref"] and not step.get("key"):
                        suggestions.append(
                            {
                                "table": table,
                                "type": "FILTERED_INDEX",
                                "rationale": "Range/ref scan without optimal index",
                                "priority": "medium",
                                "estimated_benefit": "medium",
                            }
                        )

                    # Check for join conditions that could benefit from indexes
                    if (
                        step.get("extra")
                        and "join buffer" in str(step.get("extra", "")).lower()
                    ):
                        suggestions.append(
                            {
                                "table": table,
                                "type": "JOIN_INDEX",
                                "rationale": "Join buffer detected - join condition may benefit from index",
                                "priority": "medium",
                                "estimated_benefit": "medium",
                            }
                        )

            # Parse query for additional suggestions
            query_upper = query.upper()

            # Suggest index for ORDER BY columns
            if "ORDER BY" in query_upper:
                suggestions.append(
                    {
                        "type": "SORT_INDEX",
                        "rationale": "ORDER BY clause detected",
                        "priority": "low",
                        "estimated_benefit": "low",
                        "note": "Consider index on ORDER BY columns to avoid filesort",
                    }
                )

            # Suggest index for GROUP BY columns
            if "GROUP BY" in query_upper:
                suggestions.append(
                    {
                        "type": "GROUP_INDEX",
                        "rationale": "GROUP BY clause detected",
                        "priority": "medium",
                        "estimated_benefit": "medium",
                        "note": "Consider index on GROUP BY columns for faster aggregation",
                    }
                )

        except Exception as e:
            suggestions.append(
                {
                    "error": str(e),
                    "type": "ERROR",
                    "rationale": "Failed to analyze query for index suggestions",
                }
            )

        return suggestions

    def optimize_query_with_hints(
        self, query: str, optimization_level: str = "balanced"
    ) -> str:
        """Optimize a query by adding SingleStore-specific hints.

        Parameters
        ----------
        query
            Original SQL query
        optimization_level
            Optimization level: 'speed', 'memory', 'balanced'

        Returns
        -------
        str
            Optimized query with hints
        """
        hints = []

        if optimization_level == "speed":
            hints.extend(
                [
                    "USE_COLUMNSTORE_STRATEGY",
                    "MEMORY",
                    "USE_HASH_JOIN",
                ]
            )
        elif optimization_level == "memory":
            hints.extend(
                [
                    "USE_NESTED_LOOP_JOIN",
                    "NO_MERGE_SORT",
                ]
            )
        else:  # balanced
            hints.extend(
                [
                    "ADAPTIVE_JOIN",
                ]
            )

        if hints and query.strip().upper().startswith("SELECT"):
            hint_str = ", ".join(hints)
            return f"SELECT /*+ {hint_str} */" + query[6:]

        return query

    def create_index(
        self,
        table_name: str,
        columns: list[str] | str,
        index_name: Optional[str] = None,
        unique: bool = False,
        index_type: str = "BTREE",
    ) -> None:
        """Create an index on a table.

        Parameters
        ----------
        table_name
            Name of the table
        columns
            Column name(s) to index
        index_name
            Name for the index (auto-generated if None)
        unique
            Whether to create a unique index
        index_type
            Type of index (BTREE, HASH, etc.)
        """
        if isinstance(columns, str):
            columns = [columns]

        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(columns)}"

        columns_str = ", ".join(f"`{col}`" for col in columns)
        unique_str = "UNIQUE " if unique else ""

        sql = f"CREATE {unique_str}INDEX `{index_name}` ON `{table_name}` ({columns_str}) USING {index_type}"

        with self.begin() as cur:
            cur.execute(sql)

    def drop_index(self, table_name: str, index_name: str) -> None:
        """Drop an index from a table.

        Parameters
        ----------
        table_name
            Name of the table
        index_name
            Name of the index to drop
        """
        sql = f"DROP INDEX `{index_name}` ON `{table_name}`"
        with self.begin() as cur:
            cur.execute(sql)

    def analyze_index_usage(self, table_name: Optional[str] = None) -> dict:
        """Analyze index usage statistics.

        Parameters
        ----------
        table_name
            Specific table to analyze (None for all tables)

        Returns
        -------
        dict
            Index usage statistics and recommendations
        """
        analysis = {
            "unused_indexes": [],
            "low_selectivity_indexes": [],
            "duplicate_indexes": [],
            "recommendations": [],
        }

        try:
            with self.begin() as cur:
                # Base query for index statistics
                base_query = """
                    SELECT
                        s.TABLE_SCHEMA,
                        s.TABLE_NAME,
                        s.INDEX_NAME,
                        s.COLUMN_NAME,
                        s.CARDINALITY,
                        s.NON_UNIQUE,
                        t.TABLE_ROWS
                    FROM INFORMATION_SCHEMA.STATISTICS s
                    JOIN INFORMATION_SCHEMA.TABLES t
                        ON s.TABLE_SCHEMA = t.TABLE_SCHEMA
                        AND s.TABLE_NAME = t.TABLE_NAME
                    WHERE s.TABLE_SCHEMA = DATABASE()
                """

                params = []
                if table_name:
                    base_query += " AND s.TABLE_NAME = %s"
                    params.append(table_name)

                base_query += " ORDER BY s.TABLE_NAME, s.INDEX_NAME, s.SEQ_IN_INDEX"

                cur.execute(base_query, params)
                stats = cur.fetchall()

                # Group indexes by table and name
                indexes = {}
                for row in stats:
                    (
                        schema,
                        tbl,
                        idx_name,
                        col_name,
                        cardinality,
                        non_unique,
                        table_rows,
                    ) = row

                    key = (schema, tbl, idx_name)
                    if key not in indexes:
                        indexes[key] = {
                            "schema": schema,
                            "table": tbl,
                            "name": idx_name,
                            "columns": [],
                            "cardinality": cardinality or 0,
                            "unique": non_unique == 0,
                            "table_rows": table_rows or 0,
                        }
                    indexes[key]["columns"].append(col_name)

                # Analyze each index
                for (_schema, tbl, idx_name), idx_info in indexes.items():
                    if idx_name == "PRIMARY":
                        continue  # Skip primary keys

                    # Check for low selectivity
                    if idx_info["table_rows"] > 0:
                        selectivity = idx_info["cardinality"] / idx_info["table_rows"]
                        if selectivity < 0.1:  # Less than 10% selectivity
                            analysis["low_selectivity_indexes"].append(
                                {
                                    "table": tbl,
                                    "index": idx_name,
                                    "selectivity": selectivity,
                                    "cardinality": idx_info["cardinality"],
                                    "table_rows": idx_info["table_rows"],
                                }
                            )

                # Check for duplicate indexes (simplified)
                table_indexes = {}
                for (_schema, tbl, idx_name), idx_info in indexes.items():
                    if tbl not in table_indexes:
                        table_indexes[tbl] = []
                    table_indexes[tbl].append(
                        {
                            "name": idx_name,
                            "columns": tuple(idx_info["columns"]),
                            "unique": idx_info["unique"],
                        }
                    )

                for tbl, tbl_indexes in table_indexes.items():
                    for i, idx1 in enumerate(tbl_indexes):
                        for idx2 in tbl_indexes[i + 1 :]:
                            # Check for exact duplicates
                            if (
                                idx1["columns"] == idx2["columns"]
                                and idx1["unique"] == idx2["unique"]
                            ):
                                analysis["duplicate_indexes"].append(
                                    {
                                        "table": tbl,
                                        "index1": idx1["name"],
                                        "index2": idx2["name"],
                                        "columns": list(idx1["columns"]),
                                    }
                                )

                # Generate recommendations
                if analysis["low_selectivity_indexes"]:
                    analysis["recommendations"].append(
                        {
                            "type": "REMOVE_LOW_SELECTIVITY",
                            "message": f"Consider removing {len(analysis['low_selectivity_indexes'])} low-selectivity indexes",
                            "priority": "medium",
                        }
                    )

                if analysis["duplicate_indexes"]:
                    analysis["recommendations"].append(
                        {
                            "type": "REMOVE_DUPLICATES",
                            "message": f"Remove {len(analysis['duplicate_indexes'])} duplicate indexes",
                            "priority": "high",
                        }
                    )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def auto_optimize_indexes(self, table_name: str, dry_run: bool = True) -> dict:
        """Automatically optimize indexes for a table.

        Parameters
        ----------
        table_name
            Name of the table to optimize
        dry_run
            If True, only return recommendations without making changes

        Returns
        -------
        dict
            Optimization actions taken or recommended
        """
        actions = {
            "analyzed": table_name,
            "recommendations": [],
            "executed": [],
            "errors": [],
        }

        try:
            # Get index analysis
            index_analysis = self.analyze_index_usage(table_name)

            # Remove duplicate indexes
            for dup in index_analysis.get("duplicate_indexes", []):
                if dup["table"] == table_name:
                    action = {
                        "type": "DROP_DUPLICATE",
                        "sql": f"DROP INDEX `{dup['index2']}` ON `{table_name}`",
                        "rationale": f"Duplicate of {dup['index1']}",
                    }
                    actions["recommendations"].append(action)

                    if not dry_run:
                        try:
                            self.drop_index(table_name, dup["index2"])
                            actions["executed"].append(action)
                        except Exception as e:
                            actions["errors"].append(
                                f"Failed to drop {dup['index2']}: {e}"
                            )

            # Remove low selectivity indexes (with caution)
            for low_sel in index_analysis.get("low_selectivity_indexes", []):
                if low_sel["table"] == table_name and low_sel["selectivity"] < 0.05:
                    action = {
                        "type": "DROP_LOW_SELECTIVITY",
                        "sql": f"DROP INDEX `{low_sel['index']}` ON `{table_name}`",
                        "rationale": f"Very low selectivity: {low_sel['selectivity']:.2%}",
                        "warning": "Verify this index is not used by critical queries before dropping",
                    }
                    actions["recommendations"].append(action)

                    # Only execute if explicitly not in dry run and selectivity is very low
                    if not dry_run and low_sel["selectivity"] < 0.01:
                        actions["recommendations"][-1]["note"] = (
                            "Not auto-executed due to safety - review manually"
                        )

        except Exception as e:
            actions["errors"].append(str(e))

        return actions

    def optimize_for_distributed_execution(
        self, query: str, shard_key: Optional[str] = None
    ) -> str:
        """Optimize query for distributed execution in SingleStore.

        Parameters
        ----------
        query
            SQL query to optimize
        shard_key
            Shard key column for optimization hints

        Returns
        -------
        str
            Optimized query for distributed execution
        """
        hints = []

        # Add distributed execution hints
        if "JOIN" in query.upper():
            if shard_key:
                hints.append("BROADCAST_JOIN")  # For small tables
            hints.append("USE_HASH_JOIN")  # Generally better for distributed joins

        # Optimize aggregations for distributed execution
        if "GROUP BY" in query.upper():
            hints.append("USE_DISTRIBUTED_GROUP_BY")

        # Add partitioning hints for large scans
        if "WHERE" not in query.upper():
            hints.append("PARALLEL_EXECUTION")

        if hints and query.strip().upper().startswith("SELECT"):
            hint_str = ", ".join(hints)
            return f"SELECT /*+ {hint_str} */" + query[6:]

        return query

    def get_shard_distribution(
        self, table_name: str, shard_key: Optional[str] = None
    ) -> dict:
        """Analyze how data is distributed across shards.

        Parameters
        ----------
        table_name
            Name of the table to analyze
        shard_key
            Shard key column (if known)

        Returns
        -------
        dict
            Distribution statistics across shards
        """
        distribution = {
            "table": table_name,
            "total_rows": 0,
            "shards": [],
            "balance_score": 0.0,
            "recommendations": [],
        }

        try:
            with self.begin() as cur:
                # Get partition/shard information
                cur.execute(
                    """
                    SELECT
                        PARTITION_ORDINAL_POSITION,
                        PARTITION_METHOD,
                        PARTITION_EXPRESSION,
                        TABLE_ROWS
                    FROM INFORMATION_SCHEMA.PARTITIONS
                    WHERE TABLE_NAME = %s AND TABLE_SCHEMA = DATABASE()
                """,
                    (table_name,),
                )

                partitions = cur.fetchall()

                if partitions:
                    total_rows = sum(row[3] or 0 for row in partitions)
                    distribution["total_rows"] = total_rows

                    shard_sizes = []
                    for partition in partitions:
                        shard_info = {
                            "position": partition[0],
                            "method": partition[1],
                            "expression": partition[2],
                            "rows": partition[3] or 0,
                            "percentage": (partition[3] or 0) / total_rows * 100
                            if total_rows > 0
                            else 0,
                        }
                        distribution["shards"].append(shard_info)
                        shard_sizes.append(partition[3] or 0)

                    # Calculate balance score (higher is better)
                    if shard_sizes and max(shard_sizes) > 0:
                        min_size = min(shard_sizes)
                        max_size = max(shard_sizes)
                        distribution["balance_score"] = (
                            min_size / max_size if max_size > 0 else 0
                        )

                        # Generate recommendations
                        if distribution["balance_score"] < 0.7:
                            distribution["recommendations"].append(
                                {
                                    "type": "REBALANCE_SHARDS",
                                    "message": "Data distribution is unbalanced across shards",
                                    "priority": "medium",
                                    "current_balance": distribution["balance_score"],
                                }
                            )
                else:
                    # No explicit partitions found - table might be using hash distribution
                    distribution["recommendations"].append(
                        {
                            "type": "CHECK_DISTRIBUTION",
                            "message": "No explicit partitions found - verify table distribution method",
                            "priority": "low",
                        }
                    )

        except Exception as e:
            distribution["error"] = str(e)

        return distribution

    def optimize_distributed_joins(
        self, tables: list[str], join_columns: Optional[dict] = None
    ) -> dict:
        """Provide optimization recommendations for distributed joins.

        Parameters
        ----------
        tables
            List of table names involved in joins
        join_columns
            Dict mapping table names to their join columns

        Returns
        -------
        dict
            Join optimization recommendations
        """
        recommendations = {
            "tables": tables,
            "join_strategies": [],
            "shard_key_recommendations": [],
            "performance_tips": [],
        }

        try:
            # Analyze each table's distribution
            table_stats = {}
            for table in tables:
                stats = self.get_table_statistics(table)
                distribution = self.get_shard_distribution(table)
                table_stats[table] = {
                    "rows": stats.get("row_count", 0),
                    "size": stats.get("data_size", 0),
                    "shards": len(distribution.get("shards", [])),
                    "balance_score": distribution.get("balance_score", 0),
                }

            # Recommend join strategies based on table sizes
            sorted_tables = sorted(table_stats.items(), key=lambda x: x[1]["rows"])

            if len(sorted_tables) >= 2:
                smallest_table = sorted_tables[0]
                largest_table = sorted_tables[-1]

                # Broadcast join recommendation for small tables
                if smallest_table[1]["rows"] < 10000:
                    recommendations["join_strategies"].append(
                        {
                            "type": "BROADCAST_JOIN",
                            "table": smallest_table[0],
                            "rationale": f"Small table ({smallest_table[1]['rows']} rows) - broadcast to all nodes",
                            "hint": f"/*+ BROADCAST_JOIN({smallest_table[0]}) */",
                        }
                    )

                # Hash join for large tables
                if largest_table[1]["rows"] > 100000:
                    recommendations["join_strategies"].append(
                        {
                            "type": "HASH_JOIN",
                            "table": largest_table[0],
                            "rationale": f"Large table ({largest_table[1]['rows']} rows) - use hash join",
                            "hint": "/*+ USE_HASH_JOIN */",
                        }
                    )

            # Shard key recommendations
            if join_columns:
                for table, columns in join_columns.items():
                    if isinstance(columns, str):
                        columns = [columns]

                    recommendations["shard_key_recommendations"].append(
                        {
                            "table": table,
                            "recommended_shard_key": columns[0],
                            "rationale": "Use join column as shard key to enable co-located joins",
                            "benefit": "Eliminates network shuffle during joins",
                        }
                    )

            # General performance tips
            recommendations["performance_tips"].extend(
                [
                    {
                        "tip": "CO_LOCATED_JOINS",
                        "description": "Ensure frequently joined tables share the same shard key",
                    },
                    {
                        "tip": "BROADCAST_SMALL_TABLES",
                        "description": "Use broadcast joins for small lookup tables (< 10K rows)",
                    },
                    {
                        "tip": "FILTER_EARLY",
                        "description": "Apply WHERE clauses before JOINs to reduce data movement",
                    },
                    {
                        "tip": "INDEX_JOIN_COLUMNS",
                        "description": "Create indexes on join columns for better performance",
                    },
                ]
            )

        except Exception as e:
            recommendations["error"] = str(e)

        return recommendations

    def estimate_query_cost(self, query: str) -> dict:
        """Estimate the cost of executing a query in a distributed environment.

        Parameters
        ----------
        query
            SQL query to analyze

        Returns
        -------
        dict
            Cost estimation including resource usage and execution time prediction
        """
        cost_estimate = {
            "query": query,
            "estimated_cost": 0,
            "resource_usage": {},
            "bottlenecks": [],
            "optimizations": [],
        }

        try:
            # Get execution plan for cost analysis
            plan = self.explain_query(query)

            if "text_plan" in plan:
                total_rows = 0
                scan_cost = 0
                join_cost = 0

                for step in plan["text_plan"]:
                    rows = step.get("rows", 0) or 0
                    total_rows += rows

                    # Estimate scan costs
                    if step.get("type") in ["ALL", "range", "ref"]:
                        scan_cost += rows * 0.1  # Base cost per row scanned

                        if step.get("type") == "ALL":
                            scan_cost += (
                                rows * 0.5
                            )  # Additional cost for full table scan

                    # Estimate join costs
                    if "join" in str(step.get("extra", "")).lower():
                        join_cost += rows * 0.2  # Cost per row in join

                        # Additional cost for distributed joins
                        if "join buffer" in str(step.get("extra", "")).lower():
                            join_cost += (
                                rows * 0.8
                            )  # Network cost for distributed joins

                cost_estimate["estimated_cost"] = scan_cost + join_cost
                cost_estimate["resource_usage"] = {
                    "estimated_rows_scanned": total_rows,
                    "scan_cost": scan_cost,
                    "join_cost": join_cost,
                }

                # Identify bottlenecks
                if scan_cost > join_cost * 2:
                    cost_estimate["bottlenecks"].append(
                        {
                            "type": "SCAN_BOTTLENECK",
                            "description": "Query is scan-heavy - consider adding indexes",
                            "impact": "high",
                        }
                    )

                if join_cost > scan_cost * 2:
                    cost_estimate["bottlenecks"].append(
                        {
                            "type": "JOIN_BOTTLENECK",
                            "description": "Query is join-heavy - optimize join order and strategies",
                            "impact": "high",
                        }
                    )

                # Suggest optimizations
                if total_rows > 1000000:
                    cost_estimate["optimizations"].append(
                        {
                            "type": "PARALLEL_EXECUTION",
                            "hint": "/*+ PARALLEL_EXECUTION */",
                            "expected_benefit": "30-50% reduction in execution time",
                        }
                    )

                if join_cost > 100:
                    cost_estimate["optimizations"].append(
                        {
                            "type": "JOIN_OPTIMIZATION",
                            "hint": "/*+ USE_HASH_JOIN */",
                            "expected_benefit": "20-40% reduction in join time",
                        }
                    )

        except Exception as e:
            cost_estimate["error"] = str(e)

        return cost_estimate

    def bulk_insert_optimized(
        self,
        table_name: str,
        data: Any,
        batch_size: int = 10000,
        use_load_data: bool = True,
        disable_keys: bool = True,
        **kwargs,
    ) -> dict:
        """Optimized bulk insert for large datasets.

        Parameters
        ----------
        table_name
            Target table name
        data
            Data to insert (DataFrame, list of tuples, etc.)
        batch_size
            Number of rows per batch
        use_load_data
            Use LOAD DATA LOCAL INFILE for maximum performance
        disable_keys
            Temporarily disable key checks during insert
        **kwargs
            Additional keyword arguments for optimization

        Returns
        -------
        dict
            Insert performance statistics
        """
        import os
        import tempfile
        import time

        import pandas as pd

        stats = {
            "table": table_name,
            "total_rows": 0,
            "batches": 0,
            "total_time": 0,
            "rows_per_second": 0,
            "method": "load_data" if use_load_data else "batch_insert",
            "errors": [],
        }

        try:
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if hasattr(data, "to_frame"):
                    df = data.to_frame()
                else:
                    df = pd.DataFrame(data)
            else:
                df = data

            stats["total_rows"] = len(df)
            start_time = time.time()

            if use_load_data and len(df) > batch_size:
                # Use LOAD DATA LOCAL INFILE for best performance
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".csv"
                ) as tmp_file:
                    # Write CSV without header
                    df.to_csv(tmp_file.name, index=False, header=False, na_rep="\\N")

                    try:
                        with self.begin() as cur:
                            if disable_keys:
                                cur.execute(f"ALTER TABLE `{table_name}` DISABLE KEYS")

                            # Use LOAD DATA LOCAL INFILE
                            cur.execute(f"""
                                LOAD DATA LOCAL INFILE '{tmp_file.name}'
                                INTO TABLE `{table_name}`
                                FIELDS TERMINATED BY ','
                                ENCLOSED BY '"'
                                LINES TERMINATED BY '\\n'
                            """)

                            if disable_keys:
                                cur.execute(f"ALTER TABLE `{table_name}` ENABLE KEYS")

                            stats["batches"] = 1
                    finally:
                        os.unlink(tmp_file.name)
            else:
                # Use batch inserts
                schema = self.get_schema(table_name)
                columns = list(schema.names)

                # Prepare insert statement
                placeholders = ", ".join(["%s"] * len(columns))
                insert_sql = f"INSERT INTO `{table_name}` ({', '.join(f'`{col}`' for col in columns)}) VALUES ({placeholders})"

                with self.begin() as cur:
                    if disable_keys and len(df) > 1000:
                        cur.execute(f"ALTER TABLE `{table_name}` DISABLE KEYS")

                    # Process in batches
                    for i in range(0, len(df), batch_size):
                        batch = df.iloc[i : i + batch_size]
                        batch_data = [tuple(row) for row in batch.values]

                        cur.executemany(insert_sql, batch_data)
                        stats["batches"] += 1

                    if disable_keys and len(df) > 1000:
                        cur.execute(f"ALTER TABLE `{table_name}` ENABLE KEYS")

            end_time = time.time()
            stats["total_time"] = end_time - start_time
            stats["rows_per_second"] = (
                stats["total_rows"] / stats["total_time"]
                if stats["total_time"] > 0
                else 0
            )

        except Exception as e:
            stats["errors"].append(str(e))

        return stats

    def optimize_insert_performance(
        self, table_name: str, expected_rows: Optional[int] = None
    ) -> dict:
        """Optimize table settings for bulk insert performance.

        Parameters
        ----------
        table_name
            Table to optimize
        expected_rows
            Expected number of rows to insert

        Returns
        -------
        dict
            Optimization actions and recommendations
        """
        optimizations = {
            "table": table_name,
            "actions_taken": [],
            "recommendations": [],
            "original_settings": {},
            "errors": [],
        }

        try:
            with self.begin() as cur:
                # Get current table settings
                cur.execute(f"SHOW CREATE TABLE `{table_name}`")
                create_table = cur.fetchone()[1]
                optimizations["original_settings"]["create_table"] = create_table

                # Recommendations based on expected volume
                if expected_rows and expected_rows > 100000:
                    optimizations["recommendations"].extend(
                        [
                            {
                                "type": "DISABLE_AUTOCOMMIT",
                                "sql": "SET autocommit = 0",
                                "rationale": "Reduce commit overhead for large inserts",
                                "expected_benefit": "20-30% performance improvement",
                            },
                            {
                                "type": "INCREASE_BULK_INSERT_BUFFER",
                                "sql": "SET bulk_insert_buffer_size = 256*1024*1024",
                                "rationale": "Increase buffer for bulk operations",
                                "expected_benefit": "10-20% performance improvement",
                            },
                            {
                                "type": "DISABLE_UNIQUE_CHECKS",
                                "sql": "SET unique_checks = 0",
                                "rationale": "Skip unique constraint checks during insert",
                                "warning": "Re-enable after insert completion",
                            },
                        ]
                    )

                if expected_rows and expected_rows > 1000000:
                    optimizations["recommendations"].append(
                        {
                            "type": "USE_LOAD_DATA_INFILE",
                            "rationale": "LOAD DATA INFILE is fastest for very large datasets",
                            "expected_benefit": "50-80% performance improvement vs INSERT",
                        }
                    )

                # Check for indexes that might slow inserts
                cur.execute(f"""
                    SELECT INDEX_NAME, NON_UNIQUE, COLUMN_NAME
                    FROM INFORMATION_SCHEMA.STATISTICS
                    WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = DATABASE()
                    AND INDEX_NAME != 'PRIMARY'
                """)

                indexes = cur.fetchall()
                if len(indexes) > 3:  # Many secondary indexes
                    optimizations["recommendations"].append(
                        {
                            "type": "CONSIDER_DISABLE_KEYS",
                            "sql": f"ALTER TABLE `{table_name}` DISABLE KEYS",
                            "rationale": f"Table has {len(indexes)} secondary indexes that slow inserts",
                            "warning": "Remember to ENABLE KEYS after insert",
                            "expected_benefit": "30-50% performance improvement",
                        }
                    )

        except Exception as e:
            optimizations["errors"].append(str(e))

        return optimizations

    def parallel_bulk_insert(
        self,
        table_name: str,
        data: Any,
        num_workers: int = 4,
        batch_size: int = 10000,
    ) -> dict:
        """Perform parallel bulk insert using multiple connections.

        Parameters
        ----------
        table_name
            Target table name
        data
            Data to insert
        num_workers
            Number of parallel worker connections
        batch_size
            Rows per batch per worker

        Returns
        -------
        dict
            Parallel insert performance statistics
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import pandas as pd

        stats = {
            "table": table_name,
            "total_rows": 0,
            "workers": num_workers,
            "batch_size": batch_size,
            "total_time": 0,
            "worker_stats": [],
            "errors": [],
        }

        try:
            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if hasattr(data, "to_frame"):
                    df = data.to_frame()
                else:
                    df = pd.DataFrame(data)
            else:
                df = data

            stats["total_rows"] = len(df)
            start_time = time.time()

            def worker_insert(worker_id: int, data_chunk: pd.DataFrame) -> dict:
                """Worker function for parallel inserts."""
                worker_stats = {
                    "worker_id": worker_id,
                    "rows_processed": len(data_chunk),
                    "batches": 0,
                    "time": 0,
                    "errors": [],
                }

                try:
                    # Create separate connection for this worker
                    worker_backend = self._from_url(
                        type(
                            "MockResult",
                            (),
                            {
                                "hostname": self._client._get_host_info()[0],
                                "port": self._client._get_host_info()[1],
                                "username": "root",  # Would need proper user info
                                "password": "",  # Would need proper password
                                "path": f"/{self.current_database}",
                            },
                        )()
                    )

                    worker_start = time.time()

                    # Use bulk insert for this chunk
                    result = worker_backend.bulk_insert_optimized(
                        table_name,
                        data_chunk,
                        batch_size=batch_size,
                        use_load_data=False,  # Don't use LOAD DATA for parallel workers
                        disable_keys=False,  # Don't disable keys per worker
                    )

                    worker_stats["batches"] = result.get("batches", 0)
                    worker_stats["time"] = time.time() - worker_start

                except Exception as e:
                    worker_stats["errors"].append(str(e))

                return worker_stats

            # Split data into chunks for workers
            chunk_size = len(df) // num_workers
            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_workers - 1 else len(df)
                chunks.append(df.iloc[start_idx:end_idx])

            # Execute parallel inserts
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_worker = {
                    executor.submit(worker_insert, i, chunk): i
                    for i, chunk in enumerate(chunks)
                }

                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        worker_result = future.result()
                        stats["worker_stats"].append(worker_result)
                    except Exception as e:
                        stats["errors"].append(f"Worker {worker_id} failed: {e}")

            stats["total_time"] = time.time() - start_time
            stats["rows_per_second"] = (
                stats["total_rows"] / stats["total_time"]
                if stats["total_time"] > 0
                else 0
            )

        except Exception as e:
            stats["errors"].append(str(e))

        return stats

    def benchmark_insert_methods(
        self,
        table_name: str,
        sample_data: Any,
        methods: Optional[list[str]] = None,
    ) -> dict:
        """Benchmark different insert methods to find the best one.

        Parameters
        ----------
        table_name
            Table to test inserts on
        sample_data
            Sample data for benchmarking
        methods
            List of methods to test

        Returns
        -------
        dict
            Benchmark results for different insert methods
        """
        if methods is None:
            methods = ["batch_insert", "bulk_optimized", "load_data"]

        benchmarks = {
            "table": table_name,
            "sample_size": len(sample_data) if hasattr(sample_data, "__len__") else 0,
            "methods": {},
            "recommendation": None,
        }

        # Create a test table for benchmarking
        test_table = f"_benchmark_{table_name}_{int(time.time())}"

        try:
            # Copy table structure
            schema = self.get_schema(table_name)
            self.create_table(test_table, schema=schema, temp=True)

            for method in methods:
                method_stats = {"method": method, "error": None}

                try:
                    if method == "batch_insert":
                        result = self.bulk_insert_optimized(
                            test_table,
                            sample_data,
                            use_load_data=False,
                            batch_size=1000,
                        )
                    elif method == "bulk_optimized":
                        result = self.bulk_insert_optimized(
                            test_table,
                            sample_data,
                            use_load_data=True,
                            batch_size=10000,
                        )
                    elif method == "load_data":
                        result = self.bulk_insert_optimized(
                            test_table,
                            sample_data,
                            use_load_data=True,
                            batch_size=len(sample_data),
                        )

                    method_stats.update(
                        {
                            "total_time": result.get("total_time", 0),
                            "rows_per_second": result.get("rows_per_second", 0),
                            "batches": result.get("batches", 0),
                        }
                    )

                    # Clean up for next test
                    with self.begin() as cur:
                        cur.execute(f"DELETE FROM `{test_table}`")

                except Exception as e:
                    method_stats["error"] = str(e)

                benchmarks["methods"][method] = method_stats

            # Determine best method
            best_method = None
            best_rps = 0

            for method, stats in benchmarks["methods"].items():
                if stats.get("rows_per_second", 0) > best_rps and not stats.get(
                    "error"
                ):
                    best_rps = stats["rows_per_second"]
                    best_method = method

            benchmarks["recommendation"] = {
                "method": best_method,
                "rows_per_second": best_rps,
                "rationale": f"Achieved best performance: {best_rps:.0f} rows/second",
            }

        except Exception as e:
            benchmarks["error"] = str(e)
        finally:
            # Clean up test table
            try:
                with self.begin() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS `{test_table}`")
            except Exception:
                pass

        return benchmarks

    def __init__(self, *args, **kwargs):
        """Initialize backend with connection pool support."""
        super().__init__(*args, **kwargs)
        self._connection_pool = None
        self._pool_size = 10
        self._pool_timeout = 30
        self._retry_config = {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_exceptions": (OSError, ConnectionError),
        }

    @property
    def connection_pool(self):
        """Get or create connection pool."""
        if self._connection_pool is None:
            self._create_connection_pool()
        return self._connection_pool

    def _create_connection_pool(
        self, pool_size: Optional[int] = None, timeout: Optional[int] = None
    ):
        """Create a connection pool for better performance.

        Parameters
        ----------
        pool_size
            Maximum number of connections in pool
        timeout
            Connection timeout in seconds
        """
        try:
            import queue
            import threading

            import singlestoredb as s2

            pool_size = pool_size or self._pool_size
            timeout = timeout or self._pool_timeout

            class ConnectionPool:
                def __init__(self, size, connect_params, timeout):
                    self.size = size
                    self.timeout = timeout
                    self.connect_params = connect_params
                    self._pool = queue.Queue(maxsize=size)
                    self._lock = threading.Lock()
                    self._created_connections = 0

                    # Pre-populate pool with initial connections
                    for _ in range(min(2, size)):  # Start with 2 connections
                        conn = self._create_connection()
                        if conn:
                            self._pool.put(conn)

                def _create_connection(self):
                    """Create a new database connection."""
                    try:
                        return s2.connect(**self.connect_params)
                    except Exception:
                        # Log connection failure but don't print
                        return None

                def get_connection(self, timeout=None):
                    """Get a connection from the pool."""
                    timeout = timeout or self.timeout

                    try:
                        # Try to get existing connection
                        conn = self._pool.get(timeout=timeout)

                        # Test connection health
                        try:
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.close()
                            return conn
                        except Exception:
                            # Connection is dead, create new one
                            conn.close()
                            return self._create_connection()

                    except queue.Empty:
                        # No connections available, create new if under limit
                        with self._lock:
                            if self._created_connections < self.size:
                                self._created_connections += 1
                                return self._create_connection()

                        raise ConnectionError("Connection pool exhausted")

                def return_connection(self, conn):
                    """Return a connection to the pool."""
                    try:
                        # Test if connection is still valid
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.close()

                        # Put back in pool
                        self._pool.put_nowait(conn)
                    except (queue.Full, Exception):
                        # Pool is full or connection is bad, close it
                        with contextlib.suppress(Exception):
                            conn.close()

                def close_all(self):
                    """Close all connections in the pool."""
                    while not self._pool.empty():
                        try:
                            conn = self._pool.get_nowait()
                            conn.close()
                        except (queue.Empty, Exception):
                            break

            # Get connection parameters from current client
            connect_params = {
                "host": getattr(self._client, "host", "localhost"),
                "user": getattr(self._client, "user", "root"),
                "password": getattr(self._client, "password", ""),
                "port": getattr(self._client, "port", 3306),
                "database": getattr(self._client, "database", ""),
                "autocommit": True,
                "local_infile": 0,
            }

            self._connection_pool = ConnectionPool(pool_size, connect_params, timeout)

        except ImportError:
            # Connection pooling requires singlestoredb package
            self._connection_pool = None
        except Exception:
            # Failed to create connection pool
            self._connection_pool = None

    def get_pooled_connection(self, timeout: Optional[int] = None):
        """Get a connection from the pool.

        Parameters
        ----------
        timeout
            Connection timeout in seconds

        Returns
        -------
        Connection context manager
        """
        import contextlib

        @contextlib.contextmanager
        def connection_manager():
            conn = None
            try:
                if self._connection_pool:
                    conn = self._connection_pool.get_connection(timeout)
                else:
                    # Fallback to regular connection
                    conn = self._client
                yield conn
            finally:
                if conn and self._connection_pool and conn != self._client:
                    self._connection_pool.return_connection(conn)

        return connection_manager()

    def close_connection_pool(self):
        """Close the connection pool and all its connections."""
        if self._connection_pool:
            self._connection_pool.close_all()
            self._connection_pool = None

    def _execute_with_retry(
        self, operation, *args, max_retries: Optional[int] = None, **kwargs
    ):
        """Execute an operation with automatic retry logic.

        Parameters
        ----------
        operation
            Function to execute
        args
            Positional arguments for operation
        max_retries
            Maximum number of retry attempts
        kwargs
            Keyword arguments for operation

        Returns
        -------
        Result of successful operation
        """
        import random
        import time

        max_retries = max_retries or self._retry_config["max_retries"]
        backoff_factor = self._retry_config["backoff_factor"]
        retry_exceptions = self._retry_config["retry_exceptions"]

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except retry_exceptions as e:
                last_exception = e

                if attempt == max_retries:
                    break  # Don't sleep after last attempt

                # Exponential backoff with jitter
                import random

                delay = backoff_factor * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                time.sleep(min(delay, 30))  # Cap at 30 seconds

                # Try to reconnect if it's a connection error
                try:
                    self._reconnect()
                except Exception:
                    pass  # Ignore reconnection errors, will retry operation

            except Exception as e:
                # Non-retryable exception
                raise e

        # All retries exhausted
        raise last_exception

    def _reconnect(self):
        """Attempt to reconnect to the database."""
        try:
            self.do_connect(
                *self._original_connect_params[0],
                **self._original_connect_params[1],
            )
        except Exception as e:
            raise ConnectionError(f"Failed to reconnect: {e}")

    def do_connect(self, *args: str, **kwargs: Any) -> None:
        """Create an Ibis client connected to a SingleStoreDB database with retry support.

        Parameters
        ----------
        args
            If given, the first argument is treated as a host or URL
        kwargs
            Additional connection parameters
            - host : Hostname or URL
            - user : Username
            - password : Password
            - port : Port number
            - database : Database to connect to
        """
        self._original_connect_params = (args, kwargs)

        if args:
            params = build_params(host=args[0], **kwargs)
        else:
            params = build_params(**kwargs)

        # Use SingleStoreDB client exclusively with retry logic
        def _connect():
            import singlestoredb as s2

            self._client = s2.connect(**params)

        return self._execute_with_retry(_connect)

    def configure_retry_policy(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        retry_exceptions: Optional[tuple] = None,
    ):
        """Configure retry policy for database operations.

        Parameters
        ----------
        max_retries
            Maximum number of retry attempts
        backoff_factor
            Multiplier for exponential backoff
        retry_exceptions
            Tuple of exceptions to retry on
        """
        if retry_exceptions is None:
            retry_exceptions = (OSError, ConnectionError, TimeoutError)

        self._retry_config = {
            "max_retries": max_retries,
            "backoff_factor": backoff_factor,
            "retry_exceptions": retry_exceptions,
        }

    def set_connection_timeout(self, timeout: int):
        """Set connection timeout for database operations.

        Parameters
        ----------
        timeout
            Timeout in seconds
        """
        try:
            with self.begin() as cur:
                cur.execute(f"SET SESSION wait_timeout = {timeout}")
                cur.execute(f"SET SESSION interactive_timeout = {timeout}")
        except Exception as e:
            raise ConnectionError(f"Failed to set timeout: {e}")

    def execute_with_timeout(
        self, query: str, timeout: int = 30, params: Optional[tuple] = None
    ):
        """Execute a query with a specific timeout.

        Parameters
        ----------
        query
            SQL query to execute
        timeout
            Query timeout in seconds
        params
            Query parameters

        Returns
        -------
        Query results
        """
        import threading

        result = None
        exception = None

        def query_worker():
            nonlocal result, exception
            try:
                with self.begin() as cur:
                    if params:
                        cur.execute(query, params)
                    else:
                        cur.execute(query)
                    result = cur.fetchall()
            except Exception as e:
                exception = e

        # Create and start worker thread
        worker = threading.Thread(target=query_worker)
        worker.daemon = True
        worker.start()

        # Wait for completion or timeout
        worker.join(timeout)

        if worker.is_alive():
            # Query timed out
            raise TimeoutError(f"Query timed out after {timeout} seconds")

        if exception:
            raise exception

        return result

    @contextlib.contextmanager
    def connection_timeout(self, timeout: int):
        """Context manager for temporary connection timeout.

        Parameters
        ----------
        timeout
            Temporary timeout in seconds
        """
        original_timeout = None

        try:
            # Get current timeout
            with self.begin() as cur:
                cur.execute("SELECT @@wait_timeout")
                original_timeout = cur.fetchone()[0]

            # Set new timeout
            self.set_connection_timeout(timeout)
            yield

        finally:
            # Restore original timeout
            if original_timeout is not None:
                try:
                    self.set_connection_timeout(original_timeout)
                except Exception:
                    pass  # Ignore errors during cleanup

    def test_connection_health(self, timeout: int = 5) -> dict:
        """Test connection health and performance.

        Parameters
        ----------
        timeout
            Test timeout in seconds

        Returns
        -------
        dict
            Connection health metrics
        """
        import time

        health = {
            "connected": False,
            "response_time": None,
            "server_version": None,
            "current_database": None,
            "connection_id": None,
            "uptime": None,
            "errors": [],
        }

        try:
            start_time = time.time()

            # Test basic connectivity
            with self.connection_timeout(timeout):
                with self.begin() as cur:
                    # Test response time
                    cur.execute("SELECT 1")
                    cur.fetchone()
                    health["response_time"] = time.time() - start_time
                    health["connected"] = True

                    # Get server info
                    cur.execute("SELECT VERSION()")
                    health["server_version"] = cur.fetchone()[0]

                    # Get current database
                    cur.execute("SELECT DATABASE()")
                    health["current_database"] = cur.fetchone()[0]

                    # Get connection ID
                    cur.execute("SELECT CONNECTION_ID()")
                    health["connection_id"] = cur.fetchone()[0]

                    # Get server uptime
                    cur.execute("SHOW STATUS LIKE 'Uptime'")
                    uptime_row = cur.fetchone()
                    if uptime_row:
                        health["uptime"] = int(uptime_row[1])

        except TimeoutError:
            health["errors"].append(f"Connection test timed out after {timeout}s")
        except Exception as e:
            health["errors"].append(str(e))

        return health

    def monitor_connection_pool(self) -> dict:
        """Monitor connection pool status and performance.

        Returns
        -------
        dict
            Pool monitoring information
        """
        pool_stats = {
            "pool_enabled": self._connection_pool is not None,
            "pool_size": self._pool_size,
            "pool_timeout": self._pool_timeout,
            "active_connections": 0,
            "available_connections": 0,
            "health_check_results": [],
        }

        if self._connection_pool:
            try:
                # Get pool statistics
                pool = self._connection_pool
                pool_stats["available_connections"] = pool._pool.qsize()
                pool_stats["active_connections"] = (
                    pool._created_connections - pool_stats["available_connections"]
                )

                # Health check a sample of pooled connections
                test_connections = min(3, pool_stats["available_connections"])
                for i in range(test_connections):
                    try:
                        with self.get_pooled_connection(timeout=2) as conn:
                            cursor = conn.cursor()
                            start_time = time.time()
                            cursor.execute("SELECT 1")
                            cursor.fetchone()
                            cursor.close()

                            pool_stats["health_check_results"].append(
                                {
                                    "connection": i,
                                    "healthy": True,
                                    "response_time": time.time() - start_time,
                                }
                            )
                    except Exception as e:
                        pool_stats["health_check_results"].append(
                            {
                                "connection": i,
                                "healthy": False,
                                "error": str(e),
                            }
                        )

            except Exception as e:
                pool_stats["error"] = str(e)

        return pool_stats

    def cleanup_connections(self, force: bool = False):
        """Clean up database connections and resources.

        Parameters
        ----------
        force
            Force close all connections immediately
        """
        errors = []

        try:
            # Close connection pool
            if self._connection_pool:
                self._connection_pool.close_all()
                self._connection_pool = None

        except Exception as e:
            errors.append(f"Error closing connection pool: {e}")

        try:
            # Close main client connection
            if hasattr(self, "_client") and self._client:
                if force:
                    # Force immediate close
                    self._client.close()
                else:
                    # Graceful close - finish pending transactions
                    try:
                        with self._client.cursor() as cur:
                            cur.execute("COMMIT")
                    except Exception:
                        pass  # Ignore transaction errors
                    finally:
                        self._client.close()

        except Exception as e:
            errors.append(f"Error closing main connection: {e}")

        if errors:
            raise ConnectionError(f"Cleanup errors: {'; '.join(errors)}")

    def __del__(self):
        """Ensure connections are closed when backend is destroyed."""
        try:
            self.cleanup_connections(force=True)
        except Exception:
            pass  # Ignore errors during destruction

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_connections()

    @contextlib.contextmanager
    def managed_connection(self, cleanup_on_error: bool = True):
        """Context manager for automatic connection cleanup.

        Parameters
        ----------
        cleanup_on_error
            Whether to cleanup connections on error
        """
        try:
            yield self
        except Exception as e:
            if cleanup_on_error:
                try:
                    self.cleanup_connections(force=True)
                except Exception:
                    pass  # Ignore cleanup errors when already handling an exception
            raise e
        finally:
            # Always attempt graceful cleanup
            try:
                self.cleanup_connections()
            except Exception:
                pass  # Ignore cleanup errors in finally block

    def get_connection_status(self) -> dict:
        """Get detailed status of all connections.

        Returns
        -------
        dict
            Connection status information
        """
        status = {
            "main_connection": {"active": False, "details": None},
            "connection_pool": {"enabled": False, "details": None},
            "total_connections": 0,
            "healthy_connections": 0,
            "errors": [],
        }

        # Check main connection
        try:
            if hasattr(self, "_client") and self._client:
                health = self.test_connection_health(timeout=2)
                status["main_connection"] = {
                    "active": health["connected"],
                    "details": health,
                }
                if health["connected"]:
                    status["healthy_connections"] += 1
                status["total_connections"] += 1
        except Exception as e:
            status["errors"].append(f"Main connection error: {e}")

        # Check connection pool
        try:
            if self._connection_pool:
                pool_status = self.monitor_connection_pool()
                status["connection_pool"] = {
                    "enabled": True,
                    "details": pool_status,
                }
                status["total_connections"] += pool_status.get("active_connections", 0)
                status["total_connections"] += pool_status.get(
                    "available_connections", 0
                )

                # Count healthy pooled connections
                for result in pool_status.get("health_check_results", []):
                    if result.get("healthy", False):
                        status["healthy_connections"] += 1

        except Exception as e:
            status["errors"].append(f"Connection pool error: {e}")

        return status

    def optimize_connection_settings(self) -> dict:
        """Optimize connection settings for performance.

        Returns
        -------
        dict
            Applied optimizations
        """
        optimizations = {
            "applied": [],
            "recommendations": [],
            "errors": [],
        }

        try:
            with self.begin() as cur:
                # Optimize connection-level settings
                settings = [
                    (
                        "SET SESSION sql_mode = 'NO_ENGINE_SUBSTITUTION'",
                        "Reduce SQL strictness for better compatibility",
                    ),
                    (
                        "SET SESSION autocommit = 1",
                        "Enable autocommit for better performance",
                    ),
                    (
                        "SET SESSION tx_isolation = 'READ-COMMITTED'",
                        "Use optimal isolation level",
                    ),
                    ("SET SESSION query_cache_type = ON", "Enable query caching"),
                    (
                        "SET SESSION bulk_insert_buffer_size = 64*1024*1024",
                        "Optimize bulk inserts",
                    ),
                ]

                for sql, description in settings:
                    try:
                        cur.execute(sql)
                        optimizations["applied"].append(
                            {
                                "setting": sql,
                                "description": description,
                            }
                        )
                    except Exception as e:
                        optimizations["errors"].append(f"{sql}: {e}")

                # Add recommendations for connection pooling
                if not self._connection_pool:
                    optimizations["recommendations"].append(
                        {
                            "type": "CONNECTION_POOLING",
                            "description": "Enable connection pooling for better performance",
                            "method": "backend._create_connection_pool()",
                        }
                    )

                # Add recommendations for timeout settings
                optimizations["recommendations"].append(
                    {
                        "type": "TIMEOUT_OPTIMIZATION",
                        "description": "Set appropriate timeouts for your workload",
                        "method": "backend.set_connection_timeout(300)",
                    }
                )

        except Exception as e:
            optimizations["errors"].append(f"Failed to optimize settings: {e}")

        return optimizations

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename a table in SingleStoreDB.

        Parameters
        ----------
        old_name
            Current name of the table
        new_name
            New name for the table

        Examples
        --------
        >>> con.rename_table("old_table", "new_table")
        """
        old_name = self._quote_table_name(old_name)
        new_name = self._quote_table_name(new_name)
        with self.begin() as cur:
            cur.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")

    # Method removed - SingleStoreDB doesn't support catalogs

    def _quote_table_name(self, name: str) -> str:
        """Quote a table name for safe SQL usage."""
        import sqlglot as sg

        return sg.to_identifier(name, quoted=True).sql("singlestore")


def connect(
    host: str = "localhost",
    user: str = "root",
    password: str = "",
    port: int = 3306,
    database: str = "",
    **kwargs: Any,
) -> Backend:
    """Create an Ibis client connected to a SingleStoreDB database.

    Parameters
    ----------
    host
        SingleStoreDB hostname or IP address
    user
        Username for authentication
    password
        Password for authentication
    port
        Port number (default 3306)
    database
        Database name to connect to
    kwargs
        Additional connection parameters:
        - autocommit: Enable autocommit mode (default True)
        - local_infile: Enable LOCAL INFILE capability (default 0)
        - charset: Character set (default utf8mb4)
        - ssl_disabled: Disable SSL connection
        - connect_timeout: Connection timeout in seconds
        - read_timeout: Read timeout in seconds
        - write_timeout: Write timeout in seconds

    Returns
    -------
    Backend
        An Ibis SingleStoreDB backend instance

    Examples
    --------
    Basic connection:

    >>> import ibis
    >>> con = ibis.singlestoredb.connect(
    ...     host="localhost", user="root", password="password", database="my_database"
    ... )

    Connection with additional options:

    >>> con = ibis.singlestoredb.connect(
    ...     host="singlestore.example.com",
    ...     port=3306,
    ...     user="app_user",
    ...     password="secret",
    ...     database="production",
    ...     autocommit=True,
    ...     connect_timeout=30,
    ... )

    Using connection string (alternative method):

    >>> con = ibis.connect("singlestoredb://user:password@host:port/database")
    """
    backend = Backend()
    backend.do_connect(
        host=host, user=user, password=password, port=port, database=database, **kwargs
    )
    return backend
