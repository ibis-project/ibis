"""Impala backend."""

from __future__ import annotations

import contextlib
import operator
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlparse

import impala.dbapi as impyla
import sqlglot as sg
import sqlglot.expressions as sge
from impala.error import Error as ImpylaError

import ibis.common.exceptions as com
import ibis.config
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.impala import ddl, udf
from ibis.backends.impala.client import ImpalaTable
from ibis.backends.impala.compiler import ImpalaCompiler
from ibis.backends.impala.ddl import (
    CTAS,
    CreateDatabase,
    CreateTableWithSchema,
    CreateView,
    DropDatabase,
    DropTable,
    DropView,
    RenameTable,
    TruncateTable,
)
from ibis.backends.impala.udf import (
    aggregate_function,
    scalar_function,
    wrap_uda,
    wrap_udf,
)
from ibis.config import options

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa

    import ibis.expr.operations as ops


__all__ = (
    "Backend",
    "aggregate_function",
    "scalar_function",
    "wrap_uda",
    "wrap_udf",
)


class Backend(SQLGlotBackend):
    name = "impala"
    compiler = ImpalaCompiler()

    supports_in_memory_tables = True

    class Options(ibis.config.Config):
        """Impala specific options.

        Parameters
        ----------
        temp_db : str, default "__ibis_tmp"
            Database to use for temporary objects.
        temp_path : str, default "/tmp/ibis"
            Path for storage of temporary data.

        """

        temp_db: str = "__ibis_tmp"
        temp_path: str = "/tmp/__ibis"

    def _from_url(self, url: str, **kwargs: Any) -> Backend:
        """Connect to a backend using a URL `url`.

        Parameters
        ----------
        url
            URL with which to connect to a backend.
        kwargs
            Additional keyword arguments passed to the `connect` method.

        Returns
        -------
        BaseBackend
            A backend instance

        """
        url = urlparse(url)

        for name in ("username", "hostname", "port", "password"):
            if value := (
                getattr(url, name, None)
                or os.environ.get(f"{self.name.upper()}_{name.upper()}")
            ):
                kwargs[name] = value

        with contextlib.suppress(KeyError):
            kwargs["host"] = kwargs.pop("hostname")

        (database,) = url.path[1:].split("/", 1)
        if database:
            kwargs["database"] = database

        query_params = parse_qs(url.query)

        for name, value in query_params.items():
            if len(value) > 1:
                kwargs[name] = value
            elif len(value) == 1:
                kwargs[name] = value[0]
            else:
                raise com.IbisError(f"Invalid URL parameter: {name}")

        self._convert_kwargs(kwargs)
        return self.connect(**kwargs)

    def do_connect(
        self,
        host: str = "localhost",
        port: int = 21050,
        database: str = "default",
        timeout: int = 45,
        use_ssl: bool = False,
        ca_cert: str | Path | None = None,
        user: str | None = None,
        password: str | None = None,
        auth_mechanism: Literal["NOSASL", "PLAIN", "GSSAPI", "LDAP"] = "NOSASL",
        kerberos_service_name: str = "impala",
        **params: Any,
    ):
        """Create an Impala `Backend` for use with Ibis.

        Parameters
        ----------
        host
            Host name of the impalad or HiveServer2 in Hive
        port
            Impala's HiveServer2 port
        database
            Default database when obtaining new cursors
        timeout
            Connection timeout in seconds when communicating with HiveServer2
        use_ssl
            Use SSL when connecting to HiveServer2
        ca_cert
            Local path to 3rd party CA certificate or copy of server
            certificate for self-signed certificates. If SSL is enabled, but
            this argument is ``None``, then certificate validation is skipped.
        user
            LDAP user to authenticate
        password
            LDAP password to authenticate
        auth_mechanism
            |   Value    | Meaning                        |
            | :--------: | :----------------------------- |
            | `'NOSASL'` | insecure Impala connections    |
            | `'PLAIN'`  | insecure Hive clusters         |
            |  `'LDAP'`  | LDAP authenticated connections |
            | `'GSSAPI'` | Kerberos-secured clusters      |
        kerberos_service_name
            Specify a particular `impalad` service principal.
        params
            Any additional parameters necessary to open a connection to Impala.
            Please refer to impyla documentation for the full list of
            possible arguments.

        Examples
        --------
        >>> import os
        >>> import ibis
        >>> impala_host = os.environ.get("IBIS_TEST_IMPALA_HOST", "localhost")
        >>> impala_port = int(os.environ.get("IBIS_TEST_IMPALA_PORT", 21050))
        >>> client = ibis.impala.connect(host=impala_host, port=impala_port)
        >>> client  # doctest: +ELLIPSIS
        <ibis.backends.impala.Backend object at 0x...>

        """
        if ca_cert is not None:
            params["ca_cert"] = str(ca_cert)

        # make sure the connection works
        con = impyla.connect(
            host=host,
            port=port,
            database=database,
            timeout=timeout,
            use_ssl=use_ssl,
            user=user,
            password=password,
            auth_mechanism=auth_mechanism,
            kerberos_service_name=kerberos_service_name,
            **params,
        )
        with contextlib.closing(
            con.cursor(user=params.get("user"), convert_types=True)
        ) as cur:
            cur.ping()

        self.con = con
        self.options = {}
        self._ensure_temp_db_exists()

    @cached_property
    def version(self):
        with self._safe_raw_sql("SELECT VERSION()") as cursor:
            (result,) = cursor.fetchone()
        return result

    def list_databases(self, like=None):
        with self._safe_raw_sql("SHOW DATABASES") as cur:
            databases = fetchall(cur)
        return self._filter_with_like(databases.name.tolist(), like)

    def list_tables(self, like=None, database=None):
        statement = "SHOW TABLES"
        if database is not None:
            statement += f" IN {database}"

        with self._safe_raw_sql(statement) as cursor:
            tables = fetchall(cursor)
        return self._filter_with_like(tables.name.tolist(), like=like)

    def raw_sql(self, query: str):
        cursor = self.con.cursor()

        try:
            for k, v in self.options.items():
                q = f"SET {k} = {v!r}"
                util.log(q)
                cursor.execute_async(q)

            cursor._wait_to_finish()

            util.log(query)
            cursor.execute_async(query)

            cursor._wait_to_finish()
        except (Exception, KeyboardInterrupt):
            cursor.cancel_operation()
            cursor.close()
            raise

        return cursor

    def _fetch_from_cursor(self, cursor, schema):
        from ibis.formats.pandas import PandasData

        results = fetchall(cursor)
        return PandasData.convert_table(results, schema)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query: str):
        if not isinstance(query, str):
            try:
                query = query.sql(dialect=self.dialect)
            except AttributeError:
                query = query.compile()

        assert isinstance(query, str), type(query)
        with contextlib.closing(self.raw_sql(query)) as cur:
            yield cur

    def _safe_exec_sql(self, *args, **kwargs):
        with self._safe_raw_sql(*args, **kwargs):
            pass

    def _fully_qualified_name(self, name, database):
        database = database or self.current_database
        return sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            self.dialect
        )

    @property
    def current_database(self) -> str:
        with self._safe_raw_sql("SELECT CURRENT_DATABASE()") as cur:
            [(db,)] = cur.fetchall()
        return db

    def create_database(self, name, path=None, force=False):
        """Create a new Impala database.

        Parameters
        ----------
        name
            Database name
        path
            Path where to store the database data; otherwise uses the Impala default
        force
            Forcibly create the database

        """
        statement = CreateDatabase(name, path=path, can_exist=force)
        self._safe_exec_sql(statement)

    def drop_database(self, name, force=False):
        """Drop an Impala database.

        Parameters
        ----------
        name
            Database name
        force
            If False and there are any tables in this database, raises an
            IntegrityError

        """
        if not force or name in self.list_databases():
            tables = self.list_tables(database=name)
            udfs = self.list_udfs(database=name)
            udas = self.list_udas(database=name)
        else:
            tables = []
            udfs = []
            udas = []
        if force:
            for table in tables:
                util.log(f"Dropping {name}.{table}")
                self.drop_table_or_view(table, database=name)
            for func in udfs:
                util.log(f"Dropping function {func.name}({func.inputs})")
                self.drop_udf(
                    func.name,
                    input_types=func.inputs,
                    database=name,
                    force=True,
                )
            for func in udas:
                util.log(f"Dropping aggregate function {func.name}({func.inputs})")
                self.drop_uda(
                    func.name,
                    input_types=func.inputs,
                    database=name,
                    force=True,
                )
        elif tables or udfs or udas:
            raise com.IntegrityError(
                f"Database {name} must be empty before "
                "being dropped, or set force=True"
            )
        statement = DropDatabase(name, must_exist=not force)
        self._safe_exec_sql(statement)

    def get_schema(
        self, table_name: str, schema: str | None = None, database: str | None = None
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name
        schema
            Schema name. Unused in the impala backend.
        database
            Database name

        Returns
        -------
        Schema
            Ibis schema

        """
        query = sge.Describe(
            this=sg.table(
                table_name, db=schema, catalog=database, quoted=self.compiler.quoted
            )
        )

        with self._safe_raw_sql(query) as cur:
            meta = fetchall(cur)
        return sch.Schema.from_tuples(
            zip(meta["name"], meta["type"].map(self.compiler.type_mapper.from_string))
        )

    def _metadata(self, query: str) -> Iterator[tuple[str, dt.DataType]]:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        query
            Query to execute against Impala

        Returns
        -------
        Iterator[tuple[str, dt.DataType]]
            Iterator of column name and Ibis type pairs

        """
        tmpview = util.gen_name("impala_tmpview")
        query = f"CREATE VIEW IF NOT EXISTS {tmpview} AS {query}"

        with self._safe_raw_sql(query) as cur:
            try:
                cur.execute(f"DESCRIBE {tmpview}")
                meta = fetchall(cur)
            finally:
                cur.execute(f"DROP VIEW IF EXISTS {tmpview}")

        return zip(
            meta["name"], meta["type"].map(self.compiler.type_mapper.from_string)
        )

    @property
    def client_options(self):
        return self.con.options

    def get_options(self) -> dict[str, str]:
        """Return current query options for the Impala session."""
        with self._safe_raw_sql("SET") as cur:
            opts = fetchall(cur)

        return dict(zip(opts.option, opts.value))

    def set_options(self, options):
        self.options.update(options)

    def set_compression_codec(self, codec):
        self.set_options({"COMPRESSION_CODEC": str(codec).lower()})

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        select = self.compile(obj)
        statement = CreateView(name, select, database=database, can_exist=overwrite)
        self._safe_exec_sql(statement)
        return self.table(name, database=database)

    def drop_view(self, name, database=None, force=False):
        stmt = DropView(name, database=database, must_exist=not force)
        self._safe_exec_sql(stmt)

    def table(self, name: str, database: str | None = None, **kwargs: Any) -> ir.Table:
        expr = super().table(name, database=database, **kwargs)
        return ImpalaTable(expr.op())

    def create_table(
        self,
        name: str,
        obj: ir.Table | None = None,
        *,
        schema=None,
        database=None,
        temp: bool | None = None,
        overwrite: bool = False,
        external: bool = False,
        format="parquet",
        location=None,
        partition=None,
        like_parquet=None,
    ) -> ir.Table:
        """Create a new table in Impala using an Ibis table expression.

        Parameters
        ----------
        name
            Table name
        obj
            If passed, creates table from select statement results
        schema
            Mutually exclusive with obj, creates an empty table with a
            particular schema
        database
            Database name
        temp
            Whether a table is temporary
        overwrite
            Do not create table if table with indicated name already exists
        external
            Create an external table; Impala will not delete the underlying
            data when the table is dropped
        format
            File format
        location
            Specify the directory location where Impala reads and writes files
            for the table
        partition
            Must pass a schema to use this. Cannot partition from an
            expression.
        like_parquet
            Can specify instead of a schema

        """
        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")

        if temp is not None:
            raise NotImplementedError(
                "Impala backend does not yet support temporary tables"
            )
        if like_parquet is not None:
            raise NotImplementedError

        if obj is not None:
            if not isinstance(obj, ir.Table):
                obj = ibis.memtable(obj)

            self._run_pre_execute_hooks(obj)

            select = self.compile(obj)

            if overwrite:
                self.drop_table(name, force=True)

            self._safe_exec_sql(
                CTAS(
                    name,
                    select,
                    database=database or self.current_database,
                    format=format,
                    external=True if location is not None else external,
                    partition=partition,
                    path=location,
                )
            )
        else:  # schema is not None
            if overwrite:
                self.drop_table(name, force=True)
            self._safe_exec_sql(
                CreateTableWithSchema(
                    name,
                    schema if schema is not None else obj.schema(),
                    database=database or self.current_database,
                    format=format,
                    external=external,
                    path=location,
                    partition=partition,
                )
            )
        return self.table(name, database=database or self.current_database)

    def avro_file(
        self, directory, avro_schema, name=None, database=None, external=True
    ):
        """Create a table to read a collection of Avro data.

        Parameters
        ----------
        directory
            Server path to directory containing avro files
        avro_schema
            The Avro schema for the data as a Python dict
        name
            Table name
        database
            Database name
        external
            Whether the table is external

        Returns
        -------
        ImpalaTable
            Impala table expression

        """
        name, database = self._get_concrete_table_path(name, database)

        stmt = ddl.CreateTableAvro(
            name, directory, avro_schema, database=database, external=external
        )
        self._safe_exec_sql(stmt)
        return self._wrap_new_table(name, database)

    def delimited_file(
        self,
        directory,
        schema,
        name=None,
        database=None,
        delimiter=",",
        na_rep=None,
        escapechar=None,
        lineterminator=None,
        external=True,
    ):
        """Interpret delimited text files as an Ibis table expression.

        See the `parquet_file` method for more details on what happens under
        the hood.

        Parameters
        ----------
        directory
            Server directory containing delimited text files
        schema
            Ibis schema
        name
            Name for the table; otherwise random names are generated
        database
            Database to create the table in
        delimiter
            Character used to delimit columns
        na_rep
            Character used to represent NULL values
        escapechar
            Character used to escape special characters
        lineterminator
            Character used to delimit lines
        external
            Create table as EXTERNAL (data will not be deleted on drop).

        Returns
        -------
        ImpalaTable
            Impala table expression

        """
        name, database = self._get_concrete_table_path(name, database)

        stmt = ddl.CreateTableDelimited(
            name,
            directory,
            schema,
            database=database,
            delimiter=delimiter,
            external=external,
            na_rep=na_rep,
            lineterminator=lineterminator,
            escapechar=escapechar,
        )
        self._safe_exec_sql(stmt)
        return self._wrap_new_table(name, database)

    def parquet_file(
        self,
        directory: str | Path,
        schema: sch.Schema | None = None,
        name: str | None = None,
        database: str | None = None,
        external: bool = True,
        like_file: str | Path | None = None,
        like_table: str | None = None,
    ):
        """Create an Ibis table from the passed directory of Parquet files.

        The table can be optionally named, otherwise a unique name will be
        generated.

        Parameters
        ----------
        directory
            Path
        schema
            If no schema provided, and neither of the like_* argument is
            passed, one will be inferred from one of the parquet files in the
            directory.
        like_file
            Absolute path to Parquet file on the server to use for schema
            definitions. An alternative to having to supply an explicit schema
        like_table
            Fully scoped and escaped string to an Impala table whose schema we
            will use for the newly created table.
        name
            Random unique name generated otherwise
        database
            Database to create the (possibly temporary) table in
        external
            If a table is external, the referenced data will not be deleted
            when the table is dropped in Impala. Otherwise Impala takes
            ownership of the Parquet file.

        Returns
        -------
        ImpalaTable
            Impala table expression

        """
        name, database = self._get_concrete_table_path(name, database)

        stmt = ddl.CreateTableParquet(
            name,
            directory,
            schema=schema,
            database=database,
            example_file=like_file,
            example_table=like_table,
            external=external,
            can_exist=False,
        )
        self._safe_exec_sql(stmt)
        return self._wrap_new_table(name, database)

    def _get_concrete_table_path(
        self, name: str | None, database: str | None
    ) -> tuple[str, str | None]:
        return name if name is not None else util.gen_name("impala_table"), database

    def _ensure_temp_db_exists(self):
        # TODO: session memoize to avoid unnecessary `SHOW DATABASES` calls
        name, path = options.impala.temp_db, options.impala.temp_path
        if name not in self.list_databases():
            self.create_database(name, path=path, force=True)

    def _drop_table(self, name: str) -> None:
        # database might have been dropped, so we suppress the
        # corresponding Exception
        with contextlib.suppress(ImpylaError):
            self.drop_table(name)

    def _wrap_new_table(self, name, database):
        qualified_name = self._fully_qualified_name(name, database)
        t = self.table(name, database=database)

        # Compute number of rows in table for better default query planning
        cardinality = t.count().execute()
        self._safe_exec_sql(
            f"ALTER TABLE {qualified_name} SET tblproperties('numRows'='{cardinality:d}', "
            "'STATS_GENERATED_VIA_STATS_TASK' = 'true')"
        )

        return t

    def insert(
        self,
        table_name,
        obj=None,
        database=None,
        overwrite=False,
        partition=None,
        values=None,
        validate=True,
    ):
        """Insert data into an existing table.

        See
        [`ImpalaTable.insert`](../backends/impala.qmd#ibis.backends.impala.client.ImpalaTable.insert)
        for parameters.

        Examples
        --------
        >>> table = "my_table"
        >>> con.insert(table, table_expr)  # quartodoc: +SKIP # doctest: +SKIP

        Completely overwrite contents
        >>> con.insert(table, table_expr, overwrite=True)  # quartodoc: +SKIP # doctest: +SKIP

        """
        if isinstance(obj, ir.Table):
            self._run_pre_execute_hooks(obj)
        table = self.table(table_name, database=database)
        return table.insert(
            obj=obj,
            overwrite=overwrite,
            partition=partition,
            values=values,
            validate=validate,
        )

    def drop_table(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop an Impala table.

        Parameters
        ----------
        name
            Table name
        database
            Database name
        force
            Database may throw exception if table does not exist

        Examples
        --------
        >>> table = "my_table"
        >>> db = "operations"
        >>> con.drop_table(table, database=db, force=True)  # quartodoc: +SKIP # doctest: +SKIP

        """
        statement = DropTable(name, database=database, must_exist=not force)
        self._safe_exec_sql(statement)

    def truncate_table(self, name: str, database: str | None = None) -> None:
        """Delete all rows from an existing table.

        Parameters
        ----------
        name
            Table name
        database
            Database name

        """
        statement = TruncateTable(name, database=database)
        self._safe_exec_sql(statement)

    def rename_table(self, old_name: str, new_name: str) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.

        """
        statement = RenameTable(old_name, new_name)
        self._safe_exec_sql(statement)

    def drop_table_or_view(self, name, *, database=None, force=False):
        """Drop view or table."""
        try:
            self.drop_table(name, database=database)
        except Exception as e:  # noqa: BLE001
            try:
                self.drop_view(name, database=database)
            except Exception:  # noqa: BLE001
                raise e

    def cache_table(self, table_name, *, database=None, pool="default"):
        """Caches a table in cluster memory in the given pool.

        Parameters
        ----------
        table_name
            Table name
        database
            Database name
        pool
            The name of the pool in which to cache the table

        Examples
        --------
        >>> table = "my_table"
        >>> db = "operations"
        >>> pool = "op_4GB_pool"
        >>> con.cache_table("my_table", database=db, pool=pool)  # quartodoc: +SKIP # doctest: +SKIP

        """
        statement = ddl.CacheTable(table_name, database=database, pool=pool)
        self._safe_exec_sql(statement)

    def create_function(self, func, name=None, database=None):
        """Create a function within Impala.

        Parameters
        ----------
        func
            UDF or UDAF
        name
            Function name
        database
            Database name

        """
        if name is None:
            name = func.name
        database = database or self.current_database

        if isinstance(func, udf.ImpalaUDF):
            stmt = ddl.CreateUDF(func, name=name, database=database)
        elif isinstance(func, udf.ImpalaUDA):
            stmt = ddl.CreateUDA(func, name=name, database=database)
        else:
            raise TypeError(func)
        self._safe_exec_sql(stmt)

    def drop_udf(
        self,
        name,
        input_types=None,
        database=None,
        force=False,
        aggregate=False,
    ):
        """Drop a UDF.

        If only name is given, this will search for the relevant UDF and drop
        it. To delete an overloaded UDF, give only a name and force=True

        Parameters
        ----------
        name
            Function name
        input_types
            Input types
        force
            Must be set to `True` to drop overloaded UDFs
        database
            Database name
        aggregate
            Whether the function is an aggregate

        """
        if not input_types:
            if not database:
                database = self.current_database
            result = self.list_udfs(database=database, like=name)
            if len(result) > 1:
                if force:
                    for func in result:
                        self._drop_single_function(
                            func.name,
                            func.inputs,
                            database=database,
                            aggregate=aggregate,
                        )
                    return
                else:
                    raise com.DuplicateUDFError(name)
            elif len(result) == 1:
                func = result.pop()
                self._drop_single_function(
                    func.name,
                    func.inputs,
                    database=database,
                    aggregate=aggregate,
                )
                return
            else:
                raise com.MissingUDFError(name)
        self._drop_single_function(
            name, input_types, database=database, aggregate=aggregate
        )

    def drop_uda(self, name, input_types=None, database=None, force=False):
        """Drop an aggregate function."""
        return self.drop_udf(
            name, input_types=input_types, database=database, force=force
        )

    def _drop_single_function(self, name, input_types, database=None, aggregate=False):
        stmt = ddl.DropFunction(
            name,
            input_types,
            must_exist=False,
            aggregate=aggregate,
            database=database,
        )
        self._safe_exec_sql(stmt)

    def _drop_all_functions(self, database):
        udfs = self.list_udfs(database=database)
        for fnct in udfs:
            stmt = ddl.DropFunction(
                fnct.name,
                fnct.inputs,
                must_exist=False,
                aggregate=False,
                database=database,
            )
            self._safe_exec_sql(stmt)
        udafs = self.list_udas(database=database)
        for udaf in udafs:
            stmt = ddl.DropFunction(
                udaf.name,
                udaf.inputs,
                must_exist=False,
                aggregate=True,
                database=database,
            )
            self._safe_exec_sql(stmt)

    def list_udfs(self, database=None, like=None):
        """Lists all UDFs associated with given database."""
        if not database:
            database = self.current_database
        statement = ddl.ListFunction(database, like=like, aggregate=False)
        with self._safe_raw_sql(statement) as cur:
            return self._get_udfs(cur)

    def list_udas(self, database=None, like=None):
        """Lists all UDAFs associated with a given database."""
        if not database:
            database = self.current_database
        statement = ddl.ListFunction(database, like=like, aggregate=True)
        with self._safe_raw_sql(statement) as cur:
            return self._get_udfs(cur)

    def _get_udfs(self, cur):
        rows = fetchall(cur)

        if rows.empty:
            return []

        current_database = self.current_database
        type_mapper = self.compiler.type_mapper
        result = []
        for return_type, signature, *_ in rows.itertuples(index=False):
            anon = sg.parse_one(signature)
            name = anon.this
            inputs = [
                type_mapper.from_string(expr.this.this) for expr in anon.expressions
            ]

            output = type_mapper.from_string(return_type)

            result.append((current_database, name, tuple(inputs), output))
        return result

    def exists_udf(self, name: str, database: str | None = None) -> bool:
        """Checks if a given UDF exists within a specified database."""
        return bool(self.list_udfs(database=database, like=name))

    def exists_uda(self, name: str, database: str | None = None) -> bool:
        """Checks if a given UDAF exists within a specified database."""
        return bool(self.list_udas(database=database, like=name))

    def compute_stats(
        self,
        name: str,
        database: str | None = None,
        incremental: bool = False,
    ) -> None:
        """Issue a `COMPUTE STATS` command for a given table.

        Parameters
        ----------
        name
            Can be fully qualified (with database name)
        database
            Database name
        incremental
            If True, issue COMPUTE INCREMENTAL STATS

        """
        maybe_inc = "INCREMENTAL " if incremental else ""
        cmd = f"COMPUTE {maybe_inc}STATS"

        stmt = self._table_command(cmd, name, database=database)
        self._safe_exec_sql(stmt)

    def invalidate_metadata(
        self,
        name: str | None = None,
        database: str | None = None,
    ) -> None:
        """Issue an `INVALIDATE METADATA` command.

        Optionally this applies to a specific table. See Impala documentation.

        Parameters
        ----------
        name
            Table name. Can be fully qualified (with database)
        database
            Database name

        """
        stmt = "INVALIDATE METADATA"
        if name is not None:
            stmt = self._table_command(stmt, name, database=database)
        self._safe_exec_sql(stmt)

    def refresh(self, name: str, database: str | None = None) -> None:
        """Reload metadata for a table.

        This can be useful after ingesting data as part of an ETL pipeline, for
        example.

        Related to `INVALIDATE METADATA`. See Impala documentation for more.

        Parameters
        ----------
        name
            Table name. Can be fully qualified (with database)
        database
            Database name

        """
        # TODO(wesm): can this statement be cancelled?
        stmt = self._table_command("REFRESH", name, database=database)
        self._safe_exec_sql(stmt)

    def describe_formatted(
        self,
        name: str,
        database: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve the results of a `DESCRIBE FORMATTED` command.

        See Impala documentation for more.

        Parameters
        ----------
        name
            Table name. Can be fully qualified (with database)
        database
            Database name

        """
        from ibis.backends.impala.metadata import parse_metadata

        stmt = self._table_command("DESCRIBE FORMATTED", name, database=database)
        result = self._exec_statement(stmt)

        # Leave formatting to pandas
        for c in result.columns:
            result[c] = result[c].str.strip()

        return parse_metadata(result)

    def show_files(
        self,
        name: str,
        database: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve results of a `SHOW FILES` command for a table.

        See Impala documentation for more.

        Parameters
        ----------
        name
            Table name. Can be fully qualified (with database)
        database
            Database name

        """
        stmt = self._table_command("SHOW FILES IN", name, database=database)
        return self._exec_statement(stmt)

    def list_partitions(self, name, database=None):
        stmt = self._table_command("SHOW PARTITIONS", name, database=database)
        return self._exec_statement(stmt)

    def table_stats(self, name, database=None):
        """Return results of `SHOW TABLE STATS` for the table `name`."""
        stmt = self._table_command("SHOW TABLE STATS", name, database=database)
        return self._exec_statement(stmt)

    def column_stats(self, name, database=None):
        """Return results of `SHOW COLUMN STATS` for the table `name`."""
        stmt = self._table_command("SHOW COLUMN STATS", name, database=database)
        return self._exec_statement(stmt)

    def _exec_statement(self, stmt):
        with self._safe_raw_sql(stmt) as cur:
            return fetchall(cur)

    def _table_command(self, cmd, name, database=None):
        qualified_name = self._fully_qualified_name(name, database)
        return f"{cmd} {qualified_name}"

    def to_pyarrow(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        from ibis.formats.pyarrow import PyArrowData

        self._run_pre_execute_hooks(expr)

        table_expr = expr.as_table()
        output = pa.Table.from_pandas(
            self.execute(table_expr, params=params, limit=limit, **kwargs),
            preserve_index=False,
        )
        table = PyArrowData.convert_table(output, table_expr.schema())
        return expr.__pyarrow_result__(table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1000000,
        **kwargs: Any,
    ) -> pa.ipc.RecordBatchReader:
        pa = self._import_pyarrow()
        self._run_pre_execute_hooks(expr)

        pa_table = self.to_pyarrow(
            expr.as_table(), params=params, limit=limit, **kwargs
        )
        return pa.RecordBatchReader.from_batches(
            pa_table.schema, pa_table.to_batches(max_chunksize=chunk_size)
        )

    def explain(
        self, expr: ir.Expr | str, params: Mapping[ir.Expr, Any] | None = None
    ) -> str:
        """Explain an expression.

        Return the query plan associated with the indicated expression or SQL
        query.

        Returns
        -------
        str
            Query plan

        """
        query = self.compile(expr, params=params)
        statement = f"EXPLAIN {query}"

        with self._safe_raw_sql(statement) as cur:
            results = fetchall(cur)

        return "\n".join(["Query:", util.indent(query, 2), "", *results.iloc[:, 0]])

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        schema = op.schema
        if null_columns := [col for col, dtype in schema.items() if dtype.is_null()]:
            raise com.IbisTypeError(
                "Impala cannot yet reliably handle `null` typed columns; "
                f"got null typed columns: {null_columns}"
            )

        # only register if we haven't already done so
        if (name := op.name) not in self.list_tables():
            type_mapper = self.compiler.type_mapper
            quoted = self.compiler.quoted
            column_defs = [
                sg.exp.ColumnDef(
                    this=sg.to_identifier(colname, quoted=quoted),
                    kind=type_mapper.from_ibis(typ),
                    # we don't support `NOT NULL` constraints in trino because
                    # because each trino connector differs in whether it
                    # supports nullability constraints, and whether the
                    # connector supports it isn't visible to ibis via a
                    # metadata query
                )
                for colname, typ in schema.items()
            ]

            create_stmt = sg.exp.Create(
                kind="TABLE",
                this=sg.exp.Schema(
                    this=sg.to_identifier(name, quoted=quoted), expressions=column_defs
                ),
            ).sql(self.name, pretty=True)

            data = op.data.to_frame().itertuples(index=False)
            specs = ", ".join("?" * len(schema))
            table = sg.table(name, quoted=quoted).sql(self.name)
            insert_stmt = f"INSERT INTO {table} VALUES ({specs})"
            with self._safe_raw_sql(create_stmt) as cur:
                for row in data:
                    cur.execute(insert_stmt, row)


def fetchall(cur):
    batches = cur.fetchcolumnar()
    names = list(map(operator.itemgetter(0), cur.description))
    df = _column_batches_to_dataframe(names, batches)
    return df


def _column_batches_to_dataframe(names, batches):
    import pandas as pd

    cols = {
        name: _chunks_to_pandas_array(chunks)
        for name, chunks in zip(names, zip(*(b.columns for b in batches)))
    }

    return pd.DataFrame(cols, columns=names)


_HS2_TTypeId_to_dtype = {
    "BOOLEAN": "bool",
    "TINYINT": "int8",
    "SMALLINT": "int16",
    "INT": "int32",
    "BIGINT": "int64",
    "TIMESTAMP": "datetime64[ns]",
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "STRING": "object",
    "DECIMAL": "object",
    "BINARY": "object",
    "VARCHAR": "object",
    "CHAR": "object",
    "DATE": "datetime64[ns]",
    "VOID": None,
}


def _chunks_to_pandas_array(chunks):
    import numpy as np

    total_length = 0
    have_nulls = False
    for c in chunks:
        total_length += len(c)
        have_nulls = have_nulls or c.nulls.any()

    type_ = chunks[0].data_type
    numpy_type = _HS2_TTypeId_to_dtype[type_]

    def fill_nonnull(target, chunks):
        pos = 0
        for c in chunks:
            target[pos : pos + len(c)] = c.values
            pos += len(c.values)

    def fill(target, chunks, na_rep):
        pos = 0
        for c in chunks:
            nulls = c.nulls.copy()
            nulls.bytereverse()
            bits = np.frombuffer(nulls.tobytes(), dtype="u1")
            mask = np.unpackbits(bits).view(np.bool_)

            k = len(c)

            dest = target[pos : pos + k]
            dest[:] = c.values
            dest[mask[:k]] = na_rep

            pos += k

    if have_nulls:
        if numpy_type in ("bool", "datetime64[ns]"):
            target = np.empty(total_length, dtype="O")
            na_rep = np.nan
        elif numpy_type.startswith("int"):
            target = np.empty(total_length, dtype="f8")
            na_rep = np.nan
        else:
            target = np.empty(total_length, dtype=numpy_type)
            na_rep = np.nan

        fill(target, chunks, na_rep)
    else:
        target = np.empty(total_length, dtype=numpy_type)
        fill_nonnull(target, chunks)

    return target
