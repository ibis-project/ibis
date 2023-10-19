from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import sqlglot as sg

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, CanCreateDatabase
from ibis.backends.base.sql.ddl import fully_qualified_re, is_fully_qualified
from ibis.backends.flink.compiler.core import FlinkCompiler
from ibis.backends.flink.ddl import (
    CreateDatabase,
    CreateTableFromConnector,
    DropDatabase,
    DropTable,
    InsertSelect,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import pyarrow as pa
    from pyflink.table import TableEnvironment
    from pyflink.table.table_result import TableResult

    from ibis.api import Watermark


class Backend(BaseBackend, CanCreateDatabase):
    name = "flink"
    compiler = FlinkCompiler
    supports_temporary_tables = True
    supports_python_udfs = True

    def do_connect(self, table_env: TableEnvironment) -> None:
        """Create a Flink `Backend` for use with Ibis.

        Parameters
        ----------
        table_env
            A table environment.

        Examples
        --------
        >>> import ibis
        >>> from pyflink.table import EnvironmentSettings, TableEnvironment
        >>> table_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        >>> ibis.flink.connect(table_env)
        <ibis.backends.flink.Backend at 0x...>
        """
        self._table_env = table_env

    def _exec_sql(self, query: str) -> TableResult:
        return self._table_env.execute_sql(query)

    def list_databases(self, like: str | None = None) -> list[str]:
        databases = self._table_env.list_databases()
        return self._filter_with_like(databases, like)

    @property
    def current_catalog(self) -> str:
        return self._table_env.get_current_catalog()

    @property
    def current_database(self) -> str:
        return self._table_env.get_current_database()

    def create_database(
        self,
        name: str,
        db_properties: dict | None = None,
        catalog: str | None = None,
        force: bool = False,
    ) -> None:
        """Create a new database.

        Parameters
        ----------
        name : str
            Name of the new database.
        db_properties : dict, optional
            Properties of the database. Accepts dictionary of key-value pairs
            (key1=val1, key2=val2, ...).
        catalog : str, optional
            Name of the catalog in which the new database will be created.
        force : bool, optional
            If `False`, an exception is raised if the database already exists.
        """
        statement = CreateDatabase(
            name=name, db_properties=db_properties, catalog=catalog, can_exist=force
        )
        self._exec_sql(statement.compile())

    def drop_database(
        self, name: str, catalog: str | None = None, force: bool = False
    ) -> None:
        """Drop a database with name `name`.

        Parameters
        ----------
        name : str
            Database to drop.
        catalog : str, optional
            Name of the catalog from which the database will be dropped.
        force : bool, optional
            If `False`, an exception is raised if the database does not exist.
        """
        statement = DropDatabase(name=name, catalog=catalog, must_exist=not force)
        self._exec_sql(statement.compile())

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
    ) -> list[str]:
        """Return the list of table names.

        Return the list of table names in the specified database and catalog.
        or the default one if no database/catalog is specified.

        Parameters
        ----------
        like : str, optional
            A pattern in Python's regex format.
        database : str, optional
            The database to list tables of, if not the current one.
        catalog : str, optional
            The catalog to list tables of, if not the current one.

        Returns
        -------
        list[str]
            The list of the table names that match the pattern `like`.
        """
        tables = self._table_env._j_tenv.listTables(
            catalog or self.current_catalog,
            database or self.current_database,
        )  # this is equivalent to the SQL query string `SHOW TABLES FROM|IN`,
        # but executing the SQL string directly yields a `TableResult` object
        return self._filter_with_like(tables, like)

    def _fully_qualified_name(
        self,
        name: str,
        database: str | None,
        catalog: str | None,
    ) -> str:
        if is_fully_qualified(name):
            return name

        return sg.table(
            name,
            db=database or self.current_database,
            catalog=catalog or self.current_catalog,
        ).sql(dialect="hive")

    def table(
        self,
        name: str,
        database: str | None = None,
        catalog: str | None = None,
    ) -> ir.Table:
        """Return a table expression from a table or view in the database.

        Parameters
        ----------
        name
            Table name.
        database
            Database in which the table resides.
        catalog
            Catalog in which the table resides.

        Returns
        -------
        Table
            Table named `name` from `database`
        """
        if database is not None and not isinstance(database, str):
            raise exc.IbisTypeError(
                f"`database` must be a string; got {type(database)}"
            )
        schema = self.get_schema(name, catalog=catalog, database=database)
        qualified_name = self._fully_qualified_name(name, catalog, database)
        _, quoted, unquoted = fully_qualified_re.search(qualified_name).groups()
        unqualified_name = quoted or unquoted
        node = ops.DatabaseTable(
            unqualified_name,
            schema,
            self,
            namespace=ops.Namespace(schema=database, database=catalog),
        )  # TODO(chloeh13q): look into namespacing with catalog + db
        return node.to_expr()

    def get_schema(
        self,
        table_name: str,
        database: str | None = None,
        catalog: str | None = None,
    ) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name : str
            Table name.
        database : str, optional
            Database name.
        catalog : str, optional
            Catalog name.

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        from pyflink.table.types import create_arrow_schema

        qualified_name = self._fully_qualified_name(table_name, catalog, database)
        table = self._table_env.from_path(qualified_name)
        schema = table.get_schema()
        return sch.Schema.from_pyarrow(
            create_arrow_schema(schema.get_field_names(), schema.get_field_data_types())
        )

    @property
    def version(self) -> str:
        import pyflink.version

        return pyflink.version.__version__

    def compile(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Expr, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Compile an expression."""
        return super().compile(expr, params=params)  # Discard `limit` and other kwargs.

    def _to_sql(self, expr: ir.Expr, **kwargs: Any) -> str:
        return str(self.compile(expr, **kwargs))

    def execute(self, expr: ir.Expr, **kwargs: Any) -> Any:
        """Execute an expression."""
        table_expr = expr.as_table()
        sql = self.compile(table_expr, **kwargs)
        df = self._table_env.sql_query(sql).to_pandas()
        return expr.__pandas_result__(df)

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        catalog: str | None = None,
        tbl_properties: dict | None = None,
        watermark: Watermark | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table in Flink.

        Note that: in Flink, tables can be either virtual (VIEWS) or regular
        (TABLES).

        VIEWS can be created from an existing Table object, usually the result
        of a Table API or SQL query. TABLES describe external data, such as a
        file, database table, or message queue. In other words, TABLES refer
        explicitly to tables constructed directly from source/sink connectors.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression or pandas table that will be used to
            extract the schema and the data of the new table. If not provided,
            `schema` must be given.
        schema
            The schema for the new table. Only one of `schema` or `obj` can be
            provided.
        database
            Name of the database where the table will be created, if not the
            default.
        catalog
            Name of the catalog where the table will be created, if not the
            default.
        tbl_properties
            Table properties used to create a table source/sink. The properties
            are usually used to find and create the underlying connector. Accepts
            dictionary of key-value pairs (key1=val1, key2=val2, ...).
        watermark
            Watermark strategy for the table, only applicable on sources.
        temp
            Whether a table is temporary or not.
        overwrite
            Whether to clobber existing data.

        Returns
        -------
        Table
            The table that was created.
        """
        import pandas as pd
        import pyarrow as pa

        import ibis.expr.types as ir

        if obj is None and schema is None:
            raise exc.IbisError("The schema or obj parameter is required")

        if overwrite:
            self.drop_table(name=name, catalog=catalog, database=database, force=True)

        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if isinstance(obj, pd.DataFrame):
            qualified_name = self._fully_qualified_name(name, database, catalog)
            table = self._table_env.from_pandas(obj)
            # in-memory data is created as views in `pyflink`
            # TODO(chloeh13q): alternatively, we can do CREATE TABLE and then INSERT
            # INTO ... VALUES to keep things consistent
            self._table_env.create_temporary_view(qualified_name, table)
        if isinstance(obj, ir.Table):
            # TODO(chloeh13q): implement CREATE TABLE for expressions
            raise NotImplementedError
        if schema is not None:
            if not tbl_properties:
                raise exc.IbisError(
                    "tbl_properties is required when creating table with schema"
                )
            statement = CreateTableFromConnector(
                table_name=name,
                schema=schema,
                tbl_properties=tbl_properties,
                watermark=watermark,
                temp=temp,
                database=database,
                catalog=catalog,
            )
            self._exec_sql(statement.compile())

        return self.table(name, database=database)

    def drop_table(
        self,
        name: str,
        *,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop.
        database
            Name of the database where the table exists, if not the default.
        catalog
            Name of the catalog where the table exists, if not the default.
        temp
            Whether a table is temporary or not.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        statement = DropTable(
            table_name=name,
            database=database,
            catalog=catalog,
            must_exist=not force,
            temp=temp,
        )
        self._exec_sql(statement.compile())

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new view from an expression.

        Parameters
        ----------
        name
            Name of the new view.
        obj
            An Ibis table expression that will be used to create the view.
        database
            Name of the database where the view will be created, if not
            provided the database's default is used.
        overwrite
            Whether to clobber an existing view with the same name.

        Returns
        -------
        Table
            The view that was created.
        """
        raise NotImplementedError

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        catalog: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop.
        database
            Name of the database where the view exists, if not the default.
        catalog
            Name of the catalog where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        qualified_name = self._fully_qualified_name(name, database, catalog)
        if not (self._table_env.drop_temporary_view(qualified_name) or force):
            raise exc.IntegrityError(f"View {name} does not exist.")

        # TODO(deepyaman): Support (and differentiate) permanent views.

    @classmethod
    @lru_cache
    def _get_operations(cls):
        translator = cls.compiler.translator_class
        return translator._registry.keys()

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
        return operation in cls._get_operations()

    def insert(
        self,
        table_name: str,
        obj: pa.Table | pd.DataFrame | ir.Table | list | dict,
        database: str | None = None,
        catalog: str | None = None,
        overwrite: bool = False,
    ) -> TableResult:
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to insert data into.
        obj
            The source data or expression to insert.
        database
            Name of the attached database that the table is located in.
        catalog
            Name of the attached catalog that the table is located in.
        overwrite
            If `True` then replace existing contents of table.

        Returns
        -------
        TableResult
            The table result.

        Raises
        ------
        ValueError
            If the type of `obj` isn't supported
        """
        import pandas as pd
        import pyarrow as pa

        if isinstance(obj, ir.Table):
            expr = obj
            ast = self.compiler.to_ast(expr)
            select = ast.queries[0]
            statement = InsertSelect(
                table_name,
                select,
                database=database,
                catalog=catalog,
                overwrite=overwrite,
            )
            return self._exec_sql(statement.compile())

        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if isinstance(obj, dict):
            obj = pd.DataFrame.from_dict(obj)
        if isinstance(obj, pd.DataFrame):
            table = self._table_env.from_pandas(obj)
            return table.execute_insert(table_name, overwrite=overwrite)

        if isinstance(obj, list):
            # pyflink infers datatypes, which may sometimes result in incompatible types
            table = self._table_env.from_elements(obj)
            return table.execute_insert(table_name, overwrite=overwrite)

        raise ValueError(
            "No operation is being performed. Either the obj parameter "
            "is not a pandas DataFrame or is not a ibis Table."
            f"The given obj is of type {type(obj).__name__} ."
        )
