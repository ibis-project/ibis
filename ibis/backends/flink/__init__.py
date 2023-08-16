from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pyflink.version
import sqlglot as sg
from pyflink.table.types import create_arrow_schema

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import BaseBackend, CanCreateDatabase
from ibis.backends.base.sql.ddl import fully_qualified_re, is_fully_qualified
from ibis.backends.flink.compiler.core import FlinkCompiler
from ibis.backends.flink.ddl import CreateDatabase, CreateTableFromConnector, DropTable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import pyarrow as pa
    from pyflink.table import TableEnvironment

    import ibis.expr.types as ir


class Backend(BaseBackend, CanCreateDatabase):
    name = "flink"
    compiler = FlinkCompiler
    supports_temporary_tables = True
    supports_python_udfs = True

    def do_connect(self, t_env: TableEnvironment) -> None:
        """Create a Flink `Backend` for use with Ibis.

        Parameters
        ----------
        t_env
            A table environment

        Examples
        --------
        >>> import ibis
        >>> from pyflink.table import EnvironmentSettings, TableEnvironment
        >>> t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        >>> ibis.flink.connect(t_env)
        <ibis.backends.flink.Backend at 0x...>
        """
        self._t_env = t_env

    def _exec_sql(self, query: str) -> None:
        self._t_env.execute_sql(query)

    def list_databases(self, like: str | None = None) -> list[str]:
        databases = self._t_env.list_databases()
        return self._filter_with_like(databases, like)

    @property
    def current_catalog(self) -> str:
        return self._t_env.get_current_catalog()

    @property
    def current_database(self) -> str:
        return self._t_env.get_current_database()

    def create_database(
        self,
        name: str,
        db_properties: dict,
        catalog: str | None = None,
        force: bool = False,
    ) -> None:
        statement = CreateDatabase(
            name=name, db_properties=db_properties, catalog=catalog, can_exist=force
        )
        self._exec_sql(statement.compile())

    def drop_database(self, name: str, force: bool = False) -> None:
        raise NotImplementedError

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        catalog: str | None = None,
    ) -> list[str]:
        tables = self._t_env._j_tenv.listTables(
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
            Table name
        database
            Database in which the table resides
        catalog
            Catalog in which the table resides

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
            unqualified_name, schema, self, namespace=database
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
        table_name
            Table name
        database
            Database name
        catalog
            Catalog name

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        qualified_name = self._fully_qualified_name(table_name, catalog, database)
        table = self._t_env.from_path(qualified_name)
        schema = table.get_schema()
        return sch.Schema.from_pyarrow(
            create_arrow_schema(schema.get_field_names(), schema.get_field_data_types())
        )

    @property
    def version(self) -> str:
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
        df = self._t_env.sql_query(sql).to_pandas()

        # TODO: remove the extra conversion
        return expr.__pandas_result__(table_expr.__pandas_result__(df))

    def create_table(
        self,
        name: str,
        tbl_properties: dict,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table in Flink.

        Parameters
        ----------
        name
            Name of the new table.
        tbl_properties
            Table properties used to create a table source/sink. The properties
            are usually used to find and create the underlying connector. Accepts
            dictionary of key-value pairs (key1=val1, key2=val2, ...).
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
        temp
            Whether a table is temporary or not
        overwrite
            Whether to clobber existing data

        Returns
        -------
        Table
            The table that was created.
        """
        import pandas as pd
        import pyarrow as pa

        if obj is None and schema is None:
            raise exc.IbisError("The schema or obj parameter is required")

        if overwrite is True:
            self.drop_table(name=name, catalog=catalog, database=database, force=True)

        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if isinstance(obj, pd.DataFrame):
            qualified_name = self._fully_qualified_name(name, database)
            table = self._t_env.from_pandas(obj)
            # FIXME(deepyaman): Create a catalog table, not a temp view.
            self._t_env.create_temporary_view(qualified_name, table)

        if schema is not None:
            statement = CreateTableFromConnector(
                table_name=name,
                schema=schema,
                tbl_properties=tbl_properties,
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
            Whether to clobber an existing view with the same name

        Returns
        -------
        Table
            The view that was created.
        """
        raise NotImplementedError

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a view.

        Parameters
        ----------
        name
            Name of the view to drop.
        database
            Name of the database where the view exists, if not the default.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        qualified_name = self._fully_qualified_name(name, database)
        if not (self._t_env.drop_temporary_view(qualified_name) or force):
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
