from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import pyflink.version
import sqlglot as sg
from pyflink.table.types import create_arrow_schema

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
from ibis.backends.base import BaseBackend, CanListDatabases
from ibis.backends.base.sql.ddl import fully_qualified_re, is_fully_qualified
from ibis.backends.flink.compiler.core import FlinkCompiler

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    import pyarrow as pa
    from pyflink.table import TableEnvironment

    import ibis.expr.types as ir


class Backend(BaseBackend, CanListDatabases):
    name = "flink"
    compiler = FlinkCompiler
    supports_temporary_tables = True
    supports_python_udfs = True

    def do_connect(self, table_env: TableEnvironment) -> None:
        """Create a Flink `Backend` for use with Ibis.

        Parameters
        ----------
        table_env
            A table environment

        Examples
        --------
        >>> import ibis
        >>> from pyflink.table import EnvironmentSettings, TableEnvironment
        >>> table_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        >>> ibis.flink.connect(table_env)
        <ibis.backends.flink.Backend at 0x...>
        """
        self._table_env = table_env

    def list_databases(self, like: str | None = None) -> list[str]:
        databases = self._table_env.list_databases()
        return self._filter_with_like(databases, like)

    @property
    def current_database(self) -> str:
        return self._table_env.get_current_database()

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        tables = self._table_env._j_tenv.listTables(
            self._table_env.get_current_catalog(), database or self.current_database
        )
        return self._filter_with_like(tables, like)

    def _fully_qualified_name(self, name: str, database: str | None) -> str:
        if is_fully_qualified(name):
            return name

        return sg.table(name, db=database or self.current_database).sql(dialect="hive")

    def table(self, name: str, database: str | None = None) -> ir.Table:
        """Return a table expression from a table or view in the database.

        Parameters
        ----------
        name
            Table name
        database
            Database in which the table resides

        Returns
        -------
        Table
            Table named `name` from `database`
        """
        if database is not None and not isinstance(database, str):
            raise exc.IbisTypeError(
                f"`database` must be a string; got {type(database)}"
            )
        schema = self.get_schema(name, database=database)
        qualified_name = self._fully_qualified_name(name, database)
        _, quoted, unquoted = fully_qualified_re.search(qualified_name).groups()
        unqualified_name = quoted or unquoted
        node = ops.DatabaseTable(unqualified_name, schema, self, namespace=database)
        return node.to_expr()

    def get_schema(self, table_name: str, database: str | None = None) -> sch.Schema:
        """Return a Schema object for the indicated table and database.

        Parameters
        ----------
        table_name
            Table name
        database
            Database name

        Returns
        -------
        sch.Schema
            Ibis schema
        """
        qualified_name = self._fully_qualified_name(table_name, database)
        table = self._table_env.from_path(qualified_name)
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
        df = self._table_env.sql_query(sql).to_pandas()
        return expr.__pandas_result__(df)

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new table in Flink.

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
        if isinstance(obj, pa.Table):
            obj = obj.to_pandas()
        if isinstance(obj, pd.DataFrame):
            qualified_name = self._fully_qualified_name(name, database)
            table = self._table_env.from_pandas(obj)
            # FIXME(deepyaman): Create a catalog table, not a temp view.
            self._table_env.create_temporary_view(qualified_name, table)
        else:
            raise NotImplementedError  # TODO(deepyaman)

        return self.table(name, database=database)

    def drop_table(
        self,
        name: str,
        *,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table.

        Parameters
        ----------
        name
            Name of the table to drop.
        database
            Name of the database where the table exists, if not the default.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        qualified_name = self._fully_qualified_name(name, database)
        if not (self._table_env.drop_temporary_table(qualified_name) or force):
            raise exc.IntegrityError(f"Table {name} does not exist.")

        # TODO(deepyaman): Support (and differentiate) permanent tables.

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
