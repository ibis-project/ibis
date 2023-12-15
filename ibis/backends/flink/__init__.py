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
    CreateView,
    DropDatabase,
    DropTable,
    DropView,
    InsertSelect,
    RenameTable,
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
        temp: bool = False,
        database: str | None = None,
        catalog: str | None = None,
    ) -> list[str]:
        """Return the list of table/view names.

        Return the list of table/view names in the `database` and `catalog`. If
        `database`/`catalog` are not specified, their default values will be
        used. Temporary tables can only be listed for the default database and
        catalog, hence `database` and `catalog` are ignored if `temp` is True.

        Parameters
        ----------
        like : str, optional
            A pattern in Python's regex format.
        temp : bool, optional
            Whether to list temporary tables or permanent tables.
        database : str, optional
            The database to list tables of, if not the current one.
        catalog : str, optional
            The catalog to list tables of, if not the current one.

        Returns
        -------
        list[str]
            The list of the table/view names that match the pattern `like`.
        """
        catalog = catalog or self.current_catalog
        database = database or self.current_database

        # The following is equivalent to the SQL query string `SHOW TABLES FROM|IN`,
        # but executing the SQL string directly yields a `TableResult` object
        if temp:
            # Note (mehmet): TableEnvironment does not provide a function to list
            # the temporary tables in a given catalog and database.
            # Ref: https://nightlies.apache.org/flink/flink-docs-master/api/java/org/apache/flink/table/api/TableEnvironment.html
            tables = self._table_env.list_temporary_tables()
        else:
            # Note (mehmet): `listTables` returns both tables and views.
            # Ref: Docstring for pyflink/table/table_environment.py:list_tables()
            tables = self._table_env._j_tenv.listTables(catalog, database)

        return self._filter_with_like(tables, like)

    def list_views(
        self,
        like: str | None = None,
        temp: bool = False,
    ) -> list[str]:
        """Return the list of view names.

        Return the list of view names.

        Parameters
        ----------
        like : str, optional
            A pattern in Python's regex format.
        temp : bool, optional
            Whether to list temporary views or permanent views.

        Returns
        -------
        list[str]
            The list of the view names that match the pattern `like`.
        """

        if temp:
            views = self._table_env.list_temporary_views()
        else:
            views = self._table_env.list_views()

        return self._filter_with_like(views, like)

    def _fully_qualified_name(
        self,
        name: str,
        database: str | None = None,
        catalog: str | None = None,
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

        In Flink, tables can be either virtual (VIEWS) or regular (TABLES).
        VIEWS can be created from an existing Table object, usually the result
        of a Table API or SQL query. TABLES describe external data, such as a
        file, database table, or message queue. In other words, TABLES refer
        explicitly to tables constructed directly from source/sink connectors.

        When `obj` is in-memory (e.g., Dataframe), currently this function can
        create only a TEMPORARY VIEW. If `obj` is in-memory and `temp` is False,
        it will raise an error.

        Parameters
        ----------
        name
            Name of the new table.
        obj
            An Ibis table expression, pandas DataFrame, or PyArrow Table that will
            be used to extract the schema and the data of the new table. An
            optional `schema` can be used to override the schema.
        schema
            The schema for the new table. Required if `obj` is not provided.
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
        import pyarrow_hotfix  # noqa: F401

        import ibis.expr.types as ir

        if obj is None and schema is None:
            raise exc.IbisError("`schema` or `obj` is required")
        if isinstance(obj, (pd.DataFrame, pa.Table)) and not temp:
            raise exc.IbisError(
                "`temp` cannot be False when `obj` is in-memory. "
                "Currently can create only TEMPORARY VIEW for in-memory data."
            )

        if overwrite:
            if self.list_tables(like=name, temp=temp):
                self.drop_table(
                    name=name,
                    catalog=catalog,
                    database=database,
                    temp=temp,
                    force=True,
                )

        # In-memory data is created as views in `pyflink`
        if obj is not None:
            if isinstance(obj, pd.DataFrame):
                dataframe = obj

            elif isinstance(obj, pa.Table):
                dataframe = obj.to_pandas()

            elif isinstance(obj, ir.Table):
                # Note (mehmet): If obj points to in-memory data, we create a view.
                # Other cases are unsupported for now, e.g., obj is of UnboundTable.
                # See TODO right below for more context on how we handle in-memory data.
                op = obj.op()
                if isinstance(op, ops.InMemoryTable):
                    dataframe = op.data.to_frame()
                else:
                    raise exc.IbisError(
                        "`obj` is of type ibis.expr.types.Table but it is not in-memory. "
                        "Currently, only in-memory tables are supported. "
                        "See ibis.memtable() for info on creating in-memory table."
                    )
            else:
                raise exc.IbisError(f"Unsupported `obj` type: {type(obj)}")

            # TODO (mehmet): Flink requires a source connector to create regular tables.
            # In-memory data can only be created as a view (virtual table). So we decided
            # to create views for in-memory data. Ideally, this function should only create
            # tables. However, for that, we would need the notion of a "default" table,
            # which may not be ideal. We plan to get back to this later.
            # Ref: https://github.com/ibis-project/ibis/pull/7479#discussion_r1416237088
            return self.create_view(
                name=name,
                obj=dataframe,
                schema=schema,
                database=database,
                catalog=catalog,
                temp=temp,
                overwrite=overwrite,
            )

        # External data is created as tables in `pyflink`
        else:  # obj is None, schema is not None
            if not tbl_properties:
                raise exc.IbisError(
                    "`tbl_properties` is required when creating table with schema"
                )
            elif (
                "connector" not in tbl_properties or tbl_properties["connector"] is None
            ):
                raise exc.IbisError("connector must be defined in `tbl_properties`")

            # TODO (mehmet): Given that we rely on default catalog if one is not specified,
            # is there any point to support temporary tables?
            statement = CreateTableFromConnector(
                table_name=name,
                schema=schema,
                tbl_properties=tbl_properties,
                watermark=watermark,
                temporary=temp,
                database=database,
                catalog=catalog,
            )
            sql = statement.compile()
            self._exec_sql(sql)

            return self.table(name, database=database, catalog=catalog)

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
            Whether the table is temporary or not.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        statement = DropTable(
            table_name=name,
            database=database,
            catalog=catalog,
            must_exist=not force,
            temporary=temp,
        )
        self._exec_sql(statement.compile())

    def rename_table(
        self,
        old_name: str,
        new_name: str,
        force: bool = True,
    ) -> None:
        """Rename an existing table.

        Parameters
        ----------
        old_name
            The old name of the table.
        new_name
            The new name of the table.
        force
            If `False`, an exception is raised if the table does not exist.
        """
        statement = RenameTable(
            old_name=old_name,
            new_name=new_name,
            must_exist=not force,
        )
        sql = statement.compile()
        self._exec_sql(sql)

    def create_view(
        self,
        name: str,
        obj: pd.DataFrame | ir.Table,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        catalog: str | None = None,
        force: bool = False,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a new view from a dataframe or table.

        When `obj` is in-memory (e.g., Dataframe), currently this function can
        create only a TEMPORARY VIEW. If `obj` is in-memory and `temp` is False,
        it will raise an error.

        Parameters
        ----------
        name
            Name of the new view.
        obj
            An Ibis table expression that will be used to create the view.
        schema
            The schema for the new view.
        database
            Name of the database where the view will be created, if not
            provided the database's default is used.
        catalog
            Name of the catalog where the table exists, if not the default.
        force
            If `False`, an exception is raised if the table is already present.
        temp
            Whether the table is temporary or not.
        overwrite
            If `True`, remove the existing view, and create a new one.

        Returns
        -------
        Table
            The view that was created.
        """
        import pandas as pd

        from ibis.backends.flink.datatypes import FlinkRowSchema

        if isinstance(obj, pd.DataFrame) and not temp:
            raise exc.IbisError(
                "`temp` cannot be False when `obj` is in-memory. "
                "Currently supports creating only temporary view for in-memory data."
            )

        if overwrite and self.list_views(like=name, temp=temp):
            self.drop_view(
                name=name,
                database=database,
                catalog=catalog,
                temp=temp,
                force=True,
            )

        if isinstance(obj, pd.DataFrame):
            qualified_name = self._fully_qualified_name(name, database, catalog)
            if schema:
                table = self._table_env.from_pandas(
                    obj, FlinkRowSchema.from_ibis(schema)
                )
            else:
                table = self._table_env.from_pandas(obj)
            # Note (mehmet): We use `create_temporary_view` here instead of `register_table`
            # as suggested in PyFlink source code due to the deprecation of `register_table`.
            self._table_env.create_temporary_view(qualified_name, table)

        elif isinstance(obj, ir.Table):
            query_expression = self.compile(obj)
            statement = CreateView(
                name=name,
                query_expression=query_expression,
                database=database,
                can_exist=force,
                temporary=temp,
            )
            sql = statement.compile()
            self._exec_sql(sql)

        else:
            raise exc.IbisError(f"Unsupported `obj` type: {type(obj)}")

        return self.table(name=name, database=database, catalog=catalog)

    def drop_view(
        self,
        name: str,
        *,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
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
        temp
            Whether the view is temporary or not.
        force
            If `False`, an exception is raised if the view does not exist.
        """
        # TODO(deepyaman): Support (and differentiate) permanent views.

        statement = DropView(
            name=name,
            database=database,
            catalog=catalog,
            must_exist=(not force),
            temporary=temp,
        )
        sql = statement.compile()
        self._exec_sql(sql)

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
        import pyarrow_hotfix  # noqa: F401

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
