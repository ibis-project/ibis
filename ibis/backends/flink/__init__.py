from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base import CanCreateDatabase, NoUrl
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.flink.compiler import FlinkCompiler
from ibis.backends.flink.ddl import (
    CreateDatabase,
    CreateTableWithSchema,
    DropDatabase,
    DropTable,
    DropView,
    InsertSelect,
    RenameTable,
)
from ibis.backends.tests.errors import Py4JJavaError
from ibis.util import gen_name

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    from pyflink.table import Table, TableEnvironment
    from pyflink.table.table_result import TableResult

    from ibis.expr.api import Watermark


class Backend(SQLGlotBackend, CanCreateDatabase, NoUrl):
    name = "flink"
    compiler = FlinkCompiler()
    supports_temporary_tables = True
    supports_python_udfs = True

    @property
    def dialect(self):
        # TODO: remove when ported to sqlglot
        return self.compiler.dialect

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

    def disconnect(self) -> None:
        pass

    def raw_sql(self, query: str) -> TableResult:
        return self._table_env.execute_sql(query)

    def _metadata(self, query: str):
        from pyflink.table.types import create_arrow_schema

        table = self._table_env.sql_query(query)
        schema = table.get_schema()
        pa_schema = create_arrow_schema(
            schema.get_field_names(), schema.get_field_data_types()
        )
        # sort of wasteful, but less code to write
        return sch.Schema.from_pyarrow(pa_schema).items()

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
        self.raw_sql(statement.compile())

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
        self.raw_sql(statement.compile())

    def list_tables(
        self,
        like: str | None = None,
        *,
        database: str | None = None,
        catalog: str | None = None,
        temp: bool = False,
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
        node = ops.DatabaseTable(
            name,
            schema=schema,
            source=self,
            namespace=ops.Namespace(schema=catalog, database=database),
        )
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

        from ibis.backends.flink.datatypes import get_field_data_types

        qualified_name = sg.table(table_name, db=catalog, catalog=database).sql(
            self.name
        )
        table = self._table_env.from_path(qualified_name)
        pyflink_schema = table.get_schema()

        return sch.Schema.from_pyarrow(
            create_arrow_schema(
                pyflink_schema.get_field_names(), get_field_data_types(pyflink_schema)
            )
        )

    @property
    def version(self) -> str:
        import pyflink.version

        return pyflink.version.__version__

    def compile(
        self, expr: ir.Expr, params: Mapping[ir.Expr, Any] | None = None, **_: Any
    ) -> Any:
        """Compile an Ibis expression to Flink."""
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
        primary_key: str | list[str] | None = None,
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
        primary_key
            A single column or a list of columns to be marked as primary. Raises
            an error if the column(s) in `primary_key` is NOT a subset of the
            columns in `schema`. Primary keys must be non-nullable in Flink and
            the columns indicated as primary key will be designated as non-nullable.
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
            statement = CreateTableWithSchema(
                table_name=name,
                schema=schema,
                tbl_properties=tbl_properties,
                watermark=watermark,
                primary_key=primary_key,
                temporary=temp,
                database=database,
                catalog=catalog,
            )
            sql = statement.compile()
            self.raw_sql(sql)

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
        self.raw_sql(statement.compile())

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
        self.raw_sql(sql)

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
            qualified_name = sg.table(
                name, db=database, catalog=catalog, quoted=self.compiler.quoted
            ).sql(self.name)
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
            stmt = sge.Create(
                kind="VIEW",
                this=sg.table(
                    name, db=database, catalog=catalog, quoted=self.compiler.quoted
                ),
                expression=query_expression,
                exists=force,
                properties=sge.Properties(expressions=[sge.TemporaryProperty()])
                if temp
                else None,
            )
            self.raw_sql(stmt.sql(self.name))

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
        self.raw_sql(sql)

    def _read_file(
        self,
        file_type: str,
        path: str | Path,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a file as a table in the current database.

        Parameters
        ----------
        file_type
            File type, e.g., parquet, csv, json.
        path
            The data source.
        schema
            The schema for the new table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table

        Raises
        ------
        ValueError
            If `schema` is None.

        """
        if schema is None:
            raise ValueError(
                f"`schema` must be explicitly provided when calling `read_{file_type}`"
            )

        table_name = table_name or gen_name(f"read_{file_type}")
        tbl_properties = {
            "connector": "filesystem",
            "path": path,
            "format": file_type,
        }

        return self.create_table(
            name=table_name,
            schema=schema,
            tbl_properties=tbl_properties,
        )

    def read_parquet(
        self,
        path: str | Path,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a parquet file as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        schema
            The schema for the new table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        return self._read_file(
            file_type="parquet", path=path, schema=schema, table_name=table_name
        )

    def read_csv(
        self,
        path: str | Path,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a csv file as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        schema
            The schema for the new table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        return self._read_file(
            file_type="csv", path=path, schema=schema, table_name=table_name
        )

    def read_json(
        self,
        path: str | Path,
        schema: sch.Schema | None = None,
        table_name: str | None = None,
    ) -> ir.Table:
        """Register a json file as a table in the current database.

        Parameters
        ----------
        path
            The data source.
        schema
            The schema for the new table.
        table_name
            An optional name to use for the created table. This defaults to
            a sequentially generated name.

        Returns
        -------
        ir.Table
            The just-registered table

        """
        return self._read_file(
            file_type="json", path=path, schema=schema, table_name=table_name
        )

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
            statement = InsertSelect(
                table_name,
                self.compile(obj),
                database=database,
                catalog=catalog,
                overwrite=overwrite,
            )
            return self.raw_sql(statement.compile())

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

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        pyarrow_batches = iter(
            self.to_pyarrow_batches(expr, params=params, limit=limit, **kwargs)
        )

        first_batch = next(pyarrow_batches, None)

        if first_batch is None:
            pa_table = expr.as_table().schema().to_pyarrow().empty_table()
        else:
            pa_table = pa.Table.from_batches(
                itertools.chain((first_batch,), pyarrow_batches)
            )
        return expr.__pyarrow_result__(pa_table)

    def to_pyarrow_batches(
        self,
        expr: ir.Table,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        chunk_size: int | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ):
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401

        ibis_table = expr.as_table()

        if params is None and limit is None:
            # Note (mehmet): `_from_pyflink_table_to_pyarrow_batches()` does not
            # support args `params` and `limit`.
            pyflink_table = self._from_ibis_table_to_pyflink_table(ibis_table)
            if pyflink_table:
                # Note (mehmet): `_from_pyflink_table_to_pyarrow_batches()` supports
                # only expressions that are registered as tables in Flink.
                return self._from_pyflink_table_to_pyarrow_batches(
                    table=pyflink_table,
                    chunk_size=chunk_size,
                )

        # Note (mehmet): In the following, the entire result is fetched
        # into a dataframe before converting it to arrow batches.
        df = self.execute(ibis_table, limit=limit, **kwargs)
        # TODO (mehmet): `limit` is discarded in `execute()`. Is this intentional?
        df = df.head(limit)

        ibis_schema = ibis_table.schema()
        arrow_schema = ibis_schema.to_pyarrow()
        arrow_table = pa.Table.from_pandas(df, schema=arrow_schema)
        return arrow_table.to_reader()

    def _from_ibis_table_to_pyflink_table(self, table: ir.Table) -> Table | None:
        try:
            table_name = table.get_name()
        except AttributeError:
            # `table` is not a registered table in Flink.
            return None

        qualified_name = sg.table(table_name, quoted=self.compiler.quoted).sql(
            self.name
        )
        try:
            return self._table_env.from_path(qualified_name)
        except Py4JJavaError:
            # `table` is not a registered table in Flink.
            return None

    def _from_pyflink_table_to_pyarrow_batches(
        self,
        table: Table,
        *,
        chunk_size: int | None = None,
    ):
        import pyarrow as pa
        import pyarrow_hotfix  # noqa: F401
        import pytz
        from pyflink.java_gateway import get_gateway
        from pyflink.table.serializers import ArrowSerializer
        from pyflink.table.types import create_arrow_schema

        from ibis.backends.flink.datatypes import get_field_data_types
        # Note (mehmet): Implementation of this is based on
        # pyflink/table/table.py: to_pandas().

        gateway = get_gateway()
        if chunk_size:
            max_arrow_batch_size = chunk_size
        else:
            max_arrow_batch_size = (
                table._j_table.getTableEnvironment()
                .getConfig()
                .get(
                    gateway.jvm.org.apache.flink.python.PythonOptions.MAX_ARROW_BATCH_SIZE
                )
            )
        batches_iterator = gateway.jvm.org.apache.flink.table.runtime.arrow.ArrowUtils.collectAsPandasDataFrame(
            table._j_table, max_arrow_batch_size
        )

        pyflink_schema = table.get_schema()
        arrow_schema = create_arrow_schema(
            pyflink_schema.get_field_names(), get_field_data_types(pyflink_schema)
        )

        timezone = pytz.timezone(
            table._j_table.getTableEnvironment().getConfig().getLocalTimeZone().getId()
        )
        serializer = ArrowSerializer(
            arrow_schema, pyflink_schema.to_row_data_type(), timezone
        )

        return pa.RecordBatchReader.from_batches(
            arrow_schema, serializer.load_from_iterator(batches_iterator)
        )
