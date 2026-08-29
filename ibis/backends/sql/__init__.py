from __future__ import annotations

import abc
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import sqlglot as sg
import sqlglot.expressions as sge

import ibis
import ibis.common.exceptions as exc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis import util
from ibis.backends import BaseBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import pandas as pd
    import pyarrow as pa

    from ibis.backends.sql.compilers.base import SQLGlotCompiler
    from ibis.expr.api import IntoMemtable
    from ibis.expr.schema import IntoSchema


class SQLBackend(BaseBackend):
    compiler: ClassVar[SQLGlotCompiler]
    name: ClassVar[str]

    _top_level_methods = ("from_connection",)

    @property
    def dialect(self) -> sg.Dialect:
        """Return the SQL dialect used by the backend."""
        return self.compiler.dialect

    @classmethod
    def has_operation(cls, operation: type[ops.Value], /) -> bool:
        """Return whether the backend supports the given operation.

        Parameters
        ----------
        operation
            Operation type, a Python class object.
        """
        compiler = cls.compiler
        if operation in compiler.extra_supported_ops:
            return True
        method = getattr(compiler, f"visit_{operation.__name__}", None)
        return method not in (
            None,
            compiler.visit_Undefined,
            compiler.visit_Unsupported,
        )

    def _fetch_from_cursor(self, cursor, schema: sch.Schema) -> pd.DataFrame:
        import pandas as pd

        from ibis.formats.pandas import PandasData

        try:
            df = pd.DataFrame.from_records(
                cursor, columns=schema.names, coerce_float=True
            )
        except Exception:
            # clean up the cursor if we fail to create the DataFrame
            #
            # in the sqlite case failing to close the cursor results in
            # artificially locked tables
            cursor.close()
            raise
        df = PandasData.convert_table(df, schema)
        return df

    def table(
        self, name: str, /, *, database: tuple[str, str] | str | None = None
    ) -> ir.Table:
        table_loc = self._to_sqlglot_table(database)

        catalog = table_loc.catalog or None
        database = table_loc.db or None

        table_schema = self.get_schema(name, catalog=catalog, database=database)
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=database),
        ).to_expr()

    def compile(
        self,
        expr: ir.Expr,
        /,
        *,
        limit: int | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        pretty: bool = False,
    ) -> str:
        """Compile an expression to a SQL string.

        Parameters
        ----------
        expr
            An ibis expression to compile.
        limit
            An integer to effect a specific row limit. A value of `None` means no limit.
        params
            Mapping of scalar parameter expressions to value.
        pretty
            Pretty print the SQL query during compilation.

        Returns
        -------
        str
            Compiled expression
        """
        query = self.compiler.to_sqlglot(expr, limit=limit, params=params)
        try:
            sql = query.sql(
                dialect=self.dialect,
                pretty=pretty,
                copy=False,
                unsupported_level=sg.ErrorLevel.RAISE,
            )
        except sg.UnsupportedError as e:
            raise exc.UnsupportedOperationError(
                f"Operation not supported in {self.name} backend: {e}\n\nexpression:\n{expr}\n\nsqlglot expression:\n{query}"
            ) from e
        self._log(sql)
        return sql

    def _log(self, sql: str) -> None:
        """Log `sql`.

        This method can be implemented by subclasses. Logging occurs when
        `ibis.options.verbose` is `True`.
        """
        from ibis import util

        util.log(sql)

    def sql(
        self,
        query: str,
        /,
        *,
        schema: IntoSchema | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
        """Create an Ibis table expression from a SQL query.

        Parameters
        ----------
        query
            A SQL query string
        schema
            The schema of the query. If not provided, Ibis will try to infer
            the schema of the query.
        dialect
            The SQL dialect of the query. If not provided, the backend's dialect
            is assumed. This argument can be useful when the query is written
            in a different dialect from the backend.

        Returns
        -------
        ir.Table
            The table expression representing the query
        """
        query = self._transpile_sql(query, dialect=dialect)
        if schema is None:
            schema = self._get_schema_using_query(query)
        return ops.SQLQueryResult(query, ibis.schema(schema), self).to_expr()

    @abc.abstractmethod
    def _get_schema_using_query(self, query: str) -> sch.Schema:
        """Return an ibis Schema from a backend-specific SQL string.

        Parameters
        ----------
        query
            Backend-specific SQL string

        Returns
        -------
        Schema
            The schema inferred from `query`
        """

    def _get_sql_string_view_schema(
        self, *, name: str, table: ir.Table, query: str
    ) -> sch.Schema:
        sql = self.compiler.add_query_to_expr(name=name, table=table, query=query)
        return self._get_schema_using_query(sql)

    def _register_udfs(self, expr: ir.Expr) -> None:
        udf_sources = []
        compiler = self.compiler
        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                compiler, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            if sql := compile_func(udf_node):
                udf_sources.append(sql)
        if udf_sources:
            # define every udf in one execution to avoid the overhead of db
            # round trips per udf
            with self._safe_raw_sql(";\n".join(udf_sources)):
                pass

    def create_view(
        self,
        name: str,
        /,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        """Create a view from an Ibis expression.

        Parameters
        ----------
        name
            The name of the view to create.
        obj
            The Ibis expression to create the view from.
        database
            The database that the view should be created in.
        overwrite
            If `True`, replace an existing view with the same name.

        Returns
        -------
        ir.Table
            A table expression representing the view.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        src = sge.Create(
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            kind="VIEW",
            replace=overwrite,
            expression=self.compile(obj),
        )
        self._register_in_memory_tables(obj)
        with self._safe_raw_sql(src):
            pass
        return self.table(name, database=(catalog, db))

    def drop_view(
        self, name: str, /, *, database: str | None = None, force: bool = False
    ) -> None:
        """Drop a view from the backend.

        Parameters
        ----------
        name
            The name of the view to drop.
        database
            The database that the view is located in.
        force
            If `True`, do not raise an error if the view does not exist.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        src = sge.Drop(
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            kind="VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

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

        Returns
        -------
        DataFrame | Series | scalar
            The result of the expression execution.
        """
        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, params=params, limit=limit, **kwargs)

        schema = table.schema()

        # TODO(kszucs): these methods should be abstractmethods or this default
        # implementation should be removed
        with self._safe_raw_sql(sql) as cur:
            result = self._fetch_from_cursor(cur, schema)
        return expr.__pandas_result__(result)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        """Drop a table from the backend.

        Parameters
        ----------
        name
            The name of the table to drop
        database
            The database that the table is located in.
        force
            If `True`, do not raise an error if the table does not exist.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        drop_stmt = sge.Drop(
            kind="TABLE",
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            exists=force,
        )
        with self._safe_raw_sql(drop_stmt):
            pass

    def _cursor_batches(
        self,
        expr: ir.Expr,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1 << 20,
    ) -> Iterable[list]:
        self._run_pre_execute_hooks(expr)

        with self._safe_raw_sql(
            self.compile(expr, limit=limit, params=params)
        ) as cursor:
            while batch := cursor.fetchmany(chunk_size):
                yield batch

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
        """Execute expression and return an iterator of PyArrow record batches.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        expr
            Ibis expression to export to pyarrow
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value.
        chunk_size
            Maximum number of rows in each returned record batch.

        Returns
        -------
        RecordBatchReader
            Collection of pyarrow `RecordBatch`s.
        """
        pa = self._import_pyarrow()

        schema = expr.as_table().schema()
        array_type = schema.as_struct().to_pyarrow()
        arrays = (
            pa.array(map(tuple, batch), type=array_type)
            for batch in self._cursor_batches(
                expr, params=params, limit=limit, chunk_size=chunk_size
            )
        )
        batches = map(pa.RecordBatch.from_struct_array, arrays)

        return pa.ipc.RecordBatchReader.from_batches(schema.to_pyarrow(), batches)

    def insert(
        self,
        name: str,
        /,
        obj: ir.Table | IntoMemtable,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Insert data into a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            The name of the table to which data will be inserted
        obj
            The source data or expression to insert
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        overwrite
            If `True` then replace existing contents of table
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        if overwrite:
            self.truncate_table(name, database=(catalog, db))

        source_table = self._ensure_table_to_insert(
            target_columns=self.get_schema(name, catalog=catalog, database=db),
            data=obj,
        )

        self._run_pre_execute_hooks(source_table)

        query = self._build_insert_from_table(
            table_name=name, data=source_table, db=db, catalog=catalog
        )

        with self._safe_raw_sql(query):
            pass

    def _ensure_table_to_insert(
        self,
        *,
        target_columns: Iterable[str],
        data: ir.Table | IntoMemtable,
    ) -> ir.Table:
        target_col_names = tuple(target_columns)
        if isinstance(data, ir.Table):
            source_table = data
        else:
            from ibis.expr.api import _memtable

            is_named, source_table = _memtable(data)
            if not is_named:
                if len(source_table.schema()) != len(target_col_names):
                    raise exc.IbisTypeError(
                        f"Cannot insert into table with columns {target_col_names} "
                        f"from data with {len(source_table.schema())} unnamed columns. "
                        "Please provide data with named columns."
                    )
                source_table = source_table.rename(
                    dict(zip(target_col_names, source_table.schema()))
                )
        source_col_names = tuple(source_table.schema())
        # Error on unknown columns.
        # We DO allow missing columns (they will be filled with NULLs or defaults)
        unknown_cols = set(source_col_names) - set(target_col_names)
        if unknown_cols:
            raise exc.IbisTypeError(
                f"Cannot insert into table {target_col_names} because the following "
                f"columns are not present in the target table: "
                f"{', '.join(sorted(unknown_cols))}"
            )
        return source_table

    def _build_insert_from_table(
        self,
        *,
        data: ir.Table,
        table_name: str,
        db: str | None = None,
        catalog: str | None = None,
    ):
        compiler = self.compiler
        quoted = compiler.quoted
        query = sge.insert(
            expression=self.compile(data),
            into=sg.table(table_name, db=db, catalog=catalog, quoted=quoted),
            columns=[sg.to_identifier(col, quoted=quoted) for col in data.schema()],
            dialect=compiler.dialect,
        )
        return query

    def _build_insert_template(
        self,
        name: str,
        *,
        schema: sch.Schema,
        catalog: str | None = None,
        columns: bool = False,
        placeholder: str = "?",
    ) -> str:
        """Builds an INSERT INTO table VALUES query string with placeholders.

        Parameters
        ----------
        name
            Name of the table to insert into
        schema
            Ibis schema of the table to insert into
        catalog
            Catalog name of the table to insert into
        columns
            Whether to render the columns to insert into
        placeholder
            Placeholder string.

        Returns
        -------
        str
            The query string
        """
        quoted = self.compiler.quoted
        return sge.insert(
            sge.Values(
                expressions=[
                    sge.Tuple(
                        expressions=[
                            sge.Var(this=placeholder.format(i=i, name=name))
                            for i, name in enumerate(schema.keys())
                        ]
                    )
                ]
            ),
            into=sg.table(name, catalog=catalog, quoted=quoted),
            columns=(
                map(partial(sg.to_identifier, quoted=quoted), schema.keys())
                if columns
                else None
            ),
        ).sql(self.dialect)

    def upsert(
        self,
        name: str,
        /,
        obj: ir.Table | IntoMemtable,
        on: str,
        *,
        database: str | None = None,
    ) -> None:
        """Upsert data into a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            The name of the table to which data will be upserted
        obj
            The source data or expression to upsert
        on
            Column name to join on
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        source_table = self._ensure_table_to_insert(
            target_columns=self.get_schema(name, catalog=catalog, database=db),
            data=obj,
        )

        self._run_pre_execute_hooks(source_table)

        query = self._build_upsert_from_table(
            target=name, source=source_table, on=on, db=db, catalog=catalog
        )

        with self._safe_raw_sql(query):
            pass

    def _build_upsert_from_table(
        self,
        *,
        target: str,
        source: ir.Table,
        on: str,
        db: str | None = None,
        catalog: str | None = None,
    ):
        compiler = self.compiler
        quoted = compiler.quoted

        source_alias = util.gen_name("source")
        target_alias = util.gen_name("target")
        query = sge.merge(
            sge.When(
                matched=True,
                then=sge.Update(
                    expressions=[
                        sg.column(col, quoted=quoted).eq(
                            sg.column(col, table=source_alias, quoted=quoted)
                        )
                        for col in source.schema()
                        if col != on
                    ]
                ),
            ),
            sge.When(
                matched=False,
                then=sge.Insert(
                    this=sge.Tuple(
                        expressions=[
                            sg.column(col, quoted=quoted) for col in source.schema()
                        ]
                    ),
                    expression=sge.Tuple(
                        expressions=[
                            sg.column(col, table=source_alias, quoted=quoted)
                            for col in source.schema()
                        ]
                    ),
                ),
            ),
            into=sg.table(target, db=db, catalog=catalog, quoted=quoted).as_(
                sg.to_identifier(target_alias, quoted=quoted), table=True
            ),
            using=f"({self.compile(source)}) AS {sg.to_identifier(source_alias, quoted=quoted)}",
            on=sge.Paren(
                this=sg.column(on, table=target_alias, quoted=quoted).eq(
                    sg.column(on, table=source_alias, quoted=quoted)
                )
            ),
            dialect=compiler.dialect,
        )
        return query

    def truncate_table(
        self, name: str, /, *, database: str | tuple[str, str] | None = None
    ) -> None:
        """Delete all rows from a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of `table` is referred to as a `database`.
        A collection of `database` is referred to as a `catalog`.

        These terms are mapped onto the corresponding features in each
        backend (where available), regardless of whether the backend itself
        uses the same terminology.
        :::

        Parameters
        ----------
        name
            Table name
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        """
        table_loc = self._to_sqlglot_table(database)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        ident = sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted).sql(
            self.dialect
        )
        with self._safe_raw_sql(f"TRUNCATE TABLE {ident}"):
            pass

    @util.experimental
    @classmethod
    def from_connection(cls, con: Any, /, **kwargs: Any) -> BaseBackend:
        """Create an Ibis client from an existing connection.

        Parameters
        ----------
        con
            An existing connection.
        **kwargs
            Extra arguments to be applied to the newly-created backend.
        """
        raise NotImplementedError(
            f"{cls.name} backend cannot be constructed from an existing connection"
        )

    def disconnect(self):
        """Disconnect from the backend."""
        # This is part of the Python DB-API specification so should work for
        # _most_ sqlglot backends
        self.con.close()

    def _to_catalog_db_tuple(self, table_loc: sge.Table):
        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
            sg_cat = sg_cat.sql(self.dialect)
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
            sg_db = sg_db.sql(self.dialect)

        return sg_cat, sg_db

    def _to_sqlglot_table(self, database: None | str | tuple[str, str]) -> sge.Table:
        quoted = self.compiler.quoted
        dialect = self.dialect

        if database is None:
            # Create "table" with empty catalog and db
            sgt = sge.Table(catalog=None, db=None)
        elif isinstance(database, (list, tuple)):
            if len(database) > 2:
                raise ValueError(
                    "Only database hierarchies of two or fewer levels are supported."
                    "\nYou can specify ('catalog', 'database')."
                )
            elif len(database) == 2:
                catalog, database = database
            elif len(database) == 1:
                database = database[0]
                catalog = None
            else:
                raise ValueError(
                    f"Malformed database tuple {database} provided"
                    "\nPlease specify one of:"
                    '\n("catalog", "database")'
                    '\n("database",)'
                )
            sgt = sge.Table(
                catalog=sg.to_identifier(catalog, quoted=quoted),
                db=sg.to_identifier(database, quoted=quoted),
            )
        elif isinstance(database, str):
            # There is no definition of a sqlglot catalog.database hierarchy outside
            # of the standard table expression.
            # sqlglot parsing of the string will assume that it's a Table
            # so we unpack the arguments into a new sqlglot object, switching
            # table (this) -> database (db) and database (db) -> catalog
            sgt = sg.parse_one(
                ".".join(
                    sg.to_identifier(part, quoted=quoted).sql(dialect)
                    for part in database.split(".")
                ),
                into=sge.Table,
                dialect=dialect,
            )
            if sgt.args["catalog"] is not None:
                raise exc.IbisInputError(
                    f"Overspecified table hierarchy provided: `{sgt.sql(dialect)}`"
                )
            catalog = sgt.args["db"]
            db = sgt.args["this"]
            sgt = sge.Table(catalog=catalog, db=db)
        else:
            raise ValueError(
                """Invalid database hierarchy format.  Please use either dotted
                strings ('catalog.database') or tuples ('catalog', 'database')."""
            )

        return sgt

    def _register_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:
        """No-op."""

    def _register_python_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"Python UDFs are not supported in the {self.dialect} backend"
        )

    def _register_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"PyArrow UDFs are not supported in the {self.dialect} backend"
        )

    def _register_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"pandas UDFs are not supported in the {self.dialect} backend"
        )

    def _make_memtable_finalizer(self, name: str) -> Callable[..., None]:
        this = sg.table(name, quoted=self.compiler.quoted)
        drop_stmt = sge.Drop(kind="TABLE", this=this, exists=True)
        drop_sql = drop_stmt.sql(self.dialect)

        def finalizer(drop_sql=drop_sql, con=self.con) -> None:
            with con.cursor() as cursor:
                cursor.execute(drop_sql)

        return finalizer
