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
from ibis.backends.sql.compilers.base import STAR

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pandas as pd
    import pyarrow as pa

    from ibis.backends.sql.compilers.base import SQLGlotCompiler
    from ibis.expr.schema import SchemaLike


class _DatabaseSchemaHandler:
    """Temporary mixin collecting several helper functions and code snippets.

    Help to 'gracefully' deprecate the use of `schema` as a hierarchical term.
    """

    @staticmethod
    def _warn_schema():
        util.warn_deprecated(
            name="schema",
            as_of="9.0",
            removed_in="10.0",
            instead="Use the `database` kwarg with one of the following patterns:"
            '\ndatabase="database"'
            '\ndatabase=("catalog", "database")'
            '\ndatabase="catalog.database"',
            # TODO: add option for namespace object
        )

    def _warn_and_create_table_loc(self, database=None, schema=None):
        if schema is not None:
            self._warn_schema()

        if database is not None and schema is not None:
            if isinstance(database, str):
                table_loc = f"{database}.{schema}"
            elif isinstance(database, tuple):
                table_loc = database + schema
        elif schema is not None:
            table_loc = schema
        elif database is not None:
            table_loc = database
        else:
            table_loc = None

        table_loc = self._to_sqlglot_table(table_loc)

        return table_loc


class SQLBackend(BaseBackend, _DatabaseSchemaHandler):
    compiler: ClassVar[SQLGlotCompiler]
    name: ClassVar[str]

    _top_level_methods = ("from_connection",)

    @property
    def dialect(self) -> sg.Dialect:
        return self.compiler.dialect

    @classmethod
    def has_operation(cls, operation: type[ops.Value]) -> bool:
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
        self,
        name: str,
        schema: str | None = None,
        database: tuple[str, str] | str | None = None,
    ) -> ir.Table:
        """Construct a table expression.

        Parameters
        ----------
        name
            Table name
        schema
            [deprecated] Schema name
        database
            Database name

        Returns
        -------
        Table
            Table expression

        """
        table_loc = self._warn_and_create_table_loc(database, schema)

        catalog, database = None, None
        if table_loc is not None:
            catalog = table_loc.catalog or None
            database = table_loc.db or None

        table_schema = self.get_schema(name, catalog=catalog, database=database)
        return ops.DatabaseTable(
            name,
            schema=table_schema,
            source=self,
            namespace=ops.Namespace(catalog=catalog, database=database),
        ).to_expr()

    def _to_sqlglot(
        self, expr: ir.Expr, *, limit: str | None = None, params=None, **_: Any
    ):
        """Compile an Ibis expression to a sqlglot object."""
        table_expr = expr.as_table()

        if limit == "default":
            limit = ibis.options.sql.default_limit
        if limit is not None:
            table_expr = table_expr.limit(limit)

        if params is None:
            params = {}

        sql = self.compiler.translate(table_expr.op(), params=params)
        assert not isinstance(sql, sge.Subquery)

        if isinstance(sql, sge.Table):
            sql = sg.select(STAR, copy=False).from_(sql, copy=False)

        assert not isinstance(sql, sge.Subquery)
        return sql

    def compile(
        self,
        expr: ir.Expr,
        limit: str | None = None,
        params=None,
        pretty: bool = False,
        **kwargs: Any,
    ):
        """Compile an Ibis expression to a SQL string."""
        query = self._to_sqlglot(expr, limit=limit, params=params, **kwargs)
        sql = query.sql(dialect=self.dialect, pretty=pretty, copy=False)
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
        schema: SchemaLike | None = None,
        dialect: str | None = None,
    ) -> ir.Table:
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

    def _get_sql_string_view_schema(self, name, table, query) -> sch.Schema:
        compiler = self.compiler
        dialect = compiler.dialect

        cte = self._to_sqlglot(table)
        parsed = sg.parse_one(query, read=dialect)
        parsed.args["with"] = cte.args.pop("with", [])
        parsed = parsed.with_(
            sg.to_identifier(name, quoted=compiler.quoted), as_=cte, dialect=dialect
        )

        sql = parsed.sql(dialect)
        return self._get_schema_using_query(sql)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        schema: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        table_loc = self._warn_and_create_table_loc(database, schema)
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
        self,
        name: str,
        *,
        database: str | None = None,
        schema: str | None = None,
        force: bool = False,
    ) -> None:
        table_loc = self._warn_and_create_table_loc(database, schema)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        src = sge.Drop(
            this=sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted),
            kind="VIEW",
            exists=force,
        )
        with self._safe_raw_sql(src):
            pass

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, name):
        self.drop_table(name, force=True)

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute an expression."""

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
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        table_loc = self._warn_and_create_table_loc(database, None)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        drop_stmt = sg.exp.Drop(
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
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        """Execute expression and return an iterator of pyarrow record batches.

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
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        schema: str | None = None,
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
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        schema
            [deprecated] The name of the schema that the table is located in
        database
            Name of the attached database that the table is located in.

            For backends that support multi-level table hierarchies, you can
            pass in a dotted string path like `"catalog.database"` or a tuple of
            strings like `("catalog", "database")`.
        overwrite
            If `True` then replace existing contents of table

        """
        table_loc = self._warn_and_create_table_loc(database, schema)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        if overwrite:
            self.truncate_table(table_name, database=(catalog, db))

        if not isinstance(obj, ir.Table):
            obj = ibis.memtable(obj)

        self._run_pre_execute_hooks(obj)

        query = self._build_insert_from_table(
            target=table_name, source=obj, db=db, catalog=catalog
        )

        with self._safe_raw_sql(query):
            pass

    def _build_insert_from_table(
        self, *, target: str, source, db: str | None = None, catalog: str | None = None
    ):
        compiler = self.compiler
        quoted = compiler.quoted
        # Compare the columns between the target table and the object to be inserted
        # If they don't match, assume auto-generated column names and use positional
        # ordering.
        source_cols = source.columns
        columns = (
            source_cols
            if not set(target_cols := self.get_schema(target).names).difference(
                source_cols
            )
            else target_cols
        )

        query = sge.insert(
            expression=self.compile(source),
            into=sg.table(target, db=db, catalog=catalog, quoted=quoted),
            columns=[sg.to_identifier(col, quoted=quoted) for col in columns],
            dialect=compiler.dialect,
        )
        return query

    def _build_insert_template(
        self,
        name,
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
            Placeholder string. Can be a format string with a single `{i}` spec.

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
                            sge.Var(this=placeholder.format(i=i))
                            for i in range(len(schema))
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

    def truncate_table(
        self, name: str, database: str | None = None, schema: str | None = None
    ) -> None:
        """Delete all rows from a table.

        ::: {.callout-note}
        ## Ibis does not use the word `schema` to refer to database hierarchy.

        A collection of tables is referred to as a `database`.
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
        schema
            [deprecated] Schema name

        """
        table_loc = self._warn_and_create_table_loc(database, schema)
        catalog, db = self._to_catalog_db_tuple(table_loc)

        ident = sg.table(name, db=db, catalog=catalog, quoted=self.compiler.quoted).sql(
            self.dialect
        )
        with self._safe_raw_sql(f"TRUNCATE TABLE {ident}"):
            pass

    @util.experimental
    @classmethod
    def from_connection(cls, con: Any, **kwargs: Any) -> BaseBackend:
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
        # This is part of the Python DB-API specification so should work for
        # _most_ sqlglot backends
        self.con.close()

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF | ops.AggUDF) -> None:
        """Compile a built-in UDF. No-op by default."""

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"Python UDFs are not supported in the {self.name} backend"
        )

    def _compile_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError(
            f"PyArrow UDFs are not supported in the {self.name} backend"
        )

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError(
            f"pandas UDFs are not supported in the {self.name} backend"
        )

    def _to_catalog_db_tuple(self, table_loc: sge.Table):
        if table_loc is None or table_loc == (None, None):
            return None, None

        if (sg_cat := table_loc.args["catalog"]) is not None:
            sg_cat.args["quoted"] = False
            sg_cat = sg_cat.sql(self.name)
        if (sg_db := table_loc.args["db"]) is not None:
            sg_db.args["quoted"] = False
            sg_db = sg_db.sql(self.name)

        return sg_cat, sg_db

    def _to_sqlglot_table(self, database):
        if database is None:
            return None
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
            database = sg.exp.Table(
                catalog=sg.to_identifier(catalog, quoted=self.compiler.quoted),
                db=sg.to_identifier(database, quoted=self.compiler.quoted),
            )
        elif isinstance(database, str):
            # There is no definition of a sqlglot catalog.database hierarchy outside
            # of the standard table expression.
            # sqlglot parsing of the string will assume that it's a Table
            # so we unpack the arguments into a new sqlglot object, switching
            # table (this) -> database (db) and database (db) -> catalog
            table = sg.parse_one(database, into=sg.exp.Table, dialect=self.dialect)
            if table.args["catalog"] is not None:
                raise exc.IbisInputError(
                    f"Overspecified table hierarchy provided: `{table.sql(self.dialect)}`"
                )
            catalog = table.args["db"]
            db = table.args["this"]
            database = sg.exp.Table(catalog=catalog, db=db)
        else:
            raise ValueError(
                """Invalid database hierarchy format.  Please use either dotted
                strings ('catalog.database') or tuples ('catalog', 'database')."""
            )

        return database
