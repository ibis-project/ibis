from __future__ import annotations

import abc
import ast
import typing

import adbc_driver_manager.dbapi
import sqlalchemy as sa

import ibis.backends.pyarrow.datatypes as pa_dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.base.sql.alchemy import AlchemyTable, to_sqla_type

if typing.TYPE_CHECKING:
    import pandas as pd


__all__ = ('BaseAdbcBackend',)


class AdbcTable(ops.DatabaseTable):
    catalog = rlz.optional(rlz.instance_of(str))
    db_schema = rlz.optional(rlz.instance_of(str))


class AdbcAlchemyTable(AlchemyTable):
    catalog = rlz.optional(rlz.instance_of(str))
    db_schema = rlz.optional(rlz.instance_of(str))

    def __init__(self, source, sqla_table, name, schema, catalog, db_schema):
        ops.DatabaseTable.__init__(
            self,
            name=name,
            schema=schema,
            sqla_table=sqla_table,
            source=source,
            catalog=catalog,
            db_schema=db_schema,
        )


class BaseAdbcBackend(BaseSQLBackend):
    """Backend class for backends that use ADBC and SQL."""

    table_class = AdbcTable

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_catalog = None
        self._current_db_schema = None
        self.conn = None

    def __del__(self) -> None:
        if self.conn:
            self.conn.close()

    @property
    def current_database(self) -> str | None:
        return BaseAdbcBackend._encode_database(
            self._current_catalog, self._current_db_schema
        )

    def do_connect(self, conn: adbc_driver_manager.dbapi.Connection) -> None:
        self.conn = conn

    def fetch_from_cursor(
        self, cursor: adbc_driver_manager.dbapi.Cursor, schema: sch.Schema
    ) -> pd.DataFrame:
        with cursor:
            df = cursor.fetch_df()
            return schema.apply_to(df)

    def get_schema(
        self, catalog: str | None, db_schema: str | None, table: str
    ) -> sch.Schema:
        try:
            schema = self.conn.adbc_get_table_schema(
                table, catalog_filter=catalog, db_schema_filter=db_schema
            )
        except adbc_driver_manager.dbapi.Error as e:
            raise KeyError(
                f"Could not find table '{table}' in catalog '{catalog}', schema '{db_schema}'"
            ) from e
        return pa_dt.from_pyarrow_schema(schema)

    def list_databases(self, like: str = None) -> list[str]:
        reader = self.conn.adbc_get_objects(depth="db_schemas")
        results = []
        for batch in reader:
            catalogs = batch["catalog_name"]
            catalog_db_schemas = batch["catalog_db_schemas"]
            for catalog, db_schemas in zip(catalogs, catalog_db_schemas):
                catalog = catalog.as_py()
                if not catalog:
                    # XXX: empty string == None (quirk in ADBC)
                    catalog = None

                for db_schema in db_schemas:
                    results.append(
                        BaseAdbcBackend._encode_database(
                            catalog, db_schema["db_schema_name"].as_py()
                        )
                    )
        return self._filter_with_like(results, like)

    def list_tables(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        if database:
            raise NotImplementedError
        else:
            catalog_filter = self._current_catalog
            db_schema_filter = self._current_db_schema
        result = []
        reader = self.conn.adbc_get_objects(
            depth="tables",
            catalog_filter=catalog_filter,
            db_schema_filter=db_schema_filter,
            table_name_filter=like,
        )
        for batch in reader:
            schemas = batch["catalog_db_schemas"].values
            tables = schemas.field("db_schema_tables").values
            result.extend(tables.field("table_name").to_pylist())
        return result

    def raw_sql(self, query: str) -> adbc_driver_manager.dbapi.Cursor:
        # TODO: apache/arrow-adbc#454
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            return cur
        except:
            cur.close()
            raise
        return None

    def set_context(self, catalog: str | None, db_schema: str | None) -> None:
        self._current_catalog = catalog
        self._current_db_schema = db_schema

    def table(self, name: str, database: str | None = None) -> ir.Table:
        if database:
            raise NotImplementedError
        schema = self.get_schema(self._current_catalog, self._current_db_schema, name)
        node = self.table_class(
            name, schema, self, self._current_catalog, self._current_db_schema
        )
        return self.table_expr_class(node)

    # TODO: override to_pyarrow_batches

    @property
    def version(self) -> str:
        # TODO: query get_info and cache
        return "version"

    @staticmethod
    def _decode_database(database: str | None) -> tuple[str | None, str | None]:
        # Ibis database = combination of ADBC catalog and schema
        return ast.literal_eval(database)

    @staticmethod
    def _encode_database(catalog: str | None, schema: str | None) -> str | None:
        return repr((catalog, schema))


class BaseAdbcAlchemyBackend(BaseAdbcBackend):
    """Backend class for backends that generate queries via SQLAlchemy.

    NOTE: database metadata is still handled by ADBC. Backends based
    on this class only use SQLAlchemy to generate SQL.
    """

    table_class = AdbcAlchemyTable

    @property
    @abc.abstractmethod
    def dialect(self) -> sa.engine.interfaces.Dialect:
        ...

    def do_connect(self, conn) -> None:
        super().do_connect(conn)
        self._meta = sa.MetaData()
        self._tables = {}

    def raw_sql(self, query: str):
        # TODO: apply schema to result?
        if not isinstance(query, str):
            # TODO: abstractmethod
            query = self._to_sql_string(query)
        return super().raw_sql(query)

    def table(self, name: str, database: str | None = None) -> ir.Table:
        # TODO: unify with base class
        if database:
            raise NotImplementedError

        schema = self.get_schema(self._current_catalog, self._current_db_schema, name)

        key = (self._current_catalog, self._current_db_schema, name)
        if key in self._tables:
            sa_table = self._tables[key]
        else:
            # TODO: how do we express both the catalog and the schema to SQLAlchemy?
            # possibly: maintain a meta per catalog
            sa_table = self._table_from_schema(
                name, self._meta, schema, database=self._current_db_schema
            )
            self._tables[key] = sa_table

        node = self.table_class(
            source=self,
            sqla_table=sa_table,
            name=name,
            schema=schema,
            catalog=self._current_catalog,
            db_schema=self._current_db_schema,
        )
        return self.table_expr_class(node)

    def _to_sql_string(self, query) -> str:
        return str(query.compile(dialect=self.dialect))

    def _table_from_schema(self, name, meta, schema, *, database: str | None = None):
        # Convert Ibis schema to SQLA table
        columns = []

        for colname, dtype in zip(schema.names, schema.types):
            satype = to_sqla_type(self.dialect, dtype)
            column = sa.Column(colname, satype, nullable=dtype.nullable)
            columns.append(column)

        return sa.Table(name, meta, *columns, schema=database)
