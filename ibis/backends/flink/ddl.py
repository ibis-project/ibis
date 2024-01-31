from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot as sg

import ibis.common.exceptions as exc
import ibis.expr.schema as sch
from ibis.backends.base.sql.ddl import (
    CreateTable,
    CreateTableWithSchema,
    DropObject,
    InsertSelect,
    RenameTable,
    _CreateDDL,
    _format_properties,
    _is_quoted,
    format_partition,
    is_fully_qualified,
)
from ibis.backends.base.sql.registry import quote_identifier
from ibis.backends.flink.registry import type_to_sql_string
from ibis.util import promote_list

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ibis.api import Watermark


def format_schema(schema: sch.Schema):
    elements = [
        _format_schema_element(name, t) for name, t in zip(schema.names, schema.types)
    ]

    return "({})".format(",\n ".join(elements))


def _format_schema_element(name, t):
    return f"{quote_identifier(name, force=True)} {type_to_flink_sql_string(t)}"


def type_to_flink_sql_string(tval):
    if tval.is_timestamp():
        return f"timestamp({tval.scale})" if tval.scale is not None else "timestamp"
    else:
        sql_string = type_to_sql_string(tval)
        if not tval.nullable:
            sql_string += " NOT NULL"
        return sql_string


def _format_watermark_strategy(watermark: Watermark) -> str:
    from ibis.backends.flink.utils import translate_literal

    if watermark.allowed_delay is None:
        return watermark.time_col
    return f"{watermark.time_col} - {translate_literal(watermark.allowed_delay.op())}"


def format_schema_with_watermark(
    schema: sch.Schema,
    watermark: Watermark | None = None,
    primary_keys: Sequence[str] | None = None,
) -> str:
    elements = [
        _format_schema_element(name, t) for name, t in zip(schema.names, schema.types)
    ]

    if watermark is not None:
        elements.append(
            f"WATERMARK FOR {watermark.time_col} AS {_format_watermark_strategy(watermark)}"
        )

    if primary_keys is not None and primary_keys:
        # Note (mehmet): Currently supports "NOT ENFORCED" only. For the reason
        # of this choice, the following quote from Flink docs is self-explanatory:
        # "SQL standard specifies that a constraint can either be ENFORCED or
        # NOT ENFORCED. This controls if the constraint checks are performed on
        # the incoming/outgoing data. Flink does not own the data therefore the
        # only mode we want to support is the NOT ENFORCED mode. It is up to the
        # user to ensure that the query enforces key integrity."
        # Ref: https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/dev/table/sql/create/#primary-key
        comma_separated_keys = ", ".join(f"`{key}`" for key in primary_keys)
        elements.append(f"PRIMARY KEY ({comma_separated_keys}) NOT ENFORCED")

    return "({})".format(",\n ".join(elements))


class _CatalogAwareBaseQualifiedSQLStatement:
    def _get_scoped_name(
        self, obj_name: str, database: str | None = None, catalog: str | None = None
    ) -> str:
        if is_fully_qualified(obj_name):
            return obj_name
        if _is_quoted(obj_name):
            obj_name = obj_name[1:-1]
        return sg.table(obj_name, db=database, catalog=catalog, quoted=True).sql(
            dialect="hive"
        )


class CreateTableFromConnector(
    _CatalogAwareBaseQualifiedSQLStatement, CreateTableWithSchema
):
    def __init__(
        self,
        table_name: str,
        schema: sch.Schema,
        tbl_properties: dict,
        watermark: Watermark | None = None,
        primary_key: str | Sequence[str] | None = None,
        database: str | None = None,
        catalog: str | None = None,
        temporary: bool = False,
        **kwargs,
    ):
        super().__init__(
            table_name=table_name,
            database=database,
            schema=schema,
            table_format=None,
            format=None,
            path=None,
            tbl_properties=tbl_properties,
            **kwargs,
        )
        self.catalog = catalog
        self.temporary = temporary
        self.watermark = watermark

        self.primary_keys = promote_list(primary_key)

        # Check if `primary_keys` is a subset of the columns in `schema`.
        if self.primary_keys and not set(self.primary_keys) <= set(schema.names):
            raise exc.IbisError(
                "`primary_key` must be a subset of the columns in `schema`. \n"
                f"\t primary_key= {primary_key} \n"
                f"\t schema.names= {schema.names}"
            )

    def _storage(self) -> str:
        return f"STORED AS {self.format}" if self.format else None

    def _format_tbl_properties(self) -> str:
        return f"WITH {_format_properties(self.tbl_properties)}"

    @property
    def _prefix(self) -> str:
        # `TEMPORARY` is not documented in Flink's documentation
        modifier = " TEMPORARY" if self.temporary else ""
        return f"CREATE{modifier} TABLE"

    def _create_line(self) -> str:
        scoped_name = self._get_scoped_name(
            self.table_name, self.database, self.catalog
        )
        return f"{self._prefix} {self._if_exists()}{scoped_name}"

    @property
    def _pieces(self):
        if self.partition is not None:
            main_schema = self.schema
            part_schema = self.partition
            if not isinstance(part_schema, sch.Schema):
                part_fields = {name: self.schema[name] for name in part_schema}
                part_schema = sch.Schema(part_fields)

            to_delete = {name for name in self.partition if name in self.schema}
            fields = {
                name: dtype
                for name, dtype in main_schema.items()
                if name not in to_delete
            }
            main_schema = sch.Schema(fields)

            yield format_schema_with_watermark(
                main_schema, self.watermark, self.primary_keys
            )
            yield f"PARTITIONED BY {format_schema(part_schema)}"
        else:
            yield format_schema_with_watermark(
                self.schema, self.watermark, self.primary_keys
            )

        yield self._format_tbl_properties()


class CreateView(_CatalogAwareBaseQualifiedSQLStatement, CreateTable):
    def __init__(
        self,
        name: str,
        query_expression: str,
        database: str | None = None,
        catalog: str | None = None,
        can_exist: bool = False,
        temporary: bool = False,
    ):
        super().__init__(
            table_name=name,
            database=database,
            can_exist=can_exist,
        )
        self.name = name
        self.query_expression = query_expression
        self.catalog = catalog
        self.temporary = temporary

    @property
    def _prefix(self):
        if self.temporary:
            return "CREATE TEMPORARY VIEW"
        else:
            return "CREATE VIEW"

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.name, self.database, self.catalog)
        return f"{self._prefix} {self._if_exists()}{scoped_name}"

    @property
    def pieces(self):
        yield self._create_line()
        yield f"AS {self.query_expression}"


class DropTable(_CatalogAwareBaseQualifiedSQLStatement, DropObject):
    _object_type = "TABLE"

    def __init__(
        self,
        table_name: str,
        database: str | None = None,
        catalog: str | None = None,
        must_exist: bool = True,
        temporary: bool = False,
    ):
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database
        self.catalog = catalog
        self.temporary = temporary

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database, self.catalog)

    def compile(self):
        temporary = "TEMPORARY " if self.temporary else ""
        if_exists = "" if self.must_exist else "IF EXISTS "
        object_name = self._object_name()
        return f"DROP {temporary}{self._object_type} {if_exists}{object_name}"


class DropView(DropTable):
    _object_type = "VIEW"

    def __init__(
        self,
        name: str,
        database: str | None = None,
        catalog: str | None = None,
        must_exist: bool = True,
        temporary: bool = False,
    ):
        super().__init__(
            table_name=name,
            database=database,
            catalog=catalog,
            must_exist=must_exist,
            temporary=temporary,
        )


class RenameTable(RenameTable):
    def __init__(
        self,
        old_name: str,
        new_name: str,
        old_database: str | None = None,
        new_database: str | None = None,
        must_exist: bool = True,
    ):
        super().__init__(
            old_name=old_name,
            new_name=new_name,
            old_database=old_database,
            new_database=new_database,
        )
        self.must_exist = must_exist

    def compile(self):
        if_exists = "" if self.must_exist else "IF EXISTS"
        return f"ALTER TABLE {if_exists} {self._old} RENAME TO {self._new}"


class _DatabaseObject:
    def _object_name(self):
        scoped_name = f"{quote_identifier(self.catalog)}." if self.catalog else ""
        scoped_name += quote_identifier(self.name)
        return scoped_name


class CreateDatabase(_DatabaseObject, _CreateDDL):
    def __init__(
        self,
        name: str,
        db_properties: dict | None,
        catalog: str | None = None,
        can_exist: bool = False,
    ):
        # TODO(chloeh13q): support COMMENT
        self.name = name
        self.db_properties = db_properties
        self.catalog = catalog
        self.can_exist = can_exist

    def _format_db_properties(self) -> str:
        return (
            f"WITH {_format_properties(self.db_properties)}"
            if self.db_properties
            else ""
        )

    def compile(self):
        create_decl = "CREATE DATABASE"
        create_line = f"{create_decl} {self._if_exists()}{self._object_name()}"

        return f"{create_line}\n{self._format_db_properties()}"


class DropDatabase(_DatabaseObject, DropObject):
    _object_type = "DATABASE"

    def __init__(self, name: str, catalog: str | None = None, must_exist: bool = True):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.catalog = catalog


class InsertSelect(_CatalogAwareBaseQualifiedSQLStatement, InsertSelect):
    def __init__(
        self,
        table_name,
        select_expr,
        database: str | None = None,
        catalog: str | None = None,
        partition=None,
        partition_schema=None,
        overwrite=False,
    ):
        super().__init__(
            table_name, select_expr, database, partition, partition_schema, overwrite
        )
        self.catalog = catalog

    def compile(self):
        if self.overwrite:
            cmd = "INSERT OVERWRITE"
        else:
            cmd = "INSERT INTO"

        if self.partition is not None:
            part = format_partition(self.partition, self.partition_schema)
            partition = f" {part} "
        else:
            partition = ""

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(
            self.table_name, self.database, self.catalog
        )
        return f"{cmd} {scoped_name}{partition}\n{select_query}"
