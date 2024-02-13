from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot as sg

import ibis.common.exceptions as exc
import ibis.expr.schema as sch
from ibis.backends.base.sqlglot.datatypes import FlinkType
from ibis.backends.base.sqlglot.ddl import DDL, DML, CreateDDL, DropObject
from ibis.util import promote_list

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ibis.expr.api import Watermark


class FlinkBase:
    dialect = "hive"

    def format_dtype(self, dtype):
        sql_string = FlinkType.from_ibis(dtype)
        if dtype.is_timestamp():
            return (
                f"TIMESTAMP({dtype.scale})" if dtype.scale is not None else "TIMESTAMP"
            )
        else:
            return sql_string.sql("flink") + " NOT NULL" * (not dtype.nullable)

    def format_properties(self, props):
        tokens = []
        for k, v in sorted(props.items()):
            tokens.append(f"  '{k}'='{v}'")
        return "(\n{}\n)".format(",\n".join(tokens))

    def format_watermark_strategy(self, watermark: Watermark) -> str:
        from ibis.backends.flink.utils import translate_literal

        if watermark.allowed_delay is None:
            return watermark.time_col
        return (
            f"{watermark.time_col} - {translate_literal(watermark.allowed_delay.op())}"
        )

    def format_schema_with_watermark(
        self,
        schema: sch.Schema,
        watermark: Watermark | None = None,
        primary_keys: Sequence[str] | None = None,
    ) -> str:
        elements = [
            f"{self.quote(name)} {self.format_dtype(t)}"
            for name, t in zip(schema.names, schema.types)
        ]

        if watermark is not None:
            elements.append(
                f"WATERMARK FOR {watermark.time_col} AS {self.format_watermark_strategy(watermark)}"
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


class CreateTableWithSchema(FlinkBase, CreateDDL):
    def __init__(
        self,
        table_name: str,
        schema: sch.Schema,
        database=None,
        catalog=None,
        can_exist=False,
        external=False,
        partition=None,
        primary_key: str | Sequence[str] | None = None,
        tbl_properties=None,
        temporary: bool = False,
        watermark: Watermark | None = None,
    ):
        self.can_exist = can_exist
        self.catalog = catalog
        self.database = database
        self.partition = partition
        self.primary_keys = promote_list(primary_key)
        self.schema = schema
        self.table_name = table_name
        self.tbl_properties = tbl_properties
        self.temporary = temporary
        self.watermark = watermark

        # Check if `primary_keys` is a subset of the columns in `schema`.
        if self.primary_keys and not set(self.primary_keys) <= set(schema.names):
            raise exc.IbisError(
                "`primary_key` must be a subset of the columns in `schema`. \n"
                f"\t primary_key= {primary_key} \n"
                f"\t schema.names= {schema.names}"
            )

    @property
    def _prefix(self) -> str:
        # `TEMPORARY` is not documented in Flink's documentation
        modifier = " TEMPORARY" if self.temporary else ""
        return f"CREATE{modifier} TABLE"

    def _create_line(self) -> str:
        scoped_name = self.scoped_name(self.table_name, self.database, self.catalog)
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

            yield self.format_schema_with_watermark(
                main_schema, self.watermark, self.primary_keys
            )
            yield f"PARTITIONED BY {self.format_schema(part_schema)}"
        else:
            yield self.format_schema_with_watermark(
                self.schema, self.watermark, self.primary_keys
            )

        yield f"WITH {self.format_properties(self.tbl_properties)}"

    @property
    def pieces(self):
        yield self._create_line()
        yield from filter(None, self._pieces)

    def compile(self):
        return "\n".join(self.pieces)


class CreateView(FlinkBase, CreateDDL):
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
        scoped_name = self.scoped_name(self.name, self.database, self.catalog)
        return f"{self._prefix} {self._if_exists()}{scoped_name}"

    @property
    def pieces(self):
        yield self._create_line()
        yield f"AS {self.query_expression}"

    def compile(self):
        return "\n".join(self.pieces)


class DropTable(FlinkBase, DropObject):
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
        return self.scoped_name(self.table_name, self.database, self.catalog)

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


class RenameTable(FlinkBase, DDL):
    def __init__(self, old_name: str, new_name: str, must_exist: bool = True):
        self.old_name = old_name
        self.new_name = new_name
        self.must_exist = must_exist

    def compile(self):
        if_exists = "" if self.must_exist else "IF EXISTS"
        return f"ALTER TABLE {if_exists} {self.old_name} RENAME TO {self.new_name}"


class _DatabaseObject:
    def _object_name(self):
        name = sg.to_identifier(self.name, quoted=True).sql(dialect=self.dialect)
        if self.catalog:
            catalog = sg.to_identifier(self.catalog, quoted=True).sql(
                dialect=self.dialect
            )
            return f"{catalog}.{name}"
        else:
            return name


class CreateDatabase(FlinkBase, _DatabaseObject, CreateDDL):
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
            f"WITH {self.format_properties(self.db_properties)}"
            if self.db_properties
            else ""
        )

    def compile(self):
        create_decl = "CREATE DATABASE"
        create_line = f"{create_decl} {self._if_exists()}{self._object_name()}"

        return f"{create_line}\n{self._format_db_properties()}"


class DropDatabase(FlinkBase, _DatabaseObject, DropObject):
    _object_type = "DATABASE"

    def __init__(self, name: str, catalog: str | None = None, must_exist: bool = True):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.catalog = catalog


class InsertSelect(FlinkBase, DML):
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
        self.table_name = table_name
        self.database = database
        self.catalog = catalog
        self.select = select_expr
        self.partition = partition
        self.partition_schema = partition_schema
        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = "INSERT OVERWRITE"
        else:
            cmd = "INSERT INTO"

        if self.partition is not None:
            part = self.format_partition(self.partition, self.partition_schema)
            partition = f" {part} "
        else:
            partition = ""

        select_query = self.select
        scoped_name = self.scoped_name(self.table_name, self.database, self.catalog)
        return f"{cmd} {scoped_name}{partition}\n{select_query}"
