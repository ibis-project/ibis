from __future__ import annotations

from ibis.backends.base.sql.ddl import (
    CTAS,
    AlterTable,
    CreateTable,
    CreateTableWithSchema,
    DropObject,
    InsertSelect,
    RenameTable,
)
from ibis.backends.base.sql.registry import quote_identifier
from ibis.backends.pyspark.datatypes import type_to_sql_string

_format_aliases = {"TEXTFILE": "TEXT"}


def _sanitize_format(format):
    if format is None:
        return None
    format = format.upper()
    format = _format_aliases.get(format, format)
    if format not in (
        "TEXT",
        "CSV",
        "JSON",
        "JDBC",
        "PARQUET",
        "ORC",
        "HIVE",
        "DELTA",
        "LIBSVM",
    ):
        raise ValueError(f"Invalid format: {format!r}")

    return format


def format_tblproperties(props):
    formatted_props = _format_properties(props)
    return f"TBLPROPERTIES {formatted_props}"


def _format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append(f"  '{k}'='{v}'")

    return "(\n{}\n)".format(",\n".join(tokens))


class CreateTable(CreateTable):
    """Create a table."""

    def __init__(
        self,
        table_name,
        database=None,
        format="parquet",
        can_exist=False,
        tbl_properties=None,
    ):
        super().__init__(
            table_name,
            database=database,
            external=False,
            format=format,
            can_exist=can_exist,
            partition=None,
            tbl_properties=tbl_properties,
        )

    def _storage(self):
        return f"USING {self.format}"


class CreateTableWithSchema(CreateTableWithSchema):
    def _storage(self):
        return f"USING {self.format}"


class CTAS(CTAS):
    """Create Table As Select."""

    def __init__(
        self,
        table_name,
        select,
        database=None,
        format="parquet",
        can_exist=False,
    ):
        super().__init__(
            table_name,
            select,
            database=database,
            format=format,
            can_exist=can_exist,
        )
        self.select = select

    def _storage(self):
        return f"USING {self.format}"


class CreateView(CTAS):
    """Create a view."""

    def __init__(
        self,
        table_name,
        select,
        database=None,
        can_exist=False,
        temporary=False,
    ):
        super().__init__(table_name, select, database=database, can_exist=can_exist)
        self.temporary = temporary

    @property
    def _pieces(self):
        yield "AS"
        yield self.select.compile()

    @property
    def _prefix(self):
        return f"CREATE {self._or_replace_clause()}{self._temporary_clause()}VIEW"

    def _or_replace_clause(self):
        return "OR REPLACE " if self.can_exist else ""

    def _temporary_clause(self):
        return "TEMPORARY " if self.temporary else ""

    def _if_exists(self):
        return ""


def format_schema(schema):
    elements = [
        _format_schema_element(name, t) for name, t in zip(schema.names, schema.types)
    ]
    return "({})".format(",\n ".join(elements))


def _format_schema_element(name, t):
    return f"{quote_identifier(name, force=True)} {type_to_sql_string(t)}"


class DropDatabase(DropObject):
    _object_type = "DATABASE"

    def __init__(self, name, must_exist=True, cascade=False):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.cascade = cascade

    def _object_name(self):
        return self.name

    def compile(self):
        compiled = super().compile()
        if self.cascade:
            return f"{compiled} CASCADE"
        else:
            return compiled


class DropFunction(DropObject):
    _object_type = "TEMPORARY FUNCTION"

    def __init__(self, name, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.must_exist = must_exist

    def _object_name(self):
        return self.name


class InsertSelect(InsertSelect):
    def __init__(self, table_name, select_expr, database=None, overwrite=False):
        super().__init__(
            table_name,
            select_expr,
            database=database,
            partition=None,
            partition_schema=None,
            overwrite=overwrite,
        )

    def compile(self):
        if self.overwrite:
            cmd = "INSERT OVERWRITE TABLE"
        else:
            cmd = "INSERT INTO"

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return f"{cmd} {scoped_name}\n{select_query}"


class AlterTable(AlterTable):
    def __init__(self, table, tbl_properties=None):
        super().__init__(
            table,
            location=None,
            format=None,
            tbl_properties=tbl_properties,
            serde_properties=None,
        )

    def compile(self):
        props = self._format_properties()
        action = f"{self.table} SET{props}"
        return self._wrap_command(action)


class RenameTable(RenameTable):
    def __init__(self, old_name, new_name):
        super().__init__(old_name, new_name, dialect="spark")
