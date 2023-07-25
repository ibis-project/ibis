from __future__ import annotations

import re

import sqlglot as sg

import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis.backends.base.sql.compiler import DDL, DML
from ibis.backends.base.sql.registry import quote_identifier, type_to_sql_string

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")
_format_aliases = {"TEXT": "TEXTFILE"}


def _sanitize_format(format):
    if format is None:
        return None
    format = format.upper()
    format = _format_aliases.get(format, format)
    if format not in ("PARQUET", "AVRO", "TEXTFILE"):
        raise ValueError(f"Invalid format: {format!r}")

    return format


def is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


def format_schema(schema):
    elements = [
        _format_schema_element(name, t) for name, t in zip(schema.names, schema.types)
    ]
    return "({})".format(",\n ".join(elements))


def _format_schema_element(name, t):
    return f"{quote_identifier(name, force=True)} {type_to_sql_string(t)}"


def _format_partition_kv(k, v, type):
    if type == dt.string:
        value_formatted = f'"{v}"'
    else:
        value_formatted = str(v)

    return f"{k}={value_formatted}"


def format_partition(partition, partition_schema):
    tokens = []
    if isinstance(partition, dict):
        for name in partition_schema:
            if name in partition:
                tok = _format_partition_kv(
                    name, partition[name], partition_schema[name]
                )
            else:
                # dynamic partitioning
                tok = name
            tokens.append(tok)
    else:
        for name, value in zip(partition_schema, partition):
            tok = _format_partition_kv(name, value, partition_schema[name])
            tokens.append(tok)

    return "PARTITION ({})".format(", ".join(tokens))


def _format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append(f"  '{k}'='{v}'")

    return "(\n{}\n)".format(",\n".join(tokens))


def format_tblproperties(props):
    formatted_props = _format_properties(props)
    return f"TBLPROPERTIES {formatted_props}"


def _serdeproperties(props):
    formatted_props = _format_properties(props)
    return f"SERDEPROPERTIES {formatted_props}"


class _BaseQualifiedSQLStatement:
    def _get_scoped_name(self, obj_name, database):
        if is_fully_qualified(obj_name):
            return obj_name
        if _is_quoted(obj_name):
            obj_name = obj_name[1:-1]
        return sg.table(obj_name, db=database, quoted=True).sql(dialect="hive")


class BaseDDL(DDL, _BaseQualifiedSQLStatement):
    pass


class _BaseDML(DML, _BaseQualifiedSQLStatement):
    pass


class _CreateDDL(BaseDDL):
    def _if_exists(self):
        return "IF NOT EXISTS " if self.can_exist else ""


class CreateTable(_CreateDDL):
    def __init__(
        self,
        table_name,
        database=None,
        external=False,
        format="parquet",
        can_exist=False,
        partition=None,
        path=None,
        tbl_properties=None,
    ):
        self.table_name = table_name
        self.database = database
        self.partition = partition
        self.path = path
        self.external = external
        self.can_exist = can_exist
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties

    @property
    def _prefix(self):
        if self.external:
            return "CREATE EXTERNAL TABLE"
        else:
            return "CREATE TABLE"

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return f"{self._prefix} {self._if_exists()}{scoped_name}"

    def _location(self):
        return f"LOCATION '{self.path}'" if self.path else None

    def _storage(self):
        # By the time we're here, we have a valid format
        return f"STORED AS {self.format}"

    @property
    def pieces(self):
        yield self._create_line()
        yield from filter(None, self._pieces)

    def compile(self):
        return "\n".join(self.pieces)


class CTAS(CreateTable):
    """Create Table As Select."""

    def __init__(
        self,
        table_name,
        select,
        database=None,
        external=False,
        format="parquet",
        can_exist=False,
        path=None,
        partition=None,
    ):
        super().__init__(
            table_name,
            database=database,
            external=external,
            format=format,
            can_exist=can_exist,
            path=path,
            partition=partition,
        )
        self.select = select

    @property
    def _pieces(self):
        yield self._partitioned_by()
        yield self._storage()
        yield self._location()
        yield "AS"
        yield self.select.compile()

    def _partitioned_by(self):
        if self.partition is not None:
            return "PARTITIONED BY ({})".format(
                ", ".join(quote_identifier(expr.get_name()) for expr in self.partition)
            )
        return None


class CreateView(CTAS):
    """Create a view."""

    def __init__(self, table_name, select, database=None, can_exist=False):
        super().__init__(table_name, select, database=database, can_exist=can_exist)

    @property
    def _pieces(self):
        yield "AS"
        yield self.select.compile()

    @property
    def _prefix(self):
        return "CREATE VIEW"


class CreateTableWithSchema(CreateTable):
    def __init__(self, table_name, schema, table_format=None, **kwargs):
        super().__init__(table_name, **kwargs)
        self.schema = schema
        self.table_format = table_format

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

            yield format_schema(main_schema)
            yield f"PARTITIONED BY {format_schema(part_schema)}"
        else:
            yield format_schema(self.schema)

        if self.table_format is not None:
            yield "\n".join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


class CreateDatabase(_CreateDDL):
    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = quote_identifier(self.name)

        create_decl = "CREATE DATABASE"
        create_line = f"{create_decl} {self._if_exists()}{name}"
        if self.path is not None:
            create_line += f"\nLOCATION '{self.path}'"

        return create_line


class DropObject(BaseDDL):
    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = "" if self.must_exist else "IF EXISTS "
        object_name = self._object_name()
        return f"DROP {self._object_type} {if_exists}{object_name}"


class DropDatabase(DropObject):
    _object_type = "DATABASE"

    def __init__(self, name, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.name = name

    def _object_name(self):
        return self.name


class DropTable(DropObject):
    _object_type = "TABLE"

    def __init__(self, table_name, database=None, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class DropView(DropTable):
    _object_type = "VIEW"


class TruncateTable(BaseDDL):
    _object_type = "TABLE"

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return f"TRUNCATE TABLE {name}"


class InsertSelect(_BaseDML):
    def __init__(
        self,
        table_name,
        select_expr,
        database=None,
        partition=None,
        partition_schema=None,
        overwrite=False,
    ):
        self.table_name = table_name
        self.database = database
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
            part = format_partition(self.partition, self.partition_schema)
            partition = f" {part} "
        else:
            partition = ""

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return f"{cmd} {scoped_name}{partition}\n{select_query}"


class AlterTable(BaseDDL):
    def __init__(
        self,
        table,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        self.table = table
        self.location = location
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties
        self.serde_properties = serde_properties

    def _wrap_command(self, cmd):
        return f"ALTER TABLE {cmd}"

    def _format_properties(self, prefix=""):
        tokens = []

        if self.location is not None:
            tokens.append(f"LOCATION '{self.location}'")

        if self.format is not None:
            tokens.append(f"FILEFORMAT {self.format}")

        if self.tbl_properties is not None:
            tokens.append(format_tblproperties(self.tbl_properties))

        if self.serde_properties is not None:
            tokens.append(_serdeproperties(self.serde_properties))

        if len(tokens) > 0:
            return "\n{}{}".format(prefix, "\n".join(tokens))
        else:
            return ""

    def compile(self):
        props = self._format_properties()
        action = f"{self.table} SET {props}"
        return self._wrap_command(action)


class DropFunction(DropObject):
    def __init__(self, name, inputs, must_exist=True, aggregate=False, database=None):
        super().__init__(must_exist=must_exist)
        self.name = name
        self.inputs = tuple(map(dt.dtype, inputs))
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database

    def _object_name(self):
        return self.name

    def compile(self):
        tokens = ["DROP"]
        if self.aggregate:
            tokens.append("AGGREGATE")
        tokens.append("FUNCTION")
        if not self.must_exist:
            tokens.append("IF EXISTS")

        tokens.append(self._impala_signature())
        return " ".join(tokens)


class RenameTable(AlterTable):
    def __init__(
        self,
        old_name: str,
        new_name: str,
        old_database: str | None = None,
        new_database: str | None = None,
        dialect: str = "hive",
    ):
        self._old = sg.table(old_name, db=old_database, quoted=True).sql(
            dialect=dialect
        )
        self._new = sg.table(new_name, db=new_database, quoted=True).sql(
            dialect=dialect
        )

    def compile(self):
        return self._wrap_command(f"{self._old} RENAME TO {self._new}")


__all__ = (
    "fully_qualified_re",
    "is_fully_qualified",
    "format_schema",
    "format_partition",
    "format_tblproperties",
    "BaseDDL",
    "CreateTable",
    "CTAS",
    "CreateView",
    "CreateTableWithSchema",
    "CreateDatabase",
    "DropObject",
    "DropDatabase",
    "DropTable",
    "DropView",
    "TruncateTable",
    "InsertSelect",
    "AlterTable",
    "DropFunction",
    "RenameTable",
)
