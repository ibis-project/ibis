from __future__ import annotations

import json

import sqlglot as sg

import ibis.expr.schema as sch
from ibis.backends.sql.datatypes import ImpalaType
from ibis.backends.sql.ddl import DDL, DML, CreateDDL, DropFunction, DropObject


class ImpalaBase:
    dialect = "hive"

    def sanitize_format(self, format):
        _format_aliases = {"TEXT": "TEXTFILE"}

        if format is None:
            return None
        format = format.upper()
        format = _format_aliases.get(format, format)
        if format not in ("PARQUET", "AVRO", "TEXTFILE"):
            raise ValueError(f"Invalid format: {format!r}")

        return format

    def format_dtype(self, dtype):
        return ImpalaType.to_string(dtype)

    def format_properties(self, props):
        tokens = []
        for k, v in sorted(props.items()):
            tokens.append(f"  '{k}'='{v}'")
        return "(\n{}\n)".format(",\n".join(tokens))

    def format_tblproperties(self, props):
        formatted_props = self.format_properties(props)
        return f"TBLPROPERTIES {formatted_props}"

    def format_serdeproperties(self, props):
        formatted_props = self.format_properties(props)
        return f"SERDEPROPERTIES {formatted_props}"


class CreateDatabase(ImpalaBase, CreateDDL):
    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = self.quote(self.name)

        create_decl = "CREATE DATABASE"
        create_line = f"{create_decl} {self._if_exists()}{name}"
        if self.path is not None:
            create_line += f"\nLOCATION '{self.path}'"

        return create_line


class DropDatabase(ImpalaBase, DropObject):
    _object_type = "DATABASE"

    def __init__(self, name, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.name = name

    def _object_name(self):
        return self.name


class CreateTable(ImpalaBase, CreateDDL):
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
        self.format = self.sanitize_format(format)
        self.tbl_properties = tbl_properties

    @property
    def _prefix(self):
        if self.external:
            return "CREATE EXTERNAL TABLE"
        else:
            return "CREATE TABLE"

    def _create_line(self):
        scoped_name = self.scoped_name(self.table_name, self.database)
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

            yield self.format_schema(main_schema)
            yield f"PARTITIONED BY {self.format_schema(part_schema)}"
        else:
            yield self.format_schema(self.schema)

        if self.table_format is not None:
            yield "\n".join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


class AlterTable(ImpalaBase, DDL):
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
        self.format = self.sanitize_format(format)
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
            tokens.append(self.format_tblproperties(self.tbl_properties))

        if self.serde_properties is not None:
            tokens.append(self.format_serdeproperties(self.serde_properties))

        if len(tokens) > 0:
            return "\n{}{}".format(prefix, "\n".join(tokens))
        else:
            return ""

    def compile(self):
        props = self._format_properties()
        action = f"{self.table} SET {props}"
        return self._wrap_command(action)


class RenameTable(AlterTable):
    def __init__(
        self,
        old_name: str,
        new_name: str,
        old_database: str | None = None,
        new_database: str | None = None,
    ):
        self._old = sg.table(old_name, db=old_database, quoted=True).sql(
            dialect=self.dialect
        )
        self._new = sg.table(new_name, db=new_database, quoted=True).sql(
            dialect=self.dialect
        )

    def compile(self):
        return self._wrap_command(f"{self._old} RENAME TO {self._new}")


class DropTable(ImpalaBase, DropObject):
    _object_type = "TABLE"

    def __init__(self, table_name, database=None, must_exist=True):
        super().__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self.scoped_name(self.table_name, self.database)


class TruncateTable(ImpalaBase, DDL):
    _object_type = "TABLE"

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self.scoped_name(self.table_name, self.database)
        return f"TRUNCATE TABLE {name}"


class DropView(DropTable):
    _object_type = "VIEW"


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
        yield self.select

    def _partitioned_by(self):
        if self.partition is not None:
            return "PARTITIONED BY ({})".format(
                ", ".join(self.quote(expr.get_name()) for expr in self.partition)
            )
        return None


class CreateView(CTAS):
    """Create a view."""

    def __init__(self, table_name, select, database=None, can_exist=False):
        super().__init__(table_name, select, database=database, can_exist=can_exist)

    @property
    def _pieces(self):
        yield "AS"
        yield self.select

    @property
    def _prefix(self):
        return "CREATE VIEW"


class CreateTableParquet(CreateTable):
    def __init__(
        self,
        table_name,
        path,
        example_file=None,
        example_table=None,
        schema=None,
        external=True,
        **kwargs,
    ):
        super().__init__(
            table_name,
            external=external,
            format="parquet",
            path=path,
            **kwargs,
        )
        self.example_file = example_file
        self.example_table = example_table
        self.schema = schema

    @property
    def _pieces(self):
        if self.example_file is not None:
            yield f"LIKE PARQUET '{self.example_file}'"
        elif self.example_table is not None:
            yield f"LIKE {self.example_table}"
        elif self.schema is not None:
            yield self.format_schema(self.schema)
        else:
            raise NotImplementedError

        yield self._storage()
        yield self._location()


class DelimitedFormat(ImpalaBase):
    def __init__(
        self,
        path,
        delimiter=None,
        escapechar=None,
        na_rep=None,
        lineterminator=None,
    ):
        self.path = path
        self.delimiter = delimiter
        self.escapechar = escapechar
        self.lineterminator = lineterminator
        self.na_rep = na_rep

    def to_ddl(self):
        yield "ROW FORMAT DELIMITED"

        if self.delimiter is not None:
            yield f"FIELDS TERMINATED BY '{self.delimiter}'"

        if self.escapechar is not None:
            yield f"ESCAPED BY '{self.escapechar}'"

        if self.lineterminator is not None:
            yield f"LINES TERMINATED BY '{self.lineterminator}'"

        yield "STORED AS TEXTFILE"
        yield f"LOCATION '{self.path}'"

        if self.na_rep is not None:
            props = {"serialization.null.format": self.na_rep}
            yield self.format_tblproperties(props)


class AvroFormat(ImpalaBase):
    def __init__(self, path, avro_schema):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self):
        yield "STORED AS AVRO"
        yield f"LOCATION '{self.path}'"

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = "\n".join(x.rstrip() for x in schema.splitlines())

        props = {"avro.schema.literal": schema}
        yield self.format_tblproperties(props)


class ParquetFormat(ImpalaBase):
    def __init__(self, path):
        self.path = path

    def to_ddl(self):
        yield "STORED AS PARQUET"
        yield f"LOCATION '{self.path}'"


class CreateTableDelimited(CreateTableWithSchema):
    def __init__(
        self,
        table_name,
        path,
        schema,
        delimiter=None,
        escapechar=None,
        lineterminator=None,
        na_rep=None,
        external=True,
        **kwargs,
    ):
        table_format = DelimitedFormat(
            path,
            delimiter=delimiter,
            escapechar=escapechar,
            lineterminator=lineterminator,
            na_rep=na_rep,
        )
        super().__init__(table_name, schema, table_format, external=external, **kwargs)


class CreateTableAvro(CreateTable):
    def __init__(self, table_name, path, avro_schema, external=True, **kwargs):
        super().__init__(table_name, external=external, **kwargs)
        self.table_format = AvroFormat(path, avro_schema)

    @property
    def _pieces(self):
        yield "\n".join(self.table_format.to_ddl())


class LoadData(ImpalaBase, DDL):
    """Generate DDL for LOAD DATA command.

    Cannot be cancelled
    """

    def __init__(
        self,
        table_name,
        path,
        database=None,
        partition=None,
        partition_schema=None,
        overwrite=False,
    ):
        self.table_name = table_name
        self.database = database
        self.path = path

        self.partition = partition
        self.partition_schema = partition_schema

        self.overwrite = overwrite

    def compile(self):
        overwrite = "OVERWRITE " if self.overwrite else ""

        if self.partition is not None:
            partition = "\n" + self.format_partition(
                self.partition, self.partition_schema
            )
        else:
            partition = ""

        scoped_name = self.scoped_name(self.table_name, self.database)
        return "LOAD DATA INPATH '{}' {}INTO TABLE {}{}".format(
            self.path, overwrite, scoped_name, partition
        )


class PartitionProperties(AlterTable):
    def __init__(
        self,
        table,
        partition,
        partition_schema,
        location=None,
        format=None,
        tbl_properties=None,
        serde_properties=None,
    ):
        super().__init__(
            table,
            location=location,
            format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties,
        )
        self.partition = partition
        self.partition_schema = partition_schema

    def _compile(self, cmd, property_prefix=""):
        part = self.format_partition(self.partition, self.partition_schema)
        if cmd:
            part = f"{cmd} {part}"

        props = self._format_properties(property_prefix)
        action = f"{self.table} {part}{props}"
        return self._wrap_command(action)


class AddPartition(PartitionProperties):
    dialect = "hive"

    def __init__(self, table, partition, partition_schema, location=None):
        super().__init__(table, partition, partition_schema, location=location)

    def compile(self):
        return self._compile("ADD")


class AlterPartition(PartitionProperties):
    dialect = "hive"

    def compile(self):
        return self._compile("", "SET ")


class DropPartition(PartitionProperties):
    dialect = "hive"

    def __init__(self, table, partition, partition_schema):
        super().__init__(table, partition, partition_schema)

    def compile(self):
        return self._compile("DROP")


class CacheTable(ImpalaBase, DDL):
    def __init__(self, table_name, database=None, pool="default"):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self.scoped_name(self.table_name, self.database)
        return f"ALTER TABLE {scoped_name} SET CACHED IN '{self.pool}'"


class CreateFunction(ImpalaBase, DDL):
    _object_type = "FUNCTION"

    def __init__(self, func, name=None, database=None):
        self.func = func
        self.name = name or func.name
        self.database = database

    def _impala_signature(self):
        scoped_name = self.scoped_name(self.name, self.database)
        input_sig = ", ".join(map(self.format_dtype, self.func.inputs))
        output_sig = self.format_dtype(self.func.output)

        return f"{scoped_name}({input_sig}) returns {output_sig}"


class CreateUDF(CreateFunction):
    def compile(self):
        create_decl = "CREATE FUNCTION"
        impala_sig = self._impala_signature()
        param_line = f"location '{self.func.lib_path}' symbol='{self.func.so_symbol}'"
        return f"{create_decl} {impala_sig} {param_line}"


class CreateUDA(CreateFunction):
    def compile(self):
        create_decl = "CREATE AGGREGATE FUNCTION"
        impala_sig = self._impala_signature()
        tokens = [f"location '{self.func.lib_path}'"]

        fn_names = (
            "init_fn",
            "update_fn",
            "merge_fn",
            "serialize_fn",
            "finalize_fn",
        )

        for fn in fn_names:
            value = getattr(self.func, fn)
            if value is not None:
                tokens.append(f"{fn}='{value}'")

        joined_tokens = "\n".join(tokens)
        return f"{create_decl} {impala_sig} {joined_tokens}"


class DropFunction(ImpalaBase, DropFunction):
    def _impala_signature(self):
        full_name = self.scoped_name(self.name, self.database)
        input_sig = ", ".join(map(self.format_dtype, self.inputs))
        return f"{full_name}({input_sig})"


class ListFunction(ImpalaBase, DDL):
    def __init__(self, database, like=None, aggregate=False):
        self.database = database
        self.like = like
        self.aggregate = aggregate

    def compile(self):
        statement = "SHOW "
        if self.aggregate:
            statement += "AGGREGATE "
        statement += f"FUNCTIONS IN {self.database}"
        if self.like:
            statement += f" LIKE '{self.like}'"
        return statement


class InsertSelect(ImpalaBase, DML):
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
            part = self.format_partition(self.partition, self.partition_schema)
            partition = f" {part} "
        else:
            partition = ""

        select_query = self.select
        scoped_name = self.scoped_name(self.table_name, self.database)
        return f"{cmd} {scoped_name}{partition}\n{select_query}"
