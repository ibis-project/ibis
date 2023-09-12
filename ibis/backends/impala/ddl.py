from __future__ import annotations

# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

from ibis.backends.base.sql.ddl import (
    AlterTable,
    BaseDDL,
    CreateTable,
    CreateTableWithSchema,
    DropFunction,
    format_partition,
    format_schema,
    format_tblproperties,
)
from ibis.backends.base.sql.registry import type_to_sql_string


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
            yield format_schema(self.schema)
        else:
            raise NotImplementedError

        yield self._storage()
        yield self._location()


class DelimitedFormat:
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
            yield format_tblproperties(props)


class AvroFormat:
    def __init__(self, path, avro_schema):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self):
        yield "STORED AS AVRO"
        yield f"LOCATION '{self.path}'"

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = "\n".join(x.rstrip() for x in schema.splitlines())

        props = {"avro.schema.literal": schema}
        yield format_tblproperties(props)


class ParquetFormat:
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


class LoadData(BaseDDL):
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
            partition = "\n" + format_partition(self.partition, self.partition_schema)
        else:
            partition = ""

        scoped_name = self._get_scoped_name(self.table_name, self.database)
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
        part = format_partition(self.partition, self.partition_schema)
        if cmd:
            part = f"{cmd} {part}"

        props = self._format_properties(property_prefix)
        action = f"{self.table} {part}{props}"
        return self._wrap_command(action)


class AddPartition(PartitionProperties):
    def __init__(self, table, partition, partition_schema, location=None):
        super().__init__(table, partition, partition_schema, location=location)

    def compile(self):
        return self._compile("ADD")


class AlterPartition(PartitionProperties):
    def compile(self):
        return self._compile("", "SET ")


class DropPartition(PartitionProperties):
    def __init__(self, table, partition, partition_schema):
        super().__init__(table, partition, partition_schema)

    def compile(self):
        return self._compile("DROP")


class CacheTable(BaseDDL):
    def __init__(self, table_name, database=None, pool="default"):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return f"ALTER TABLE {scoped_name} SET CACHED IN '{self.pool}'"


class CreateFunction(BaseDDL):
    _object_type = "FUNCTION"

    def __init__(self, func, name=None, database=None):
        self.func = func
        self.name = name or func.name
        self.database = database

    def _impala_signature(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        input_sig = _impala_input_signature(self.func.inputs)
        output_sig = type_to_sql_string(self.func.output)

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


class DropFunction(DropFunction):
    def _impala_signature(self):
        full_name = self._get_scoped_name(self.name, self.database)
        input_sig = _impala_input_signature(self.inputs)
        return f"{full_name}({input_sig})"


class ListFunction(BaseDDL):
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


def _impala_input_signature(inputs):
    # TODO: varargs '{}...'.format(val)
    return ", ".join(map(type_to_sql_string, inputs))
