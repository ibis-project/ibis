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

import re
import json

from ibis.sql.compiler import DDL, DML
from .compiler import quote_identifier, _type_to_sql_string

import ibis.expr.schema as sch
import ibis.expr.datatypes as dt


fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


def _is_fully_qualified(x):
    return bool(fully_qualified_re.search(x))


def _is_quoted(x):
    regex = re.compile(r"(?:`(.*)`|(.*))")
    quoted, _ = regex.match(x).groups()
    return quoted is not None


class ImpalaQualifiedSQLStatement(object):

    def _get_scoped_name(self, obj_name, database):
        if database:
            scoped_name = '{}.`{}`'.format(database, obj_name)
        else:
            if not _is_fully_qualified(obj_name):
                if _is_quoted(obj_name):
                    return obj_name
                else:
                    return '`{}`'.format(obj_name)
            else:
                return obj_name
        return scoped_name


class ImpalaDDL(DDL, ImpalaQualifiedSQLStatement):
    pass


class ImpalaDML(DML, ImpalaQualifiedSQLStatement):
    pass


class CreateDDL(ImpalaDDL):

    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


_format_aliases = {
    'TEXT': 'TEXTFILE'
}


def _sanitize_format(format):
    if format is None:
        return
    format = format.upper()
    format = _format_aliases.get(format, format)
    if format not in ('PARQUET', 'AVRO', 'TEXTFILE'):
        raise ValueError('Invalid format: {!r}'.format(format))

    return format


def _serdeproperties(props):
    formatted_props = _format_properties(props)
    return 'SERDEPROPERTIES {}'.format(formatted_props)


def format_tblproperties(props):
    formatted_props = _format_properties(props)
    return 'TBLPROPERTIES {}'.format(formatted_props)


def _format_properties(props):
    tokens = []
    for k, v in sorted(props.items()):
        tokens.append("  '{}'='{}'".format(k, v))

    return '(\n{}\n)'.format(',\n'.join(tokens))


class CreateTable(CreateDDL):

    """

    Parameters
    ----------
    partition :

    """

    def __init__(self, table_name, database=None, external=False,
                 format='parquet', can_exist=False,
                 partition=None, path=None,
                 tbl_properties=None):
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
            return 'CREATE EXTERNAL TABLE'
        else:
            return 'CREATE TABLE'

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return '{} {}{}'.format(
            self._prefix, self._if_exists(), scoped_name
        )

    def _location(self):
        return "LOCATION '{}'".format(self.path) if self.path else None

    def _storage(self):
        # By the time we're here, we have a valid format
        return 'STORED AS {}'.format(self.format)

    @property
    def pieces(self):
        yield self._create_line()
        for piece in filter(None, self._pieces):
            yield piece

    def compile(self):
        return '\n'.join(self.pieces)


class CTAS(CreateTable):

    """
    Create Table As Select
    """

    def __init__(self, table_name, select, database=None,
                 external=False, format='parquet', can_exist=False,
                 path=None, partition=None):
        super(CTAS, self).__init__(
            table_name, database=database, external=external, format=format,
            can_exist=can_exist, path=path, partition=partition,
        )
        self.select = select

    @property
    def _pieces(self):
        yield self._partitioned_by()
        yield self._storage()
        yield self._location()
        yield 'AS'
        yield self.select.compile()

    def _partitioned_by(self):
        if self.partition is not None:
            return 'PARTITIONED BY ({})'.format(
                ', '.join(
                    quote_identifier(expr._name) for expr in self.partition
                )
            )
        return None


class CreateView(CTAS):

    """Create a view"""

    def __init__(self, table_name, select, database=None, can_exist=False):
        super(CreateView, self).__init__(
            table_name, select, database=database, can_exist=can_exist
        )

    @property
    def _pieces(self):
        yield 'AS'
        yield self.select.compile()

    @property
    def _prefix(self):
        return 'CREATE VIEW'


class CreateTableParquet(CreateTable):

    def __init__(self, table_name, path, example_file=None, example_table=None,
                 schema=None, external=True, **kwargs):
        super(CreateTableParquet, self).__init__(
            table_name, external=external, format='parquet', path=path,
            **kwargs
        )
        self.example_file = example_file
        self.example_table = example_table
        self.schema = schema

    @property
    def _pieces(self):
        if self.example_file is not None:
            yield "LIKE PARQUET '{0}'".format(self.example_file)
        elif self.example_table is not None:
            yield "LIKE {0}".format(self.example_table)
        elif self.schema is not None:
            yield format_schema(self.schema)
        else:
            raise NotImplementedError

        yield self._storage()
        yield self._location()


class CreateTableWithSchema(CreateTable):

    def __init__(self, table_name, schema, table_format=None, **kwargs):
        super(CreateTableWithSchema, self).__init__(table_name, **kwargs)
        self.schema = schema
        self.table_format = table_format

    @property
    def _pieces(self):
        if self.partition is not None:
            main_schema = self.schema
            part_schema = self.partition
            if not isinstance(part_schema, sch.Schema):
                part_schema = sch.Schema(
                    part_schema,
                    [self.schema[name] for name in part_schema])

            to_delete = []
            for name in self.partition:
                if name in self.schema:
                    to_delete.append(name)

            if len(to_delete):
                main_schema = main_schema.delete(to_delete)

            yield format_schema(main_schema)
            yield 'PARTITIONED BY {}'.format(format_schema(part_schema))
        else:
            yield format_schema(self.schema)

        if self.table_format is not None:
            yield '\n'.join(self.table_format.to_ddl())
        else:
            yield self._storage()

        yield self._location()


class DelimitedFormat(object):

    def __init__(self, path, delimiter=None, escapechar=None,
                 na_rep=None, lineterminator=None):
        self.path = path
        self.delimiter = delimiter
        self.escapechar = escapechar
        self.lineterminator = lineterminator
        self.na_rep = na_rep

    def to_ddl(self):
        yield 'ROW FORMAT DELIMITED'

        if self.delimiter is not None:
            yield "FIELDS TERMINATED BY '{}'".format(self.delimiter)

        if self.escapechar is not None:
            yield "ESCAPED BY '{}'".format(self.escapechar)

        if self.lineterminator is not None:
            yield "LINES TERMINATED BY '{}'".format(self.lineterminator)

        yield "LOCATION '{}'".format(self.path)

        if self.na_rep is not None:
            props = {
                'serialization.null.format': self.na_rep
            }
            yield format_tblproperties(props)


class AvroFormat(object):

    def __init__(self, path, avro_schema):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self):
        yield 'STORED AS AVRO'
        yield "LOCATION '{}'".format(self.path)

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = '\n'.join(x.rstrip() for x in schema.splitlines())

        props = {'avro.schema.literal': schema}
        yield format_tblproperties(props)


class ParquetFormat(object):

    def __init__(self, path):
        self.path = path

    def to_ddl(self):
        yield 'STORED AS PARQUET'
        yield "LOCATION '{}'".format(self.path)


class CreateTableDelimited(CreateTableWithSchema):

    def __init__(self, table_name, path, schema,
                 delimiter=None, escapechar=None, lineterminator=None,
                 na_rep=None, external=True, **kwargs):
        table_format = DelimitedFormat(path, delimiter=delimiter,
                                       escapechar=escapechar,
                                       lineterminator=lineterminator,
                                       na_rep=na_rep)
        super(CreateTableDelimited, self).__init__(
            table_name, schema, table_format, external=external, **kwargs
        )


class CreateTableAvro(CreateTable):

    def __init__(self, table_name, path, avro_schema, external=True, **kwargs):
        super(CreateTableAvro, self).__init__(
            table_name, external=external, **kwargs
        )
        self.table_format = AvroFormat(path, avro_schema)

    @property
    def _pieces(self):
        yield '\n'.join(self.table_format.to_ddl())


class InsertSelect(ImpalaDML):

    def __init__(self, table_name, select_expr, database=None,
                 partition=None,
                 partition_schema=None,
                 overwrite=False):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

        self.partition = partition
        self.partition_schema = partition_schema

        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE'
        else:
            cmd = 'INSERT INTO'

        if self.partition is not None:
            part = _format_partition(self.partition,
                                     self.partition_schema)
            partition = ' {} '.format(part)
        else:
            partition = ''

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return'{0} {1}{2}\n{3}'.format(cmd, scoped_name, partition,
                                       select_query)


def _format_partition(partition, partition_schema):
    tokens = []
    if isinstance(partition, dict):
        for name in partition_schema:
            if name in partition:
                tok = _format_partition_kv(name, partition[name],
                                           partition_schema[name])
            else:
                # dynamic partitioning
                tok = name
            tokens.append(tok)
    else:
        for name, value in zip(partition_schema, partition):
            tok = _format_partition_kv(name, value, partition_schema[name])
            tokens.append(tok)

    return 'PARTITION ({})'.format(', '.join(tokens))


def _format_partition_kv(k, v, type):
    if type == dt.string:
        value_formatted = '"{}"'.format(v)
    else:
        value_formatted = str(v)

    return '{}={}'.format(k, value_formatted)


class LoadData(ImpalaDDL):

    """
    Generate DDL for LOAD DATA command. Cannot be cancelled
    """

    def __init__(self, table_name, path, database=None,
                 partition=None, partition_schema=None,
                 overwrite=False):
        self.table_name = table_name
        self.database = database
        self.path = path

        self.partition = partition
        self.partition_schema = partition_schema

        self.overwrite = overwrite

    def compile(self):
        overwrite = 'OVERWRITE ' if self.overwrite else ''

        if self.partition is not None:
            partition = '\n' + _format_partition(self.partition,
                                                 self.partition_schema)
        else:
            partition = ''

        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return ("LOAD DATA INPATH '{}' {}INTO TABLE {}{}"
                .format(self.path, overwrite, scoped_name, partition))


class AlterTable(ImpalaDDL):

    def __init__(self, table, location=None, format=None, tbl_properties=None,
                 serde_properties=None):
        self.table = table
        self.location = location
        self.format = _sanitize_format(format)
        self.tbl_properties = tbl_properties
        self.serde_properties = serde_properties

    def _wrap_command(self, cmd):
        return 'ALTER TABLE {}'.format(cmd)

    def _format_properties(self, prefix=''):
        tokens = []

        if self.location is not None:
            tokens.append("LOCATION '{}'".format(self.location))

        if self.format is not None:
            tokens.append("FILEFORMAT {}".format(self.format))

        if self.tbl_properties is not None:
            tokens.append(format_tblproperties(self.tbl_properties))

        if self.serde_properties is not None:
            tokens.append(_serdeproperties(self.serde_properties))

        if len(tokens) > 0:
            return '\n{}{}'.format(prefix, '\n'.join(tokens))
        else:
            return ''

    def compile(self):
        props = self._format_properties()
        action = '{} SET {}'.format(self.table, props)
        return self._wrap_command(action)


class PartitionProperties(AlterTable):

    def __init__(self, table, partition, partition_schema,
                 location=None, format=None,
                 tbl_properties=None, serde_properties=None):
        super(PartitionProperties, self).__init__(
            table,
            location=location, format=format,
            tbl_properties=tbl_properties,
            serde_properties=serde_properties
        )
        self.partition = partition
        self.partition_schema = partition_schema

    def _compile(self, cmd, property_prefix=''):
        part = _format_partition(self.partition, self.partition_schema)
        if cmd:
            part = '{} {}'.format(cmd, part)

        props = self._format_properties(property_prefix)
        action = '{} {}{}'.format(self.table, part, props)
        return self._wrap_command(action)


class AddPartition(PartitionProperties):

    def __init__(self, table, partition, partition_schema, location=None):
        super(AddPartition, self).__init__(
            table, partition, partition_schema, location=location
        )

    def compile(self):
        return self._compile('ADD')


class AlterPartition(PartitionProperties):

    def compile(self):
        return self._compile('', 'SET ')


class DropPartition(PartitionProperties):

    def __init__(self, table, partition, partition_schema):
        super(DropPartition, self).__init__(table, partition, partition_schema)

    def compile(self):
        return self._compile('DROP')


class RenameTable(AlterTable):

    def __init__(self, old_name, new_name, old_database=None,
                 new_database=None):
        # if either database is None, the name is assumed to be fully scoped
        self.old_name = old_name
        self.old_database = old_database
        self.new_name = new_name
        self.new_database = new_database

        new_qualified_name = new_name
        if new_database is not None:
            new_qualified_name = self._get_scoped_name(new_name, new_database)

        old_qualified_name = old_name
        if old_database is not None:
            old_qualified_name = self._get_scoped_name(old_name, old_database)

        self.old_qualified_name = old_qualified_name
        self.new_qualified_name = new_qualified_name

    def compile(self):
        cmd = '{} RENAME TO {}'.format(self.old_qualified_name,
                                       self.new_qualified_name)
        return self._wrap_command(cmd)


class DropObject(ImpalaDDL):

    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        return 'DROP {} {}{}'.format(self._object_type, if_exists, object_name)


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        super(DropTable, self).__init__(must_exist=must_exist)
        self.table_name = table_name
        self.database = database

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class TruncateTable(ImpalaDDL):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {}'.format(name)


class DropView(DropTable):

    _object_type = 'VIEW'


class CacheTable(ImpalaDDL):

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return "ALTER TABLE {} SET CACHED IN '{}'" .format(
            scoped_name, self.pool
        )


class CreateDatabase(CreateDDL):

    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = quote_identifier(self.name)

        create_decl = 'CREATE DATABASE'
        create_line = '{} {}{}'.format(create_decl, self._if_exists(), name)
        if self.path is not None:
            create_line += "\nLOCATION '{}'".format(self.path)

        return create_line


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True):
        super(DropDatabase, self).__init__(must_exist=must_exist)
        self.name = name

    def _object_name(self):
        return self.name


def format_schema(schema):
    elements = [_format_schema_element(name, t)
                for name, t in zip(schema.names, schema.types)]
    return '({})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{} {}'.format(
        quote_identifier(name, force=True), _type_to_sql_string(t)
    )


class CreateFunction(ImpalaDDL):

    _object_type = 'FUNCTION'

    def __init__(self, func, name=None, database=None):
        self.func = func
        self.name = name or func.name
        self.database = database

    def _impala_signature(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        input_sig = _impala_input_signature(self.func.inputs)
        output_sig = _type_to_sql_string(self.func.output)

        return '{}({}) returns {}'.format(scoped_name, input_sig, output_sig)


class CreateUDF(CreateFunction):

    def compile(self):
        create_decl = 'CREATE FUNCTION'
        impala_sig = self._impala_signature()
        param_line = ("location '{}' symbol='{}'"
                      .format(self.func.lib_path, self.func.so_symbol))
        return ' '.join([create_decl, impala_sig, param_line])


class CreateUDA(CreateFunction):

    def compile(self):
        create_decl = 'CREATE AGGREGATE FUNCTION'
        impala_sig = self._impala_signature()
        tokens = ["location '{}'".format(self.func.lib_path)]

        fn_names = ('init_fn', 'update_fn', 'merge_fn', 'serialize_fn',
                    'finalize_fn')

        for fn in fn_names:
            value = getattr(self.func, fn)
            if value is not None:
                tokens.append("{}='{}'".format(fn, value))

        return ' '.join([create_decl, impala_sig]) + ' ' + '\n'.join(tokens)


class DropFunction(DropObject):

    def __init__(self, name, inputs, must_exist=True,
                 aggregate=False, database=None):
        super(DropFunction, self).__init__(must_exist=must_exist)
        self.name = name
        self.inputs = tuple(map(dt.dtype, inputs))
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database

    def _impala_signature(self):
        full_name = self._get_scoped_name(self.name, self.database)
        input_sig = _impala_input_signature(self.inputs)
        return '{}({})'.format(full_name, input_sig)

    def _object_name(self):
        return self.name

    def compile(self):
        tokens = ['DROP']
        if self.aggregate:
            tokens.append('AGGREGATE')
        tokens.append('FUNCTION')
        if not self.must_exist:
            tokens.append('IF EXISTS')

        tokens.append(self._impala_signature())
        return ' '.join(tokens)


class ListFunction(ImpalaDDL):

    def __init__(self, database, like=None, aggregate=False):
        self.database = database
        self.like = like
        self.aggregate = aggregate

    def compile(self):
        statement = 'SHOW '
        if self.aggregate:
            statement += 'AGGREGATE '
        statement += 'FUNCTIONS IN {}'.format(self.database)
        if self.like:
            statement += " LIKE '{}'".format(self.like)
        return statement


def _impala_input_signature(inputs):
    # TODO: varargs '{}...'.format(val)
    return ', '.join(map(_type_to_sql_string, inputs))
