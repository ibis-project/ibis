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

from io import BytesIO
import re

from ibis.sql.exprs import (ExprTranslator, quote_identifier,
                            _sql_type_names, _type_to_sql_string)
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.common as com
import ibis.util as util


class DDLStatement(object):

    def _get_scoped_name(self, table_name, database):
        if database:
            scoped_name = '{0}.`{1}`'.format(database, table_name)
        else:
            if not _is_fully_qualified(table_name):
                if _is_quoted(table_name):
                    return table_name
                else:
                    return '`{0}`'.format(table_name)
            else:
                return table_name
        return scoped_name


def _is_fully_qualified(x):
    regex = re.compile("(.*)\.(?:`(.*)`|(.*))")
    m = regex.search(x)
    return bool(m)


def _is_quoted(x):
    regex = re.compile("(?:`(.*)`|(.*))")
    quoted, unquoted = regex.match(x).groups()
    return quoted is not None


class Select(DDLStatement):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(self, context, table_set, select_set,
                 subqueries=None, where=None, group_by=None, having=None,
                 order_by=None, limit=None,
                 distinct=False, indent=2,
                 result_handler=None, parent_expr=None):
        self.context = context

        self.select_set = select_set
        self.table_set = table_set
        self.distinct = distinct

        self.parent_expr = parent_expr

        self.where = where or []

        # Group keys and post-predicates for aggregations
        self.group_by = group_by or []
        self.having = having or []
        self.order_by = order_by or []

        self.limit = limit
        self.subqueries = subqueries or []

        self.indent = indent

        self.result_handler = result_handler

    def equals(self, other):
        if not isinstance(other, Select):
            return False

        this_exprs = self._all_exprs()
        other_exprs = other._all_exprs()

        if self.limit != other.limit:
            return False

        for x, y in zip(this_exprs, other_exprs):
            if not x.equals(y):
                return False

        return True

    def _all_exprs(self):
        # Gnarly, maybe we can improve this somehow
        expr_attrs = ['select_set', 'table_set', 'where', 'group_by', 'having',
                      'order_by', 'subqueries']
        exprs = []
        for attr in expr_attrs:
            val = getattr(self, attr)
            if isinstance(val, list):
                exprs.extend(val)
            else:
                exprs.append(val)

        return exprs

    def compile(self, context=None, semicolon=False):
        """
        This method isn't yet idempotent; calling multiple times may yield
        unexpected results
        """
        if context is None:
            context = self.context

        # Can't tell if this is a hack or not. Revisit later
        context.set_query(self)

        # If any subqueries, translate them and add to beginning of query as
        # part of the WITH section
        with_frag = self.format_subqueries(context)

        # SELECT
        select_frag = self.format_select_set(context)

        # FROM, JOIN, UNION
        from_frag = self.format_table_set(context)

        # WHERE
        where_frag = self.format_where(context)

        # GROUP BY and HAVING
        groupby_frag = self.format_group_by(context)

        # ORDER BY and LIMIT
        order_frag = self.format_postamble(context)

        # Glue together the query fragments and return
        query = _join_not_none('\n', [with_frag, select_frag, from_frag,
                                      where_frag, groupby_frag, order_frag])

        return query

    def format_subqueries(self, context):
        if len(self.subqueries) == 0:
            return

        buf = BytesIO()
        buf.write('WITH ')

        for i, expr in enumerate(self.subqueries):
            if i > 0:
                buf.write(',\n')
            formatted = util.indent(context.get_formatted_query(expr), 2)
            alias = context.get_alias(expr)
            buf.write('{0} AS (\n{1}\n)'.format(alias, formatted))

        return buf.getvalue()

    def format_select_set(self, context):
        # TODO:
        formatted = []
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                expr_str = translate_expr(expr, context=context, named=True)
            elif isinstance(expr, ir.TableExpr):
                # A * selection, possibly prefixed
                if context.need_aliases():
                    alias = context.get_alias(expr)

                    # materialized join will not have an alias. see #491
                    expr_str = '{0}.*'.format(alias) if alias else '*'
                else:
                    expr_str = '*'
            formatted.append(expr_str)

        buf = BytesIO()
        line_length = 0
        max_length = 70
        tokens = 0
        for i, val in enumerate(formatted):
            # always line-break for multi-line expressions
            if val.count('\n'):
                if i:
                    buf.write(',')
                buf.write('\n')
                indented = util.indent(val, self.indent)
                buf.write(indented)

                # set length of last line
                line_length = len(indented.split('\n')[-1])
                tokens = 1
            elif (tokens > 0 and line_length and
                  len(val) + line_length > max_length):
                # There is an expr, and adding this new one will make the line
                # too long
                buf.write(',\n       ') if i else buf.write('\n')
                buf.write(val)
                line_length = len(val) + 7
                tokens = 1
            else:
                if i:
                    buf.write(',')
                buf.write(' ')
                buf.write(val)
                tokens += 1
                line_length += len(val) + 2

        if self.distinct:
            select_key = 'SELECT DISTINCT'
        else:
            select_key = 'SELECT'

        return '{0}{1}'.format(select_key, buf.getvalue())

    def format_table_set(self, ctx):
        if self.table_set is None:
            return None

        fragment = 'FROM '

        helper = _TableSetFormatter(ctx, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_group_by(self, context):
        if len(self.group_by) == 0:
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            clause = 'GROUP BY {0}'.format(', '.join([
                str(x + 1) for x in self.group_by]))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = translate_expr(expr, context=context)
                trans_exprs.append(translated)
            lines.append('HAVING {0}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_where(self, context):
        if len(self.where) == 0:
            return None

        buf = BytesIO()
        buf.write('WHERE ')
        fmt_preds = [translate_expr(pred, context=context,
                                    permit_subquery=True)
                     for pred in self.where]
        conj = ' AND\n{0}'.format(' ' * 6)
        buf.write(conj.join(fmt_preds))
        return buf.getvalue()

    def format_postamble(self, context):
        buf = BytesIO()
        lines = 0

        if len(self.order_by) > 0:
            buf.write('ORDER BY ')
            formatted = []
            for key in self.order_by:
                translated = translate_expr(key.expr, context=context)
                if not key.ascending:
                    translated += ' DESC'
                formatted.append(translated)
            buf.write(', '.join(formatted))
            lines += 1

        if self.limit is not None:
            if lines:
                buf.write('\n')
            n, offset = self.limit['n'], self.limit['offset']
            buf.write('LIMIT {0}'.format(n))
            if offset is not None and offset != 0:
                buf.write(' OFFSET {0}'.format(offset))
            lines += 1

        if not lines:
            return None

        return buf.getvalue()


class _TableSetFormatter(object):
    _join_names = {
        ops.InnerJoin: 'INNER JOIN',
        ops.LeftJoin: 'LEFT OUTER JOIN',
        ops.RightJoin: 'RIGHT OUTER JOIN',
        ops.OuterJoin: 'FULL OUTER JOIN',
        ops.LeftAntiJoin: 'LEFT ANTI JOIN',
        ops.LeftSemiJoin: 'LEFT SEMI JOIN',
        ops.CrossJoin: 'CROSS JOIN'
    }

    def __init__(self, context, expr, indent=2):
        self.context = context
        self.expr = expr
        self.indent = indent

        self.join_tables = []
        self.join_types = []
        self.join_predicates = []

    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        # TODO: Now actually format the things
        buf = BytesIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(self.join_types, self.join_tables[1:],
                                       self.join_predicates):
            buf.write('\n')
            buf.write(util.indent('{0} {1}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = [translate_expr(pred, context=self.context)
                             for pred in preds]
                conj = ' AND\n{0}'.format(' ' * 3)
                fmt_preds = util.indent('ON ' + conj.join(fmt_preds),
                                        self.indent * 2)
                buf.write(fmt_preds)

        return buf.getvalue()

    def _walk_join_tree(self, op):
        left = op.left.op()
        right = op.right.op()

        if util.all_of([left, right], ops.Join):
            raise NotImplementedError('Do not support joins between '
                                      'joins yet')

        self._validate_join_predicates(op.predicates)

        jname = self._join_names[type(op)]

        # Impala requires this
        if len(op.predicates) == 0:
            jname = self._join_names[ops.CrossJoin]

        # Read off tables and join predicates left-to-right in
        # depth-first order
        if isinstance(left, ops.Join):
            self._walk_join_tree(left)
            self.join_tables.append(self._format_table(op.right))
            self.join_types.append(jname)
            self.join_predicates.append(op.predicates)
        elif isinstance(right, ops.Join):
            # When rewrites are possible at the expression IR stage, we should
            # do them. Otherwise subqueries might be necessary in some cases
            # here
            raise NotImplementedError('not allowing joins on right '
                                      'side yet')
        else:
            # Both tables
            self.join_tables.append(self._format_table(op.left))
            self.join_tables.append(self._format_table(op.right))
            self.join_types.append(jname)
            self.join_predicates.append(op.predicates)

    def _format_table(self, expr):
        return _format_table(self.context, expr)

    # Placeholder; revisit when supporting other databases
    _non_equijoin_supported = True

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()

            if (not isinstance(op, ops.Equals) and
                    not self._non_equijoin_supported):
                raise com.TranslationError(
                    'Non-equality join predicates, '
                    'i.e. non-equijoins, are not supported')


def _format_table(ctx, expr, indent=2):
    # TODO: This could probably go in a class and be significantly nicer

    ref_expr = expr
    op = ref_op = expr.op()
    if isinstance(op, ops.SelfReference):
        ref_expr = op.table
        ref_op = ref_expr.op()

    if isinstance(ref_op, ops.PhysicalTable):
        name = ref_op.name
        if name is None:
            raise com.RelationError('Table did not have a name: {0!r}'
                                    .format(expr))
        result = quote_identifier(name)
        is_subquery = False
    else:
        # A subquery
        if ctx.is_extracted(ref_expr):
            # Was put elsewhere, e.g. WITH block, we just need to grab its
            # alias
            alias = ctx.get_alias(expr)

            # HACK: self-references have to be treated more carefully here
            if isinstance(op, ops.SelfReference):
                return '{0} {1}'.format(ctx.get_alias(ref_expr), alias)
            else:
                return alias

        subquery = ctx.get_formatted_query(expr)
        result = '(\n{0}\n)'.format(util.indent(subquery, indent))
        is_subquery = True

    if is_subquery or ctx.need_aliases():
        result += ' {0}'.format(ctx.get_alias(expr))

    return result


class Union(DDLStatement):

    def __init__(self, left_table, right_table, distinct=False,
                 context=None):
        self.context = context
        self.left = left_table
        self.right = right_table

        self.distinct = distinct

    def compile(self, context=None, semicolon=False):
        if context is None:
            context = self.context

        if self.distinct:
            union_keyword = 'UNION'
        else:
            union_keyword = 'UNION ALL'

        left_set = context.get_formatted_query(self.left)
        right_set = context.get_formatted_query(self.right)

        query = '{0}\n{1}\n{2}'.format(left_set, union_keyword, right_set)
        return query


class CreateDDL(DDLStatement):

    def _if_exists(self):
        return 'IF NOT EXISTS ' if self.can_exist else ''


class CreateTable(CreateDDL):

    """

    Parameters
    ----------
    partition :

    """

    def __init__(self, table_name, database=None, external=False,
                 format='parquet', can_exist=False,
                 partition=None, path=None):
        self.table_name = table_name
        self.database = database
        self.partition = partition
        self.path = path
        self.external = external
        self.can_exist = can_exist
        self.format = self._validate_storage_format(format)

    def _validate_storage_format(self, format):
        format = format.lower()
        if format not in ('parquet', 'avro'):
            raise ValueError('Invalid format: {0}'.format(format))
        return format

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)

        if self.external:
            create_decl = 'CREATE EXTERNAL TABLE'
        else:
            create_decl = 'CREATE TABLE'

        create_line = '{0} {1}{2}'.format(create_decl, self._if_exists(),
                                          scoped_name)
        return create_line

    def _location(self):
        if self.path:
            return "\nLOCATION '{0}'".format(self.path)
        return ''

    def _storage(self):
        storage_lines = {
            'parquet': '\nSTORED AS PARQUET',
            'avro': '\nSTORED AS AVRO'
        }
        return storage_lines[self.format]


class CTAS(CreateTable):

    """
    Create Table As Select
    """

    def __init__(self, table_name, select, database=None,
                 external=False, format='parquet', can_exist=False,
                 path=None):
        self.select = select
        CreateTable.__init__(self, table_name, database=database,
                             external=external, format=format,
                             can_exist=can_exist, path=path)

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())
        buf.write(self._storage())
        buf.write(self._location())

        select_query = self.select.compile()
        buf.write('\nAS\n{0}'.format(select_query))
        return buf.getvalue()


class CreateView(CreateDDL):

    """
    Create Table As Select
    """

    def __init__(self, name, select, database=None, can_exist=False):
        self.name = name
        self.database = database
        self.select = select
        self.can_exist = can_exist

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        select_query = self.select.compile()
        buf.write('\nAS\n{0}'.format(select_query))
        return buf.getvalue()

    def _create_line(self):
        scoped_name = self._get_scoped_name(self.name, self.database)
        return '{0} {1}{2}'.format('CREATE VIEW', self._if_exists(),
                                   scoped_name)


class CreateTableParquet(CreateTable):

    def __init__(self, table_name, path,
                 example_file=None,
                 example_table=None,
                 schema=None,
                 external=True,
                 **kwargs):
        self.example_file = example_file
        self.example_table = example_table
        self.schema = schema
        CreateTable.__init__(self, table_name, external=external,
                             format='parquet', path=path, **kwargs)

        self._validate()

    def _validate(self):
        pass

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        if self.example_file is not None:
            buf.write("\nLIKE PARQUET '{0}'".format(self.example_file))
        elif self.example_table is not None:
            buf.write("\nLIKE {0}".format(self.example_table))
        elif self.schema is not None:
            schema = format_schema(self.schema)
            buf.write('\n{0}'.format(schema))
        else:
            raise NotImplementedError

        buf.write(self._storage())
        buf.write(self._location())
        return buf.getvalue()


class CreateTableWithSchema(CreateTable):

    def __init__(self, table_name, schema, table_format, **kwargs):
        self.schema = schema
        self.table_format = table_format

        CreateTable.__init__(self, table_name, **kwargs)

    def compile(self):
        from ibis.expr.api import schema

        buf = BytesIO()
        buf.write(self._create_line())

        def _push_schema(x):
            formatted = format_schema(x)
            buf.write('{0}'.format(formatted))

        if self.partition is not None:
            modified_schema = []
            partition_schema = []
            for name, dtype in zip(self.schema.names, self.schema.types):
                if name in self.partition:
                    partition_schema.append((name, dtype))
                else:
                    modified_schema.append((name, dtype))

            buf.write('\n')
            _push_schema(schema(modified_schema))
            buf.write('\nPARTITIONED BY ')
            _push_schema(schema(partition_schema))
        else:
            buf.write('\n')
            _push_schema(self.schema)

        format_ddl = self.table_format.to_ddl()
        if format_ddl:
            buf.write(format_ddl)

        buf.write(self._location())

        return buf.getvalue()


class NoFormat(object):

    def to_ddl(self):
        return None


class DelimitedFormat(object):

    def __init__(self, path, delimiter=None, escapechar=None,
                 lineterminator=None):
        self.path = path
        self.delimiter = delimiter
        self.escapechar = escapechar
        self.lineterminator = lineterminator

    def to_ddl(self):
        buf = BytesIO()

        buf.write("\nROW FORMAT DELIMITED")

        if self.delimiter is not None:
            buf.write("\nFIELDS TERMINATED BY '{0}'".format(self.delimiter))

        if self.escapechar is not None:
            buf.write("\nESCAPED BY '{0}'".format(self.escapechar))

        if self.lineterminator is not None:
            buf.write("\nLINES TERMINATED BY '{0}'"
                      .format(self.lineterminator))

        buf.write("\nLOCATION '{0}'".format(self.path))

        return buf.getvalue()


class AvroFormat(object):

    def __init__(self, path, avro_schema):
        self.path = path
        self.avro_schema = avro_schema

    def to_ddl(self):
        import json

        buf = BytesIO()
        buf.write('\nSTORED AS AVRO')
        buf.write("\nLOCATION '{0}'".format(self.path))

        schema = json.dumps(self.avro_schema, indent=2, sort_keys=True)
        schema = '\n'.join([x.rstrip() for x in schema.split('\n')])
        buf.write("\nTBLPROPERTIES ('avro.schema.literal'='{0}')"
                  .format(schema))

        return buf.getvalue()


class CreateTableDelimited(CreateTableWithSchema):

    def __init__(self, table_name, path, schema,
                 delimiter=None, escapechar=None, lineterminator=None,
                 external=True, **kwargs):
        table_format = DelimitedFormat(path, delimiter=delimiter,
                                       escapechar=escapechar,
                                       lineterminator=lineterminator)
        CreateTableWithSchema.__init__(self, table_name, schema,
                                       table_format, external=external,
                                       **kwargs)


class CreateTableAvro(CreateTable):

    def __init__(self, table_name, path, avro_schema, external=True, **kwargs):
        self.table_format = AvroFormat(path, avro_schema)

        CreateTable.__init__(self, table_name, external=external, **kwargs)

    def compile(self):
        buf = BytesIO()
        buf.write(self._create_line())

        format_ddl = self.table_format.to_ddl()
        buf.write(format_ddl)

        return buf.getvalue()


class InsertSelect(DDLStatement):

    def __init__(self, table_name, select_expr, database=None,
                 overwrite=False):
        self.table_name = table_name
        self.database = database
        self.select = select_expr

        self.overwrite = overwrite

    def compile(self):
        if self.overwrite:
            cmd = 'INSERT OVERWRITE'
        else:
            cmd = 'INSERT INTO'

        select_query = self.select.compile()
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        return'{0} {1}\n{2}'.format(cmd, scoped_name, select_query)


class DropObject(DDLStatement):

    def __init__(self, must_exist=True):
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        object_name = self._object_name()
        drop_line = 'DROP {0} {1}{2}'.format(self._object_type, if_exists,
                                             object_name)
        return drop_line


class DropTable(DropObject):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None, must_exist=True):
        self.table_name = table_name
        self.database = database
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self._get_scoped_name(self.table_name, self.database)


class TruncateTable(DDLStatement):

    _object_type = 'TABLE'

    def __init__(self, table_name, database=None):
        self.table_name = table_name
        self.database = database

    def compile(self):
        name = self._get_scoped_name(self.table_name, self.database)
        return 'TRUNCATE TABLE {0}'.format(name)


class DropView(DropTable):

    _object_type = 'VIEW'


class CacheTable(DDLStatement):

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        cache_line = ('ALTER TABLE {0} SET CACHED IN \'{1}\''
                      .format(scoped_name, self.pool))
        return cache_line


class CreateDatabase(CreateDDL):

    def __init__(self, name, path=None, can_exist=False):
        self.name = name
        self.path = path
        self.can_exist = can_exist

    def compile(self):
        name = quote_identifier(self.name)

        create_decl = 'CREATE DATABASE'
        create_line = '{0} {1}{2}'.format(create_decl, self._if_exists(),
                                          name)
        if self.path is not None:
            create_line += "\nLOCATION '{0}'".format(self.path)

        return create_line


class DropDatabase(DropObject):

    _object_type = 'DATABASE'

    def __init__(self, name, must_exist=True):
        self.name = name
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self.name


def _join_not_none(sep, pieces):
    pieces = [x for x in pieces if x is not None]
    return sep.join(pieces)


def format_schema(schema):
    elements = [_format_schema_element(name, t)
                for name, t in zip(schema.names, schema.types)]
    return '({0})'.format(',\n '.join(elements))


def _format_schema_element(name, t):
    return '{0} {1}'.format(quote_identifier(name, force=True),
                            _format_type(t))


def _format_type(t):
    if isinstance(t, dt.Decimal):
        return 'DECIMAL({0},{1})'.format(t.precision, t.scale)
    else:
        return _impala_type_names[t]


class CreateFunction(DDLStatement):

    _object_type = 'FUNCTION'

    def __init__(self, hdfs_file, so_symbol, inputs, output,
                 name, database=None):
        self.hdfs_file = hdfs_file
        self.so_symbol = so_symbol
        self.inputs = [_type_to_sql_string(x) for x in inputs]
        self.output = _type_to_sql_string(output)
        self.name = name
        self.database = database

    def get_name(self):
        return self.name

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def compile(self):
        create_decl = 'CREATE FUNCTION'
        scoped_name = self._get_scoped_name()
        create_line = ('{0!s}({1!s}) returns {2!s}'
                       .format(scoped_name, ', '.join(self.inputs),
                               self.output))
        param_line = "location '{0!s}' symbol='{1!s}'".format(self.hdfs_file,
                                                              self.so_symbol)
        full_line = ' '.join([create_decl, create_line, param_line])
        return full_line


class CreateAggregateFunction(DDLStatement):

    _object_type = 'FUNCTION'

    def __init__(self, hdfs_file, inputs, output, init_fn, update_fn,
                 merge_fn, finalize_fn, name, database=None):
        self.hdfs_file = hdfs_file
        self.inputs = [_type_to_sql_string(x) for x in inputs]
        self.output = _type_to_sql_string(output)
        self.init = init_fn
        self.update = update_fn
        self.merge = merge_fn
        self.finalize = finalize_fn
        self.name = name
        self.database = database

    def get_name(self):
        return self.name

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def compile(self):
        create_decl = 'CREATE AGGREGATE FUNCTION'
        scoped_name = self._get_scoped_name()
        create_line = ('{0!s}({1!s}) returns {2!s}'
                       .format(scoped_name, ', '.join(self.inputs),
                               self.output))
        loc_ln = "location '{0!s}'".format(self.hdfs_file)
        init_ln = "init_fn='{0}'".format(self.init)
        update_ln = "update_fn='{0}'".format(self.update)
        merge_ln = "merge_fn='{0}'".format(self.merge)
        finalize_ln = "finalize_fn='{0}'".format(self.finalize)
        full_line = ' '.join([create_decl, create_line, loc_ln,
                              init_ln, update_ln, merge_ln, finalize_ln])
        return full_line


class DropFunction(DropObject):

    def __init__(self, name, input_types, must_exist=True,
                 aggregate=False, database=None):
        self.name = name
        self.inputs = [self._ibis_string_to_impala(x) for x in input_types]
        self.must_exist = must_exist
        self.aggregate = aggregate
        self.database = database
        DropObject.__init__(self, must_exist=must_exist)

    def _object_name(self):
        return self.name

    def _get_scoped_name(self):
        if self.database:
            return '{0}.{1}'.format(self.database, self.name)
        else:
            return self.name

    def _ibis_string_to_impala(self, tval):
        if tval in _sql_type_names.keys():
            return _sql_type_names[tval]
        result = dt._parse_decimal(tval)
        if result:
            return 'decimal({0},{1})'.format(result.precision,
                                             result.scale)

    def compile(self):
        statement = 'DROP'
        if self.aggregate:
            statement += ' AGGREGATE'
        statement += ' FUNCTION'
        if not self.must_exist:
            statement += ' IF EXISTS'
        full_name = self._get_scoped_name()
        func_line = ' {0!s}({1!s})'.format(full_name, ', '.join(self.inputs))
        statement += func_line
        return statement


class ListFunction(DDLStatement):

    def __init__(self, database, like=None, aggregate=False):

        self.database = database
        self.like = like
        self.aggregate = aggregate

    def compile(self):
        statement = 'SHOW '
        if self.aggregate:
            statement += 'AGGREGATE '
        statement += 'FUNCTIONS IN {0}'.format(self.database)
        if self.like:
            statement += " LIKE '{0}'".format(self.like)
        return statement


_impala_type_names = {
    'int8': 'TINYINT',
    'int16': 'SMALLINT',
    'int32': 'INT',
    'int64': 'BIGINT',
    'float': 'FLOAT',
    'double': 'DOUBLE',
    'boolean': 'BOOLEAN',
    'timestamp': 'TIMESTAMP',
    'string': 'STRING'
}


def translate_expr(expr, context=None, named=False, permit_subquery=False):
    translator = ExprTranslator(expr, context=context, named=named,
                                permit_subquery=permit_subquery)
    return translator.get_result()
