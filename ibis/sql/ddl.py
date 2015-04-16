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

from ibis.sql.exprs import ExprTranslator, quote_identifier
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.common as com
import ibis.util as util


class DDLStatement(object):

    def _get_scoped_name(self, table_name, database):
        if database:
            scoped_name = '{}.{}'.format(database, table_name)
        else:
            scoped_name = table_name
        return scoped_name


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
            buf.write('{} AS (\n{}\n)'.format(alias, formatted))

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
                    expr_str = '{}.*'.format(context.get_alias(expr))
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

        return '{}{}'.format(select_key, buf.getvalue())

    def format_table_set(self, ctx):
        fragment = 'FROM '

        helper = _TableSetFormatter(ctx, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_group_by(self, context):
        if len(self.group_by) == 0:
            # There is no aggregation, nothing to see here
            return None

        # Verify that the group by exprs match the first few tokens in the
        # select set
        for i, expr in enumerate(self.group_by):
            if expr is not self.select_set[i]:
                raise com.InternalError('Select was improperly formed')

        lines = []
        if len(self.group_by) > 0:
            clause = 'GROUP BY {}'.format(', '.join([
                str(x + 1) for x in range(len(self.group_by))]))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = translate_expr(expr, context=context)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_where(self, context):
        if len(self.where) == 0:
            return None

        buf = BytesIO()
        buf.write('WHERE ')
        fmt_preds = [translate_expr(pred, context=context,
                                    permit_subquery=True)
                     for pred in self.where]
        conj = ' AND\n{}'.format(' ' * 6)
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
            buf.write('LIMIT {}'.format(n))
            if offset is not None and offset != 0:
                buf.write(' OFFSET {}'.format(offset))
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
            buf.write(util.indent('{} {}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = [translate_expr(pred, context=self.context)
                             for pred in preds]
                conj = ' AND\n{}'.format(' ' * 3)
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
    _non_equijoin_supported = False

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
        name = op.name
        if name is None:
            raise com.RelationError('Table did not have a name: {!r}'
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
                return '{} {}'.format(ctx.get_alias(ref_expr), alias)
            else:
                return alias

        subquery = ctx.get_formatted_query(expr)
        result = '(\n{}\n)'.format(util.indent(subquery, indent))
        is_subquery = True

    if is_subquery or ctx.need_aliases():
        result += ' {}'.format(ctx.get_alias(expr))

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

        query = '{}\n{}\n{}'.format(left_set, union_keyword, right_set)
        return query


class CTAS(DDLStatement):

    """
    Create Table As Select
    """

    def __init__(self, table_name, select, context, database=None,
                 external=False, format='parquet', overwrite=False,
                 partition=None):
        self.select = select
        self.context = context

        self.table_name = table_name
        self.database = database
        self.external = external
        self.overwrite = overwrite
        self.format = format.lower()

    def compile(self):
        if_exists = '' if self.overwrite else 'IF NOT EXISTS '

        buf = BytesIO()

        scoped_name = self._get_scoped_name(self.table_name, self.database)
        create_line = 'CREATE TABLE {}{}'.format(if_exists, scoped_name)

        buf.write(create_line)
        buf.write(self._storage())

        select_query = self.select.compile()
        buf.write('\nAS\n{}'.format(select_query))

        return buf.getvalue()

    def _storage(self):
        if self.format == 'parquet':
            return '\nSTORED AS PARQUET'
        else:
            raise NotImplementedError


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
        return'{} {}\n{}'.format(cmd, scoped_name, select_query)


class DropTable(DDLStatement):

    def __init__(self, table_name, database=None, must_exist=True):
        self.table_name = table_name
        self.database = database
        self.must_exist = must_exist

    def compile(self):
        if_exists = '' if self.must_exist else 'IF EXISTS '
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        drop_line = 'DROP TABLE {}{}'.format(if_exists, scoped_name)
        return drop_line


class CacheTable(DDLStatement):

    def __init__(self, table_name, database=None, pool='default'):
        self.table_name = table_name
        self.database = database
        self.pool = pool

    def compile(self):
        scoped_name = self._get_scoped_name(self.table_name, self.database)
        cache_line = 'ALTER TABLE {} SET CACHED IN \'{}\''.format(scoped_name, self.pool)
        return cache_line


def _join_not_none(sep, pieces):
    pieces = [x for x in pieces if x is not None]
    return sep.join(pieces)


def translate_expr(expr, context=None, named=False, permit_subquery=False):
    translator = ExprTranslator(expr, context=context, named=named,
                                permit_subquery=permit_subquery)
    return translator.get_result()
