from six import StringIO

# import ibis
# import ibis.expr.analysis as L
# import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
# import ibis.expr.temporal as tempo

import ibis.sql.compiler as comp
# import ibis.sql.transforms as transforms

# TODO: create absolute import
from .identifiers import quote_identifier
# TODO: remove this import
from .operations import _operation_registry, _name_expr

import ibis.common as com
import ibis.util as util


def build_ast(expr, context=None, params=None):
    builder = ClickhouseQueryBuilder(expr, context=context, params=params)
    return builder.get_result()


def _get_query(expr, context):
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    query = _get_query(expr, context)
    return query.compile()


# ----------------------------------------------------------------------
# Select compilation

class ClickhouseSelectBuilder(comp.SelectBuilder):

    @property
    def _select_class(self):
        return ClickhouseSelect

    def _convert_group_by(self, exprs):
        return exprs


class ClickhouseQueryBuilder(comp.QueryBuilder):

    select_builder = ClickhouseSelectBuilder

    @property
    def _make_context(self):
        return ClickhouseQueryContext

    @property
    def _union_class(self):
        return ClickhouseUnion


class ClickhouseQueryContext(comp.QueryContext):

    def _to_sql(self, expr, ctx):
        return to_sql(expr, context=ctx)


class ClickhouseSelect(comp.Select):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def compile(self):
        """
        This method isn't yet idempotent; calling multiple times may yield
        unexpected results
        """
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        # If any subqueries, translate them and add to beginning of query as
        # part of the WITH section
        with_frag = self.format_subqueries()

        # SELECT
        select_frag = self.format_select_set()

        # FROM, JOIN, UNION
        from_frag = self.format_table_set()

        # WHERE
        where_frag = self.format_where()

        # GROUP BY and HAVING
        groupby_frag = self.format_group_by()

        # ORDER BY and LIMIT
        order_frag = self.format_postamble()

        # Glue together the query fragments and return
        query = _join_not_none('\n', [with_frag, select_frag, from_frag,
                                      where_frag, groupby_frag, order_frag])

        return query

    def format_subqueries(self):
        if len(self.subqueries) == 0:
            return

        context = self.context

        buf = StringIO()
        buf.write('WITH ')

        for i, expr in enumerate(self.subqueries):
            if i > 0:
                buf.write(',\n')
            formatted = util.indent(context.get_compiled_expr(expr), 2)
            alias = context.get_ref(expr)
            buf.write('{0} AS (\n{1}\n)'.format(alias, formatted))

        return buf.getvalue()

    def format_select_set(self):
        # TODO:
        context = self.context
        formatted = []
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                expr_str = self._translate(expr, named=True)
            elif isinstance(expr, ir.TableExpr):
                # A * selection, possibly prefixed
                if context.need_aliases():
                    alias = context.get_ref(expr)

                    # materialized join will not have an alias. see #491
                    expr_str = '{0}.*'.format(alias) if alias else '*'
                else:
                    expr_str = '*'
            formatted.append(expr_str)

        buf = StringIO()
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

    def format_table_set(self):
        if self.table_set is None:
            return None

        fragment = 'FROM '

        helper = _TableSetFormatter(self, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_group_by(self):
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            columns = ['`{0}`'.format(expr.get_name())
                       for expr in self.group_by]
            clause = 'GROUP BY {0}'.format(', '.join(columns))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {0}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_where(self):
        if len(self.where) == 0:
            return None

        buf = StringIO()
        buf.write('WHERE ')
        fmt_preds = []
        for pred in self.where:
            new_pred = self._translate(pred, permit_subquery=True)
            if isinstance(pred.op(), ops.Or):
                # parens for OR exprs because it binds looser than AND
                new_pred = '({0!s})'.format(new_pred)
            fmt_preds.append(new_pred)

        conj = ' AND\n{0}'.format(' ' * 6)
        buf.write(conj.join(fmt_preds))
        return buf.getvalue()

    def format_postamble(self):
        buf = StringIO()
        lines = 0

        if len(self.order_by) > 0:
            buf.write('ORDER BY ')
            formatted = []
            for expr in self.order_by:
                key = expr.op()
                translated = self._translate(key.expr)
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

    @property
    def translator(self):
        return ClickhouseExprTranslator


def _join_not_none(sep, pieces):
    pieces = [x for x in pieces if x is not None]
    return sep.join(pieces)


class _TableSetFormatter(comp.TableSetFormatter):

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
        buf = StringIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(self.join_types, self.join_tables[1:],
                                       self.join_predicates):
            buf.write('\n')
            buf.write(util.indent('{0} {1}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = map(self._format_predicate, preds)
                # fmt_preds = [self._translate(pred) for pred in preds]
                fmt_preds = util.indent('USING ' + ', '.join(fmt_preds),
                                        self.indent * 2)
                buf.write(fmt_preds)

        return buf.getvalue()

    _join_names = {
        ops.InnerJoin: 'ALL INNER JOIN',
        ops.LeftJoin: 'ALL LEFT JOIN',
        ops.InnerSemiJoin: 'ANY INNER JOIN',
        ops.LeftSemiJoin: 'ANY LEFT JOIN'
    }

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()
            if not isinstance(op, ops.Equals):
                raise com.TranslationError('Non-equality join predicates are '
                                           'not supported')

            left_on, right_on = op.args
            if left_on.get_name() != right_on.get_name():
                raise com.TranslationError('Joining on different column names '
                                           'is not supported')

    def _get_join_type(self, op):
        return self._join_names[type(op)]

    def _format_predicate(self, predicate):
        column = predicate.op().args[0]
        return quote_identifier(column.get_name(), force=True)

    def _format_table(self, expr):
        # TODO: This could probably go in a class and be significantly nicer
        ctx = self.context

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
                alias = ctx.get_ref(expr)

                # HACK: self-references have to be treated more carefully here
                if isinstance(op, ops.SelfReference):
                    return '{0} {1}'.format(ctx.get_ref(ref_expr), alias)
                else:
                    return alias

            subquery = ctx.get_compiled_expr(expr)
            result = '(\n{0}\n)'.format(util.indent(subquery, self.indent))
            is_subquery = True

        if is_subquery or ctx.need_aliases():
            result += ' {0}'.format(ctx.get_ref(expr))

        return result


class ClickhouseUnion(comp.Union):

    def _extract_subqueries(self):
        self.subqueries = comp._extract_subqueries(self)
        for subquery in self.subqueries:
            self.context.set_extracted(subquery)

    def format_subqueries(self):
        context = self.context
        subqueries = self.subqueries

        return ',\n'.join([
            '{0} AS (\n{1}\n)'.format(
                context.get_ref(expr),
                util.indent(context.get_compiled_expr(expr), 2)
            ) for expr in subqueries
        ])

    def format_relation(self, expr):
        ref = self.context.get_ref(expr)
        if ref is not None:
            return 'SELECT *\nFROM {0}'.format(ref)
        return self.context.get_compiled_expr(expr)

    def compile(self):
        union_keyword = 'UNION' if self.distinct else 'UNION ALL'

        self._extract_subqueries()

        left_set = self.format_relation(self.left)
        right_set = self.format_relation(self.right)
        extracted = self.format_subqueries()

        buf = []

        if extracted:
            buf.append('WITH {0}'.format(extracted))

        buf.extend([left_set, union_keyword, right_set])

        return '\n'.join(buf)


class ClickhouseExprTranslator(comp.ExprTranslator):

    _registry = _operation_registry
    _context_class = ClickhouseQueryContext

    def name(self, translated, name, force=True):
        return _name_expr(translated,
                          quote_identifier(name, force=force))


compiles = ClickhouseExprTranslator.compiles
rewrites = ClickhouseExprTranslator.rewrites


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()
