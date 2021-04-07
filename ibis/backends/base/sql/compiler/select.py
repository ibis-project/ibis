from io import StringIO

import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util

from . import base, table_set_formatter


class Select(base.DML):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(
        self,
        table_set,
        select_set,
        context,
        subqueries=None,
        where=None,
        group_by=None,
        having=None,
        order_by=None,
        limit=None,
        distinct=False,
        indent=2,
        result_handler=None,
        parent_expr=None,
    ):
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

    @property
    def translator(self):
        return self.context.dialect.translator

    def _translate(self, expr, named=False, permit_subquery=False):
        context = self.context
        translator = self.translator(
            expr, context=context, named=named, permit_subquery=permit_subquery
        )
        return translator.get_result()

    def equals(self, other, cache=None):
        if cache is None:
            cache = {}

        key = self, other

        try:
            return cache[key]
        except KeyError:
            cache[key] = result = self is other or (
                isinstance(other, Select)
                and self.limit == other.limit
                and ops.all_equal(self._all_exprs(), other._all_exprs())
            )
            return result

    def _all_exprs(self):
        # Gnarly, maybe we can improve this somehow
        expr_attrs = (
            'select_set',
            'table_set',
            'where',
            'group_by',
            'having',
            'order_by',
            'subqueries',
        )
        exprs = []
        for attr in expr_attrs:
            val = getattr(self, attr)
            if isinstance(val, list):
                exprs.extend(val)
            else:
                exprs.append(val)

        return exprs

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

        # ORDER BY
        order_frag = self.format_order_by()

        # LIMIT
        limit_frag = self.format_limit()

        # Glue together the query fragments and return
        query = '\n'.join(
            filter(
                None,
                [
                    with_frag,
                    select_frag,
                    from_frag,
                    where_frag,
                    groupby_frag,
                    order_frag,
                    limit_frag,
                ],
            )
        )
        return query

    def format_subqueries(self):
        if not self.subqueries:
            return

        context = self.context

        buf = []

        for i, expr in enumerate(self.subqueries):
            formatted = util.indent(context.get_compiled_expr(expr), 2)
            alias = context.get_ref(expr)
            buf.append('{} AS (\n{}\n)'.format(alias, formatted))

        return 'WITH {}'.format(',\n'.join(buf))

    def format_select_set(self):
        # TODO:
        context = self.context
        formatted = []
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                expr_str = self._translate(expr, named=True)
            elif isinstance(expr, ir.TableExpr):
                # A * selection, possibly prefixed
                if context.need_aliases(expr):
                    alias = context.get_ref(expr)

                    # materialized join will not have an alias. see #491
                    expr_str = '{}.*'.format(alias) if alias else '*'
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
            elif (
                tokens > 0
                and line_length
                and len(val) + line_length > max_length
            ):
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

    @property
    def table_set_formatter(self):
        return table_set_formatter.TableSetFormatter

    def format_table_set(self):
        if self.table_set is None:
            return None

        fragment = 'FROM '

        helper = self.table_set_formatter(self, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_group_by(self):
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            clause = 'GROUP BY {}'.format(
                ', '.join([str(x + 1) for x in self.group_by])
            )
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_where(self):
        if not self.where:
            return None

        buf = StringIO()
        buf.write('WHERE ')
        fmt_preds = []
        npreds = len(self.where)
        for pred in self.where:
            new_pred = self._translate(pred, permit_subquery=True)
            if npreds > 1:
                new_pred = '({})'.format(new_pred)
            fmt_preds.append(new_pred)

        conj = ' AND\n{}'.format(' ' * 6)
        buf.write(conj.join(fmt_preds))
        return buf.getvalue()

    def format_order_by(self):
        if not self.order_by:
            return None

        buf = StringIO()
        buf.write('ORDER BY ')

        formatted = []
        for expr in self.order_by:
            key = expr.op()
            translated = self._translate(key.expr)
            if not key.ascending:
                translated += ' DESC'
            formatted.append(translated)

        buf.write(', '.join(formatted))
        return buf.getvalue()

    def format_limit(self):
        if not self.limit:
            return None

        buf = StringIO()

        n, offset = self.limit['n'], self.limit['offset']
        buf.write('LIMIT {}'.format(n))
        if offset is not None and offset != 0:
            buf.write(' OFFSET {}'.format(offset))

        return buf.getvalue()
