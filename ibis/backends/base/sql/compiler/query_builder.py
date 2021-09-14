from __future__ import annotations

from io import StringIO

import toolz

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql.registry import quote_identifier
from ibis.config import options

from .base import DML, QueryAST, SetOp
from .select_builder import SelectBuilder
from .translator import ExprTranslator, QueryContext


class TableSetFormatter:

    _join_names = {
        ops.InnerJoin: 'INNER JOIN',
        ops.LeftJoin: 'LEFT OUTER JOIN',
        ops.RightJoin: 'RIGHT OUTER JOIN',
        ops.OuterJoin: 'FULL OUTER JOIN',
        ops.LeftAntiJoin: 'LEFT ANTI JOIN',
        ops.LeftSemiJoin: 'LEFT SEMI JOIN',
        ops.CrossJoin: 'CROSS JOIN',
    }

    def __init__(self, parent, expr, indent=2):
        # `parent` is a `Select` instance, not a `TableSetFormatter`
        self.parent = parent
        self.context = parent.context
        self.expr = expr
        self.indent = indent

        self.join_tables = []
        self.join_types = []
        self.join_predicates = []

    def _translate(self, expr):
        return self.parent._translate(expr)

    def _walk_join_tree(self, op):
        left = op.left.op()
        right = op.right.op()

        if util.all_of([left, right], ops.Join):
            raise NotImplementedError(
                'Do not support joins between ' 'joins yet'
            )

        self._validate_join_predicates(op.predicates)

        jname = self._get_join_type(op)

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
            raise NotImplementedError(
                'not allowing joins on right ' 'side yet'
            )
        else:
            # Both tables
            self.join_tables.append(self._format_table(op.left))
            self.join_tables.append(self._format_table(op.right))
            self.join_types.append(jname)
            self.join_predicates.append(op.predicates)

    # Placeholder; revisit when supporting other databases
    _non_equijoin_supported = True

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()

            if (
                not isinstance(op, ops.Equals)
                and not self._non_equijoin_supported
            ):
                raise com.TranslationError(
                    'Non-equality join predicates, '
                    'i.e. non-equijoins, are not '
                    'supported'
                )

    def _get_join_type(self, op):
        return self._join_names[type(op)]

    def _quote_identifier(self, name):
        return quote_identifier(name)

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
                raise com.RelationError(f'Table did not have a name: {expr!r}')
            result = self._quote_identifier(name)
            is_subquery = False
        else:
            # A subquery
            if ctx.is_extracted(ref_expr):
                # Was put elsewhere, e.g. WITH block, we just need to grab its
                # alias
                alias = ctx.get_ref(expr)

                # HACK: self-references have to be treated more carefully here
                if isinstance(op, ops.SelfReference):
                    return f'{ctx.get_ref(ref_expr)} {alias}'
                else:
                    return alias

            subquery = ctx.get_compiled_expr(expr)
            result = f'(\n{util.indent(subquery, self.indent)}\n)'
            is_subquery = True

        if is_subquery or ctx.need_aliases(expr):
            result += f' {ctx.get_ref(expr)}'

        return result

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
        for jtype, table, preds in zip(
            self.join_types, self.join_tables[1:], self.join_predicates
        ):
            buf.write('\n')
            buf.write(util.indent(f'{jtype} {table}', self.indent))

            fmt_preds = []
            npreds = len(preds)
            for pred in preds:
                new_pred = self._translate(pred)
                if npreds > 1:
                    new_pred = f'({new_pred})'
                fmt_preds.append(new_pred)

            if len(fmt_preds):
                buf.write('\n')

                conj = ' AND\n{}'.format(' ' * 3)
                fmt_preds = util.indent(
                    'ON ' + conj.join(fmt_preds), self.indent * 2
                )
                buf.write(fmt_preds)

        return buf.getvalue()


class Select(DML):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(
        self,
        table_set,
        select_set,
        translator_class,
        table_set_formatter_class,
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
        self.translator_class = translator_class
        self.table_set_formatter_class = table_set_formatter_class
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

    def _translate(self, expr, named=False, permit_subquery=False):
        translator = self.translator_class(
            expr,
            context=self.context,
            named=named,
            permit_subquery=permit_subquery,
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
            buf.append(f'{alias} AS (\n{formatted}\n)')

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
                    expr_str = f'{alias}.*' if alias else '*'
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

        return f'{select_key}{buf.getvalue()}'

    def format_table_set(self):
        if self.table_set is None:
            return None

        fragment = 'FROM '

        helper = self.table_set_formatter_class(self, self.table_set)
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
                new_pred = f'({new_pred})'
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
        buf.write(f'LIMIT {n}')
        if offset is not None and offset != 0:
            buf.write(f' OFFSET {offset}')

        return buf.getvalue()


class Union(SetOp):
    def __init__(self, tables, expr, context, distincts):
        super().__init__(tables, expr, context)
        self.distincts = distincts

    @staticmethod
    def keyword(distinct):
        return 'UNION' if distinct else 'UNION ALL'

    def _get_keyword_list(self):
        return map(self.keyword, self.distincts)


class Intersection(SetOp):
    def _get_keyword_list(self):
        return ["INTERSECT"] * (len(self.tables) - 1)


class Difference(SetOp):
    def _get_keyword_list(self):
        return ["EXCEPT"] * (len(self.tables) - 1)


def flatten_union(table: ir.TableExpr):
    """Extract all union queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr, bool]]
    """
    op = table.op()
    if isinstance(op, ops.Union):
        # For some reason mypy considers `op.left` and `op.right`
        # of `Argument` type, and fails the validation. While in
        # `flatten` types are the same, and it works
        return toolz.concatv(
            flatten_union(op.left),  # type: ignore
            [op.distinct],
            flatten_union(op.right),  # type: ignore
        )
    return [table]


def flatten(table: ir.TableExpr):
    """Extract all intersection or difference queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr]]
    """
    op = table.op()
    return list(toolz.concatv(flatten_union(op.left), flatten_union(op.right)))


class Compiler:
    translator_class = ExprTranslator
    context_class = QueryContext
    select_builder_class = SelectBuilder
    table_set_formatter_class = TableSetFormatter
    select_class = Select
    union_class = Union

    @classmethod
    def make_context(cls, params=None):
        params = params or {}
        params = {expr.op(): value for expr, value in params.items()}
        return cls.context_class(compiler=cls, params=params)

    @classmethod
    def to_ast(cls, expr, context=None):
        if context is None:
            context = cls.make_context()

        op = expr.op()

        # collect setup and teardown queries
        setup_queries = cls._generate_setup_queries(expr, context)
        teardown_queries = cls._generate_teardown_queries(expr, context)

        # TODO: any setup / teardown DDL statements will need to be done prior
        # to building the result set-generating statements.
        if isinstance(op, ops.Union):
            query = cls._make_union(cls.union_class, expr, context)
        elif isinstance(op, ops.Intersection):
            query = Intersection(flatten(expr), expr, context=context)
        elif isinstance(op, ops.Difference):
            query = Difference(flatten(expr), expr, context=context)
        else:
            query = cls.select_builder_class().to_select(
                select_class=cls.select_class,
                table_set_formatter_class=cls.table_set_formatter_class,
                expr=expr,
                context=context,
                translator_class=cls.translator_class,
            )

        return QueryAST(
            context,
            query,
            setup_queries=setup_queries,
            teardown_queries=teardown_queries,
        )

    @classmethod
    def to_ast_ensure_limit(cls, expr, limit=None, params=None):
        context = cls.make_context(params=params)
        query_ast = cls.to_ast(expr, context)

        # note: limit can still be None at this point, if the global
        # default_limit is None
        for query in reversed(query_ast.queries):
            if (
                isinstance(query, Select)
                and not isinstance(expr, ir.ScalarExpr)
                and query.table_set is not None
            ):
                if query.limit is None:
                    if limit == 'default':
                        query_limit = options.sql.default_limit
                    else:
                        query_limit = limit
                    if query_limit:
                        query.limit = {'n': query_limit, 'offset': 0}
                elif limit is not None and limit != 'default':
                    query.limit = {'n': limit, 'offset': query.limit['offset']}

        return query_ast

    @classmethod
    def to_sql(cls, expr, context=None, params=None):
        if context is None:
            context = cls.make_context(params=params)
        return cls.to_ast(expr, context).queries[0].compile()

    @staticmethod
    def _generate_setup_queries(expr, context):
        return []

    @staticmethod
    def _generate_teardown_queries(expr, context):
        return []

    @staticmethod
    def _make_union(union_class, expr, context):
        # flatten unions so that we can codegen them all at once
        union_info = list(flatten_union(expr))

        # since op is a union, we have at least 3 elements in union_info (left
        # distinct right) and if there is more than a single union we have an
        # additional two elements per union (distinct right) which means the
        # total number of elements is at least 3 + (2 * number of unions - 1)
        # and is therefore an odd number
        npieces = len(union_info)
        assert npieces >= 3 and npieces % 2 != 0, 'Invalid union expression'

        # 1. every other object starting from 0 is a TableExpr instance
        # 2. every other object starting from 1 is a bool indicating the type
        #    of union (distinct or not distinct)
        table_exprs, distincts = union_info[::2], union_info[1::2]
        return union_class(
            table_exprs, expr, distincts=distincts, context=context
        )
