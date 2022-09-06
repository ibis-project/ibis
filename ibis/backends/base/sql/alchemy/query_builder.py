from __future__ import annotations

import functools

import sqlalchemy as sa
import sqlalchemy.sql as sql

import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql.alchemy.database import AlchemyTable
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type
from ibis.backends.base.sql.alchemy.translator import (
    AlchemyContext,
    AlchemyExprTranslator,
)
from ibis.backends.base.sql.compiler import (
    Compiler,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.base.sql.compiler.base import SetOp


def _schema_to_sqlalchemy_columns(schema: sch.Schema) -> list[sa.Column]:
    return [sa.column(n, to_sqla_type(t)) for n, t in schema.items()]


class _AlchemyTableSetFormatter(TableSetFormatter):
    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        result = self.join_tables[0]
        for jtype, table, preds in zip(
            self.join_types, self.join_tables[1:], self.join_predicates
        ):
            if len(preds):
                sqla_preds = [self._translate(pred) for pred in preds]
                onclause = functools.reduce(sql.and_, sqla_preds)
            else:
                onclause = None

            if jtype is ops.InnerJoin:
                result = result.join(table, onclause)
            elif jtype is ops.CrossJoin:
                result = result.join(table, sa.literal(True))
            elif jtype is ops.LeftJoin:
                result = result.join(table, onclause, isouter=True)
            elif jtype is ops.RightJoin:
                result = table.join(result, onclause, isouter=True)
            elif jtype is ops.OuterJoin:
                result = result.outerjoin(table, onclause, full=True)
            elif jtype is ops.LeftSemiJoin:
                result = result.select().where(
                    sa.exists(sa.select(1).where(onclause))
                )
            elif jtype is ops.LeftAntiJoin:
                result = result.select().where(
                    ~sa.exists(sa.select(1).where(onclause))
                )
            else:
                raise NotImplementedError(jtype)

        self.context.set_ref(self.expr, result)
        return result

    def _get_join_type(self, op):
        return type(op)

    def _format_table(self, expr):
        ctx = self.context
        ref_expr = expr
        op = ref_op = expr.op()

        if isinstance(op, ops.SelfReference):
            ref_expr = op.table
            ref_op = ref_expr.op()

        alias = ctx.get_ref(expr)

        if isinstance(ref_op, AlchemyTable):
            result = ref_op.sqla_table
        elif isinstance(ref_op, ops.UnboundTable):
            # use SQLAlchemy's TableClause for unbound tables
            result = sa.table(
                ref_op.name,
                *_schema_to_sqlalchemy_columns(ref_op.schema),
            )
        elif isinstance(ref_op, ops.SQLQueryResult):
            columns = _schema_to_sqlalchemy_columns(ref_op.schema)
            result = sa.text(ref_op.query).columns(*columns)
        elif isinstance(ref_op, ops.SQLStringView):
            columns = _schema_to_sqlalchemy_columns(ref_op.schema)
            result = sa.text(ref_op.query).columns(*columns).cte(ref_op.name)
        elif isinstance(ref_op, ops.View):
            definition = ref_op.child.compile()
            result = sa.table(
                ref_op.name,
                *_schema_to_sqlalchemy_columns(ref_op.schema),
            )
            backend = ref_op.child._find_backend()
            backend._create_temp_view(view=result, definition=definition)
        elif isinstance(ref_op, ops.InMemoryTable):
            columns = _schema_to_sqlalchemy_columns(ref_op.schema)

            if self.context.compiler.cheap_in_memory_tables:
                result = sa.table(ref_op.name, *columns)
            else:
                # this has horrendous performance for medium to large tables
                # should we warn?
                rows = list(ref_op.data.to_frame().itertuples(index=False))
                result = sa.values(*columns).data(rows)
        else:
            # A subquery
            if ctx.is_extracted(ref_expr):
                # Was put elsewhere, e.g. WITH block, we just need to grab
                # its alias
                alias = ctx.get_ref(expr)

                # hack
                if isinstance(op, ops.SelfReference):
                    table = ctx.get_ref(ref_expr)
                    self_ref = (
                        alias if hasattr(alias, "name") else table.alias(alias)
                    )
                    ctx.set_ref(expr, self_ref)
                    return self_ref
                return alias

            alias = ctx.get_ref(expr)
            result = ctx.get_compiled_expr(expr)

        result = alias if hasattr(alias, "name") else result.alias(alias)
        ctx.set_ref(expr, result)
        return result


def _can_lower_sort_column(table_set, expr):
    # TODO(wesm): This code is pending removal through cleaner internal
    # semantics

    # we can currently sort by just-appeared aggregate metrics, but the way
    # these are references in the expression DSL is as a SortBy (blocking
    # table operation) on an aggregation. There's a hack in _collect_SortBy
    # in the generic SQL compiler that "fuses" the sort with the
    # aggregation so they appear in same query. It's generally for
    # cosmetics and doesn't really affect query semantics.
    bases = {op: op.to_expr() for op in expr.op().root_tables()}
    if len(bases) != 1:
        return False

    base = list(bases.values())[0]
    base_op = base.op()

    if isinstance(base_op, ops.Aggregation):
        return base_op.table.equals(table_set)
    elif isinstance(base_op, ops.Selection):
        return base.equals(table_set)
    else:
        return False


class AlchemySelect(Select):
    def __init__(self, *args, **kwargs):
        self.exists = kwargs.pop('exists', False)
        super().__init__(*args, **kwargs)

    def compile(self):
        # Can't tell if this is a hack or not. Revisit later
        self.context.set_query(self)

        self._compile_subqueries()

        frag = self._compile_table_set()
        steps = [
            self._add_select,
            self._add_groupby,
            self._add_where,
            self._add_order_by,
            self._add_limit,
        ]

        for step in steps:
            frag = step(frag)

        return frag

    def _compile_subqueries(self):
        if not self.subqueries:
            return

        for expr in self.subqueries:
            result = self.context.get_compiled_expr(expr)
            alias = self.context.get_ref(expr)
            result = result.cte(alias)
            self.context.set_ref(expr, result)

    def _compile_table_set(self):
        if self.table_set is not None:
            helper = self.table_set_formatter_class(self, self.table_set)
            result = helper.get_result()
            if isinstance(result, sql.selectable.Select):
                return result.subquery()
            return result
        else:
            return None

    def _add_select(self, table_set):
        to_select = []

        has_select_star = False
        for expr in self.select_set:
            if isinstance(expr, ir.Value):
                arg = self._translate(expr, named=True)
            elif isinstance(expr, ir.Table):
                if expr.equals(self.table_set):
                    cached_table = self.context.get_ref(expr)
                    if cached_table is None:
                        has_select_star = True
                        continue
                    else:
                        arg = table_set
                else:
                    arg = self.context.get_ref(expr)
                    if arg is None:
                        raise ValueError(expr)

            to_select.append(arg)

        if has_select_star:
            if table_set is None:
                raise ValueError('table_set cannot be None here')

            clauses = [table_set] + to_select
        else:
            clauses = to_select

        if self.exists:
            result = sa.exists(clauses)
        else:
            result = sa.select(clauses)

        if self.distinct:
            result = result.distinct()

        # if we're SELECT *-ing or there's no table_set (e.g., SELECT 1) then
        # we can return early
        if has_select_star or table_set is None:
            return result

        # if we're selecting from something that isn't a subquery e.g., Select,
        # Alias, Table
        if not isinstance(table_set, sa.sql.Subquery):
            return result.select_from(table_set)

        final_froms = result.get_final_froms()
        num_froms = len(final_froms)

        # if the result subquery has no FROMs then we can select from the
        # table_set since there's only a single possibility for FROM
        if not num_froms:
            return result.select_from(table_set)

        # otherwise we expect a single FROM in `result`
        assert num_froms == 1, f"num_froms == {num_froms:d}"

        # and then we need to replace every occurrence of `result`'s `FROM`
        # with `table_set` to handle correlated EXISTs coming from
        # semi/anti-join
        #
        # previously this was `replace_selectable`, but that's deprecated so we
        # inline its implementation here
        #
        # sqlalchemy suggests using the functionality in sa.sql.visitors, but
        # that would effectively require reimplementing ClauseAdapter
        return sa.sql.util.ClauseAdapter(table_set).traverse(result)

    def _add_groupby(self, fragment):
        # GROUP BY and HAVING
        if not len(self.group_by):
            return fragment

        group_keys = [self._translate(arg) for arg in self.group_by]
        fragment = fragment.group_by(*group_keys)

        if len(self.having) > 0:
            having_args = [self._translate(arg) for arg in self.having]
            having_clause = functools.reduce(sql.and_, having_args)
            fragment = fragment.having(having_clause)

        return fragment

    def _add_where(self, fragment):
        if not len(self.where):
            return fragment

        args = [
            self._translate(pred, permit_subquery=True) for pred in self.where
        ]
        clause = functools.reduce(sql.and_, args)
        return fragment.where(clause)

    def _add_order_by(self, fragment):
        if not len(self.order_by):
            return fragment

        clauses = []
        for expr in self.order_by:
            key = expr.op()
            sort_expr = key.expr

            # here we have to determine if key.expr is in the select set (as it
            # will be in the case of order_by fused with an aggregation
            if _can_lower_sort_column(self.table_set, sort_expr):
                arg = sort_expr.get_name()
            else:
                arg = self._translate(sort_expr)

            if not key.ascending:
                arg = sa.desc(arg)

            clauses.append(arg)

        return fragment.order_by(*clauses)

    def _among_select_set(self, expr):
        for other in self.select_set:
            if expr.equals(other):
                return True
        return False

    def _add_limit(self, fragment):
        if self.limit is None:
            return fragment

        fragment = fragment.limit(self.limit.n)
        if offset := self.limit.offset:
            fragment = fragment.offset(offset)
        return fragment


class AlchemySelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class AlchemySetOp(SetOp):
    def compile(self):
        context = self.context
        selects = []

        def call(distinct, *args):
            return (
                self.distinct_func(*args)
                if distinct
                else self.non_distinct_func(*args)
            )

        for table in self.tables:
            table_set = context.get_compiled_expr(table)
            selects.append(table_set.cte().select())

        if len(set(self.distincts)) == 1:
            # distinct is either all True or all False, handle with a single
            # call. This generates much more concise SQL.
            return call(self.distincts[0], *selects)
        else:
            # We need to iteratively apply the set operations to handle
            # disparate `distinct` values. Subqueries _must_ be converted using
            # `.subquery().select()` to get sqlalchemy to put parenthesis in
            # the proper places.
            result = selects[0]
            for select, distinct in zip(selects[1:], self.distincts):
                result = call(distinct, result.subquery().select(), select)
            return result


class AlchemyUnion(AlchemySetOp):
    distinct_func = staticmethod(sa.union)
    non_distinct_func = staticmethod(sa.union_all)


class AlchemyIntersection(AlchemySetOp):
    distinct_func = staticmethod(sa.intersect)
    non_distinct_func = staticmethod(sa.intersect_all)


class AlchemyDifference(AlchemySetOp):
    distinct_func = staticmethod(sa.except_)
    non_distinct_func = staticmethod(sa.except_all)


class AlchemyCompiler(Compiler):
    translator_class = AlchemyExprTranslator
    context_class = AlchemyContext
    table_set_formatter_class = _AlchemyTableSetFormatter
    select_builder_class = AlchemySelectBuilder
    select_class = AlchemySelect
    union_class = AlchemyUnion
    intersect_class = AlchemyIntersection
    difference_class = AlchemyDifference

    @classmethod
    def to_sql(cls, expr, context=None, params=None, exists=False):
        if context is None:
            context = cls.make_context(params=params)
        query = cls.to_ast(expr, context).queries[0]
        if exists:
            query.exists = True
        return query.compile()
