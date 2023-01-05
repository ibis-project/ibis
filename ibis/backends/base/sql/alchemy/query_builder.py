from __future__ import annotations

import functools

import sqlalchemy as sa
from sqlalchemy import sql

import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy.database import AlchemyTable
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


class _AlchemyTableSetFormatter(TableSetFormatter):
    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.node

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(op))

        result = self.join_tables[0]
        for jtype, table, preds in zip(
            self.join_types, self.join_tables[1:], self.join_predicates
        ):
            if preds:
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
                # subquery is required for semi and anti joins done using
                # sqlalchemy, otherwise multiple references to the original
                # select are treated as distinct tables
                #
                # with a subquery, the result is a distinct table and so there's only one
                # thing for subsequent expressions to reference
                result = (
                    result.select()
                    .where(sa.exists(sa.select(1).where(onclause)))
                    .subquery()
                )
            elif jtype is ops.LeftAntiJoin:
                result = (
                    result.select()
                    .where(~sa.exists(sa.select(1).where(onclause)))
                    .subquery()
                )
            else:
                raise NotImplementedError(jtype)

        self.context.set_ref(op, result)
        return result

    def _get_join_type(self, op):
        return type(op)

    def _format_table(self, op):
        ctx = self.context
        ref_op = op

        if isinstance(op, ops.SelfReference):
            ref_op = op.table

        alias = ctx.get_ref(op)

        translator = ctx.compiler.translator_class(ref_op, ctx)

        if isinstance(ref_op, AlchemyTable):
            result = ref_op.sqla_table
        elif isinstance(ref_op, ops.UnboundTable):
            # use SQLAlchemy's TableClause for unbound tables
            result = sa.table(
                ref_op.name, *translator._schema_to_sqlalchemy_columns(ref_op.schema)
            )
        elif isinstance(ref_op, ops.SQLQueryResult):
            columns = translator._schema_to_sqlalchemy_columns(ref_op.schema)
            result = sa.text(ref_op.query).columns(*columns)
        elif isinstance(ref_op, ops.SQLStringView):
            columns = translator._schema_to_sqlalchemy_columns(ref_op.schema)
            result = sa.text(ref_op.query).columns(*columns).cte(ref_op.name)
        elif isinstance(ref_op, ops.View):
            # TODO(kszucs): avoid converting to expression
            child_expr = ref_op.child.to_expr()
            definition = child_expr.compile()
            result = sa.table(
                ref_op.name, *translator._schema_to_sqlalchemy_columns(ref_op.schema)
            )
            backend = child_expr._find_backend()
            backend._create_temp_view(view=result, definition=definition)
        elif isinstance(ref_op, ops.InMemoryTable):
            columns = translator._schema_to_sqlalchemy_columns(ref_op.schema)

            if self.context.compiler.cheap_in_memory_tables:
                result = sa.table(ref_op.name, *columns)
            else:
                # this has horrendous performance for medium to large tables
                # should we warn?
                rows = list(ref_op.data.to_frame().itertuples(index=False))
                result = sa.values(*columns, name=ref_op.name).data(rows)
        else:
            # A subquery
            if ctx.is_extracted(ref_op):
                # Was put elsewhere, e.g. WITH block, we just need to grab
                # its alias
                alias = ctx.get_ref(op)

                # hack
                if isinstance(op, ops.SelfReference):
                    table = ctx.get_ref(ref_op)
                    self_ref = alias if hasattr(alias, "name") else table.alias(alias)
                    ctx.set_ref(op, self_ref)
                    return self_ref
                return alias

            alias = ctx.get_ref(op)
            result = ctx.get_compiled_expr(op)

        result = alias if hasattr(alias, "name") else result.alias(alias)
        ctx.set_ref(op, result)
        return result


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
            self._add_group_by,
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
        for op in self.select_set:
            if isinstance(op, ops.Value):
                arg = self._translate(op, named=True)
            elif isinstance(op, ops.TableNode):
                if op.equals(self.table_set):
                    cached_table = self.context.get_ref(op)
                    if cached_table is None:
                        has_select_star = True
                        continue
                    else:
                        arg = table_set
                else:
                    arg = self.context.get_ref(op)
                    if arg is None:
                        raise ValueError(op)
            else:
                raise TypeError(op)

            to_select.append(arg)

        if has_select_star:
            if table_set is None:
                raise ValueError('table_set cannot be None here')

            clauses = [table_set] + to_select
        else:
            clauses = to_select

        result_func = sa.exists if self.exists else sa.select
        result = result_func(*clauses)

        if self.distinct:
            result = result.distinct()

        # if we're SELECT *-ing or there's no table_set (e.g., SELECT 1) then
        # we can return early
        if has_select_star or table_set is None:
            return result

        return result.select_from(table_set)

    def _add_group_by(self, fragment):
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

        args = [self._translate(pred, permit_subquery=True) for pred in self.where]
        clause = functools.reduce(sql.and_, args)
        return fragment.where(clause)

    def _add_order_by(self, fragment):
        if not len(self.order_by):
            return fragment

        clauses = []
        for key in self.order_by:
            sort_expr = key.expr
            arg = self._translate(sort_expr)
            fn = sa.asc if key.ascending else sa.desc

            clauses.append(fn(arg))

        return fragment.order_by(*clauses)

    def _among_select_set(self, expr):
        return any(expr.equals(other) for other in self.select_set)

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
                self.distinct_func(*args) if distinct else self.non_distinct_func(*args)
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
