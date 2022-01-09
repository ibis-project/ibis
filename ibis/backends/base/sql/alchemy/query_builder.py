import functools

import sqlalchemy as sa
import sqlalchemy.sql as sql

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.compiler import (
    Compiler,
    Select,
    SelectBuilder,
    TableSetFormatter,
    Union,
)

from .database import AlchemyTable
from .datatypes import to_sqla_type
from .translator import AlchemyContext, AlchemyExprTranslator


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

            if jtype in (ops.InnerJoin, ops.CrossJoin):
                result = result.join(table, onclause)
            elif jtype is ops.LeftJoin:
                result = result.join(table, onclause, isouter=True)
            elif jtype is ops.RightJoin:
                result = table.join(result, onclause, isouter=True)
            elif jtype is ops.OuterJoin:
                result = result.outerjoin(table, onclause, full=True)
            elif jtype is ops.LeftSemiJoin:
                result = sa.select([result]).where(
                    sa.exists(sa.select([1]).where(onclause))
                )
            elif jtype is ops.LeftAntiJoin:
                result = sa.select([result]).where(
                    ~(sa.exists(sa.select([1]).where(onclause)))
                )
            else:
                raise NotImplementedError(jtype)

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
            # use SQLAlchemy's TableClause and ColumnClause for unbound tables
            schema = ref_op.schema
            result = sa.table(
                ref_op.name,
                *(sa.column(n, to_sqla_type(t)) for n, t in schema.items()),
            )
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
    bases = ops.find_all_base_tables(expr)
    if len(bases) > 1:
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
            helper = _AlchemyTableSetFormatter(self, self.table_set)
            result = helper.get_result()
            if isinstance(result, sql.selectable.Select) and hasattr(
                result, 'subquery'
            ):
                return result.subquery()
            return result
        else:
            return None

    def _add_select(self, table_set):
        to_select = []

        has_select_star = False
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                arg = self._translate(expr, named=True)
            elif isinstance(expr, ir.TableExpr):
                if expr.equals(self.table_set):
                    cached_table = self.context.get_ref(expr)
                    if cached_table is None:
                        # the select * case from materialized join
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

        if not has_select_star:
            if table_set is not None:
                return result.select_from(table_set)
            else:
                return result
        else:
            return result

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

        n, offset = self.limit['n'], self.limit['offset']
        fragment = fragment.limit(n)
        if offset is not None and offset != 0:
            fragment = fragment.offset(offset)

        return fragment


class AlchemySelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class AlchemyUnion(Union):
    def compile(self):
        def reduce_union(left, right, distincts=iter(self.distincts)):
            distinct = next(distincts)
            sa_func = sa.union if distinct else sa.union_all
            return sa_func(left, right)

        context = self.context
        selects = []

        for table in self.tables:
            table_set = context.get_compiled_expr(table)
            selects.append(table_set.cte().select())

        return functools.reduce(reduce_union, selects)


class AlchemyCompiler(Compiler):
    translator_class = AlchemyExprTranslator
    context_class = AlchemyContext
    table_set_formatter_class = _AlchemyTableSetFormatter
    select_builder_class = AlchemySelectBuilder
    select_class = AlchemySelect
    union_class = AlchemyUnion

    @classmethod
    def to_sql(cls, expr, context=None, params=None, exists=False):
        if context is None:
            context = cls.make_context(params=params)
        query = cls.to_ast(expr, context).queries[0]
        if exists:
            query.exists = True
        return query.compile()
