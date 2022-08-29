from __future__ import annotations

import functools
import operator
from typing import NamedTuple

import toolz

import ibis
import ibis.common.exceptions as com
import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base.sql.compiler.base import (
    _extract_common_table_expressions,
)


class _LimitSpec(NamedTuple):
    n: int
    offset: int


class _CorrelatedRefCheck:
    def __init__(self, query, expr):
        self.query = query
        self.ctx = query.context
        self.expr = expr
        self.query_roots = frozenset(self.query.table_set.op().root_tables())
        self.has_foreign_root = False
        self.has_query_root = False

    def get_result(self):
        self.visit(self.expr)
        return self.has_query_root and self.has_foreign_root

    def visit(
        self, expr, in_subquery=False, visit_cache=None, visit_table_cache=None
    ):
        if visit_cache is None:
            visit_cache = set()

        node = expr.op()
        key = node, in_subquery
        if key in visit_cache:
            return

        visit_cache.add(key)

        in_subquery |= self.is_subquery(node)

        for arg in node.flat_args():
            if isinstance(arg, ir.Table):
                self.visit_table(
                    arg,
                    in_subquery=in_subquery,
                    visit_cache=visit_cache,
                    visit_table_cache=visit_table_cache,
                )
            elif isinstance(arg, ir.Expr):
                self.visit(
                    arg,
                    in_subquery=in_subquery,
                    visit_cache=visit_cache,
                    visit_table_cache=visit_table_cache,
                )

    def is_subquery(self, node):
        return isinstance(
            node,
            (
                ops.TableArrayView,
                ops.ExistsSubquery,
                ops.NotExistsSubquery,
            ),
        ) or (
            isinstance(node, ops.TableColumn)
            and not self.is_root(node.table.op())
        )

    def visit_table(
        self, expr, in_subquery=False, visit_cache=None, visit_table_cache=None
    ):
        if visit_table_cache is None:
            visit_table_cache = set()

        key = expr._key, in_subquery
        if key in visit_table_cache:
            return
        visit_table_cache.add(key)

        node = expr.op()

        if isinstance(node, (ops.PhysicalTable, ops.SelfReference)):
            self.ref_check(node, in_subquery=in_subquery)

        for arg in node.flat_args():
            if isinstance(arg, ir.Expr):
                self.visit(
                    arg,
                    in_subquery=in_subquery,
                    visit_cache=visit_cache,
                    visit_table_cache=visit_table_cache,
                )

    def ref_check(self, node, in_subquery: bool = False) -> None:
        ctx = self.ctx

        is_root = self.is_root(node)

        self.has_query_root |= is_root and in_subquery
        self.has_foreign_root |= not is_root and in_subquery

        if (
            not is_root
            and not ctx.has_ref(node)
            and (not in_subquery or ctx.has_ref(node, parent_contexts=True))
        ):
            ctx.make_alias(node)

    def is_root(self, what: ops.TableNode) -> bool:
        return what in self.query_roots


def _get_scalar(field):
    def scalar_handler(results):
        return results[field][0]

    return scalar_handler


def _get_column(name):
    def column_handler(results):
        return results[name]

    return column_handler


class SelectBuilder:

    """
    Transforms expression IR to a query pipeline (potentially multiple
    queries). There will typically be a primary SELECT query, perhaps with some
    subqueries and other DDL to ingest and tear down intermediate data sources.

    Walks the expression tree and catalogues distinct query units, builds
    select statements (and other DDL types, where necessary), and records
    relevant query unit aliases to be used when actually generating SQL.
    """

    def to_select(
        self,
        select_class,
        table_set_formatter_class,
        expr,
        context,
        translator_class,
    ):
        self.select_class = select_class
        self.table_set_formatter_class = table_set_formatter_class
        self.expr = expr
        self.context = context
        self.translator_class = translator_class

        self.query_expr, self.result_handler = self._adapt_expr(self.expr)

        self.table_set = None
        self.select_set = None
        self.group_by = None
        self.having = None
        self.filters = []
        self.limit = None
        self.sort_by = []
        self.subqueries = []
        self.distinct = False

        select_query = self._build_result_query()

        self.queries = [select_query]

        return select_query

    @staticmethod
    def _foreign_ref_check(query, expr):
        checker = _CorrelatedRefCheck(query, expr)
        return checker.get_result()

    @staticmethod
    def _adapt_expr(expr):
        # Non-table expressions need to be adapted to some well-formed table
        # expression, along with a way to adapt the results to the desired
        # arity (whether array-like or scalar, for example)
        #
        # Canonical case is scalar values or arrays produced by some reductions
        # (simple reductions, or distinct, say)

        if isinstance(expr, ir.Table):
            return expr, toolz.identity

        if isinstance(expr, ir.Scalar):
            if not expr.has_name():
                expr = expr.name('tmp')

            if L.is_scalar_reduction(expr):
                table_expr = L.reduction_to_aggregation(expr)
                return table_expr, _get_scalar(expr.get_name())
            else:
                return expr, _get_scalar(expr.get_name())

        elif isinstance(expr, ir.Analytic):
            return expr.to_aggregation(), toolz.identity

        elif isinstance(expr, ir.Column):
            op = expr.op()

            if isinstance(op, ops.TableColumn):
                table_expr = op.table[[op.name]]
                result_handler = _get_column(op.name)
            else:
                if not expr.has_name():
                    expr = expr.name('tmp')
                table_expr = expr.to_projection()
                result_handler = _get_column(expr.get_name())

            return table_expr, result_handler
        else:
            raise com.TranslationError(
                f'Do not know how to execute: {type(expr)}'
            )

    def _build_result_query(self):
        self._collect_elements()

        self._analyze_select_exprs()
        self._analyze_subqueries()
        self._populate_context()

        return self.select_class(
            self.table_set,
            self.select_set,
            translator_class=self.translator_class,
            table_set_formatter_class=self.table_set_formatter_class,
            context=self.context,
            subqueries=self.subqueries,
            where=self.filters,
            group_by=self.group_by,
            having=self.having,
            limit=self.limit,
            order_by=self.sort_by,
            distinct=self.distinct,
            result_handler=self.result_handler,
            parent_expr=self.query_expr,
        )

    def _populate_context(self):
        # Populate aliases for the distinct relations used to output this
        # select statement.
        if self.table_set is not None:
            self._make_table_aliases(self.table_set)

        # XXX: This is a temporary solution to the table-aliasing / correlated
        # subquery problem. Will need to revisit and come up with a cleaner
        # design (also as one way to avoid pathological naming conflicts; for
        # example, we could define a table alias before we know that it
        # conflicts with the name of a table used in a subquery, join, or
        # another part of the query structure)

        # There may be correlated subqueries inside the filters, requiring that
        # we use an explicit alias when outputting as SQL. For now, we're just
        # going to see if any table nodes appearing in the where stack have
        # been marked previously by the above code.
        for expr in self.filters:
            needs_alias = self._foreign_ref_check(self, expr)
            if needs_alias:
                self.context.set_always_alias()

    def _make_table_aliases(self, expr):
        ctx = self.context
        node = expr.op()
        if isinstance(node, ops.Join):
            for arg in node.args:
                if isinstance(arg, ir.Table):
                    self._make_table_aliases(arg)
        else:
            if not ctx.is_extracted(expr):
                ctx.make_alias(expr)
            else:
                # The compiler will apply a prefix only if the current context
                # contains two or more table references. So, if we've extracted
                # a subquery into a CTE, we need to propagate that reference
                # down to child contexts so that they aren't missing any refs.
                ctx.set_ref(expr, ctx.top_context.get_ref(expr))

    # ---------------------------------------------------------------------
    # Expr analysis / rewrites

    def _analyze_select_exprs(self):
        new_select_set = []

        for expr in self.select_set:
            new_expr = self._visit_select_expr(expr)
            new_select_set.append(new_expr)

        self.select_set = new_select_set

    def _visit_select_expr(self, expr):
        op = expr.op()

        method = f'_visit_select_{type(op).__name__}'
        if hasattr(self, method):
            f = getattr(self, method)
            return f(expr)

        unchanged = True

        if isinstance(op, ops.Value):
            new_args = []
            for arg in op.args:
                if isinstance(arg, ir.Expr):
                    new_arg = self._visit_select_expr(arg)
                    if arg is not new_arg:
                        unchanged = False
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)

            if not unchanged:
                new_op = type(op)(*new_args)
                return new_op.to_expr()
            else:
                return expr
        else:
            return expr

    def _visit_select_Histogram(self, expr):
        op = expr.op()

        EPS = 1e-13

        if op.binwidth is None or op.base is None:
            aux_hash = op.aux_hash or util.guid()[:6]

            min_name = 'min_%s' % aux_hash
            max_name = 'max_%s' % aux_hash

            minmax = self.table_set.aggregate(
                [op.arg.min().name(min_name), op.arg.max().name(max_name)]
            )
            self.table_set = self.table_set.cross_join(minmax)

            if op.base is None:
                base = minmax[min_name] - EPS
            else:
                base = op.base

            binwidth = (minmax[max_name] - base) / (op.nbins - 1)
        else:
            # Have both a bin width and a base
            binwidth = op.binwidth
            base = op.base

        bucket = ((op.arg - base) / binwidth).floor()
        if expr.has_name():
            bucket = bucket.name(expr.get_name())

        return bucket

    @util.deprecated(
        instead=(
            "do nothing; Any/NotAny is transformed into "
            "ExistsSubquery/NotExistsSubquery at expression construction time"
        ),
        version="4.0.0",
    )
    def _visit_filter_Any(self, expr):  # pragma: no cover
        return expr

    _visit_filter_NotAny = _visit_filter_Any

    @util.deprecated(
        instead="do nothing; SummaryFilter will be removed",
        version="4.0.0",
    )
    def _visit_filter_SummaryFilter(self, expr):  # pragma: no cover
        # Top K is rewritten as an
        # - aggregation
        # - sort by
        # - limit
        # - left semi join with table set
        parent_op = expr.op()
        summary_expr = parent_op.args[0]
        op = summary_expr.op()

        rank_set = summary_expr.to_aggregation(
            backup_metric_name='__tmp__', parent_table=self.table_set
        )

        # GH 1393: previously because of GH667 we were substituting parents,
        # but that introduced a bug when comparing reductions to columns on the
        # same relation, so we leave this alone.
        arg = op.arg
        pred = arg == getattr(rank_set, arg.get_name())
        self.table_set = self.table_set.semi_join(rank_set, [pred])

    # ---------------------------------------------------------------------
    # Analysis of table set

    def _collect_elements(self):
        # If expr is a Value, we must seek out the Tables that it
        # references, build their ASTs, and mark them in our QueryContext

        # For now, we need to make the simplifying assumption that a value
        # expression that is being translated only depends on a single table
        # expression.

        source_expr = self.query_expr

        # hm, is this the best place for this?
        root_op = source_expr.op()

        if isinstance(root_op, ops.TableNode):
            self._collect(source_expr, toplevel=True)
            assert self.table_set is not None
        else:
            self.select_set = [source_expr]

    def _collect(self, expr, toplevel=False):
        op = expr.op()
        method = f'_collect_{type(op).__name__}'

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr, toplevel=toplevel)
        elif isinstance(op, (ops.PhysicalTable, ops.SQLQueryResult)):
            self._collect_PhysicalTable(expr, toplevel=toplevel)
        elif isinstance(op, ops.Join):
            self._collect_Join(expr, toplevel=toplevel)
        else:
            raise NotImplementedError(type(op))

    def _collect_Distinct(self, expr, toplevel=False):
        if toplevel:
            self.distinct = True

        self._collect(expr.op().table, toplevel=toplevel)

    def _collect_DropNa(self, expr, toplevel=False):
        if toplevel:
            op = expr.op()
            if op.subset is None:
                columns = [op.table[c] for c in op.table.columns]
            else:
                columns = op.subset
            if columns:
                filters = [
                    functools.reduce(
                        operator.and_ if op.how == "any" else operator.or_,
                        [c.notnull() for c in columns],
                    )
                ]
            elif op.how == "all":
                filters = [ibis.literal(False)]
            else:
                filters = []
            self.table_set = op.table
            self.select_set = [op.table]
            self.filters = filters

    def _collect_Limit(self, expr, toplevel=False):
        if not toplevel:
            return

        op = expr.op()
        n = op.n
        offset = op.offset or 0

        if self.limit is None:
            self.limit = _LimitSpec(n, offset)
        else:
            self.limit = _LimitSpec(
                min(n, self.limit.n),
                offset + self.limit.offset,
            )

        self._collect(op.table, toplevel=toplevel)

    def _collect_Union(self, expr, toplevel=False):
        if toplevel:
            raise NotImplementedError()

    def _collect_Difference(self, expr, toplevel=False):
        if toplevel:
            raise NotImplementedError()

    def _collect_Intersection(self, expr, toplevel=False):
        if toplevel:
            raise NotImplementedError()

    def _collect_Aggregation(self, expr, toplevel=False):
        # The select set includes the grouping keys (if any), and these are
        # duplicated in the group_by set. SQL translator can decide how to
        # format these depending on the database. Most likely the
        # GROUP BY 1, 2, ... style
        if toplevel:
            subbed_expr = self._sub(expr)
            sub_op = subbed_expr.op()

            self.group_by = self._convert_group_by(sub_op.by)
            self.having = sub_op.having
            self.select_set = sub_op.by + sub_op.metrics
            self.table_set = sub_op.table
            self.filters = sub_op.predicates
            self.sort_by = sub_op.sort_keys

            self._collect(expr.op().table)

    def _collect_Selection(self, expr, toplevel=False):
        op = expr.op()
        table = op.table

        if toplevel:
            if isinstance(table.op(), ops.Join):
                self._collect_Join(table)
            else:
                self._collect(table)

            selections = op.selections
            sort_keys = op.sort_keys
            filters = op.predicates

            if not selections:
                # select *
                selections = [table]

            self.sort_by = sort_keys
            self.select_set = selections
            self.table_set = table
            self.filters = filters

    def _collect_PandasInMemoryTable(self, expr, toplevel=False):
        if toplevel:
            self.select_set = [expr]
            self.table_set = expr

    def _convert_group_by(self, exprs):
        return list(range(len(exprs)))

    def _collect_Join(self, expr, toplevel=False):
        if toplevel:
            subbed = self._sub(expr)
            self.table_set = subbed
            self.select_set = [subbed]

    def _collect_PhysicalTable(self, expr, toplevel=False):
        if toplevel:
            self.select_set = [expr]
            self.table_set = expr

    def _collect_SelfReference(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            self._collect(op.table, toplevel=toplevel)

    def _sub(self, what):
        return L.substitute_parents(what)

    # --------------------------------------------------------------------
    # Subquery analysis / extraction

    def _analyze_subqueries(self):
        # Somewhat temporary place for this. A little bit tricky, because
        # subqueries can be found in many places
        # - With the table set
        # - Inside the where clause (these may be able to place directly, some
        #   cases not)
        # - As support queries inside certain expressions (possibly needing to
        #   be extracted and joined into the table set where they are
        #   used). More complex transformations should probably not occur here,
        #   though.
        #
        # Duplicate subqueries might appear in different parts of the query
        # structure, e.g. beneath two aggregates that are joined together, so
        # we have to walk the entire query structure.
        #
        # The default behavior is to only extract into a WITH clause when a
        # subquery appears multiple times (for DRY reasons). At some point we
        # can implement a more aggressive policy so that subqueries always
        # appear in the WITH part of the SELECT statement, if that's what you
        # want.

        # Find the subqueries, and record them in the passed query context.
        subqueries = _extract_common_table_expressions(
            [self.table_set, *self.filters]
        )

        self.subqueries = []
        for expr in subqueries:
            # See #173. Might have been extracted already in a parent context.
            if not self.context.is_extracted(expr):
                self.subqueries.append(expr)
                self.context.set_extracted(expr)
