import toolz

import ibis.common.exceptions as com
import ibis.expr.analysis as L
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util

from .extract_subqueries import ExtractSubqueries


class _AnyToExistsTransform:

    """
    Some code duplication with the correlated ref check; should investigate
    better code reuse.
    """

    def __init__(self, context, expr, parent_table):
        self.context = context
        self.expr = expr
        self.parent_table = parent_table
        self.query_roots = frozenset(self.parent_table.op().root_tables())

    def get_result(self):
        self.foreign_table = None
        self.predicates = []

        self._visit(self.expr)

        if type(self.expr.op()) == ops.Any:
            op = ops.ExistsSubquery(self.foreign_table, self.predicates)
        else:
            op = ops.NotExistsSubquery(self.foreign_table, self.predicates)

        expr_type = dt.boolean.column_type()
        return expr_type(op)

    def _visit(self, expr):
        node = expr.op()

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg)
            elif isinstance(arg, ir.BooleanColumn):
                for sub_expr in L.flatten_predicate(arg):
                    self.predicates.append(sub_expr)
                    self._visit(sub_expr)
            elif isinstance(arg, ir.Expr):
                self._visit(arg)
            else:
                continue

    def _find_blocking_table(self, expr):
        node = expr.op()

        if node.blocks():
            return expr

        for arg in node.flat_args():
            if isinstance(arg, ir.Expr):
                result = self._find_blocking_table(arg)
                if result is not None:
                    return result

    def _visit_table(self, expr):
        node = expr.op()

        if isinstance(expr, ir.TableExpr):
            base_table = self._find_blocking_table(expr)
            if base_table is not None:
                base_node = base_table.op()
                if self._is_root(base_node):
                    pass
                else:
                    # Foreign ref
                    self.foreign_table = expr
        else:
            if not node.blocks():
                for arg in node.flat_args():
                    if isinstance(arg, ir.Expr):
                        self._visit(arg)

    def _is_root(self, what):
        if isinstance(what, ir.Expr):
            what = what.op()
        return what in self.query_roots


class _CorrelatedRefCheck:
    def __init__(self, query, expr):
        self.query = query
        self.ctx = query.context
        self.expr = expr
        self.query_roots = frozenset(self.query.table_set.op().root_tables())

        # aliasing required
        self.foreign_refs = []

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

        in_subquery = in_subquery or self.is_subquery(node)

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
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
        # XXX
        if isinstance(
            node,
            (
                ops.TableArrayView,
                ops.ExistsSubquery,
                ops.NotExistsSubquery,
            ),
        ):
            return True

        if isinstance(node, ops.TableColumn):
            return not self.is_root(node.table)

        return False

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

    def ref_check(self, node, in_subquery=False):
        ctx = self.ctx
        is_aliased = ctx.has_ref(node)

        if self.is_root(node):
            if in_subquery:
                self.has_query_root = True
        else:
            if in_subquery:
                self.has_foreign_root = True
                if not is_aliased and ctx.has_ref(node, parent_contexts=True):
                    ctx.make_alias(node)
            elif not ctx.has_ref(node):
                ctx.make_alias(node)

    def is_root(self, what):
        if isinstance(what, ir.Expr):
            what = what.op()
        return what in self.query_roots


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

        self.sub_memo = {}

        self.queries = []

        self.table_set = None
        self.select_set = None
        self.group_by = None
        self.having = None
        self.filters = []
        self.limit = None
        self.sort_by = []
        self.subqueries = []
        self.distinct = False

        self.op_memo = set()

        # make idempotent
        if self.queries:
            return self._wrap_result()

        select_query = self._build_result_query()

        self.queries.append(select_query)

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

        if isinstance(expr, ir.TableExpr):
            return expr, toolz.identity

        def _get_scalar(field):
            def scalar_handler(results):
                return results[field][0]

            return scalar_handler

        if isinstance(expr, ir.ScalarExpr):

            if L.is_scalar_reduction(expr):
                table_expr, name = L.reduction_to_aggregation(
                    expr, default_name='tmp'
                )
                return table_expr, _get_scalar(name)
            else:
                base_table = ir.find_base_table(expr)
                if base_table is None:
                    # exprs with no table refs
                    # TODO(phillipc): remove ScalarParameter hack
                    if isinstance(expr.op(), ops.ScalarParameter):
                        name = expr.get_name()
                        assert (
                            name is not None
                        ), f'scalar parameter {expr} has no name'
                        return expr, _get_scalar(name)
                    return expr.name('tmp'), _get_scalar('tmp')

                raise NotImplementedError(repr(expr))

        elif isinstance(expr, ir.AnalyticExpr):
            return expr.to_aggregation(), toolz.identity

        elif isinstance(expr, ir.ColumnExpr):
            op = expr.op()

            def _get_column(name):
                def column_handler(results):
                    return results[name]

                return column_handler

            if isinstance(op, ops.TableColumn):
                table_expr = op.table[[op.name]]
                result_handler = _get_column(op.name)
            else:
                # Something more complicated.
                base_table = L.find_source_table(expr)

                if isinstance(op, ops.DistinctColumn):
                    expr = op.arg
                    try:
                        name = op.arg.get_name()
                    except Exception:
                        name = 'tmp'

                    table_expr = base_table.projection(
                        [expr.name(name)]
                    ).distinct()
                    result_handler = _get_column(name)
                else:
                    table_expr = base_table.projection([expr.name('tmp')])
                    result_handler = _get_column('tmp')

            return table_expr, result_handler
        else:
            raise com.TranslationError(
                f'Do not know how to execute: {type(expr)}'
            )

    @staticmethod
    def _get_subtables(expr):
        subtables = []

        stack = [expr]
        seen = set()

        while stack:
            e = stack.pop()
            op = e.op()

            if op not in seen:
                seen.add(op)

                if isinstance(op, ops.Join):
                    stack.append(op.right)
                    stack.append(op.left)
                else:
                    subtables.append(e)

        return subtables

    @classmethod
    def _blocking_base(cls, expr):
        node = expr.op()
        if node.blocks() or isinstance(node, ops.Join):
            return expr
        else:
            for arg in expr.op().flat_args():
                if isinstance(arg, ir.TableExpr):
                    return cls._blocking_base(arg)

    @classmethod
    def _all_distinct_roots(cls, subtables):
        bases = []
        for t in subtables:
            base = cls._blocking_base(t)
            for x in bases:
                if base.equals(x):
                    return False
            bases.append(base)
        return True

    def _build_result_query(self):
        self._collect_elements()

        self._analyze_select_exprs()
        self._analyze_filter_exprs()
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
                if isinstance(arg, ir.TableExpr):
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

        if isinstance(op, ops.ValueOp):
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
                return expr._factory(type(op)(*new_args))
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

        bucket = (op.arg - base) / binwidth
        return bucket.floor().name(expr._name)

    def _analyze_filter_exprs(self):
        # What's semantically contained in the filter predicates may need to be
        # rewritten. Not sure if this is the right place to do this, but a
        # starting point

        # Various kinds of semantically valid WHERE clauses may need to be
        # rewritten into a form that we can actually translate into valid SQL.
        new_where = []
        for expr in self.filters:
            new_expr = self._visit_filter(expr)

            # Transformations may result in there being no outputted filter
            # predicate
            if new_expr is not None:
                new_where.append(new_expr)

        self.filters = new_where

    def _visit_filter(self, expr):
        # Dumping ground for analysis of WHERE expressions
        # - Subquery extraction
        # - Conversion to explicit semi/anti joins
        # - Rewrites to generate subqueries

        op = expr.op()

        method = f'_visit_filter_{type(op).__name__}'
        if hasattr(self, method):
            f = getattr(self, method)
            return f(expr)

        unchanged = True
        if isinstance(expr, ir.ScalarExpr):
            if L.is_reduction(expr):
                return self._rewrite_reduction_filter(expr)

        if isinstance(op, ops.BinaryOp):
            left = self._visit_filter(op.left)
            right = self._visit_filter(op.right)
            unchanged = left is op.left and right is op.right
            if not unchanged:
                return expr._factory(type(op)(left, right))
            else:
                return expr
        elif isinstance(
            op, (ops.Any, ops.BooleanValueOp, ops.TableColumn, ops.Literal)
        ):
            return expr
        elif isinstance(op, ops.ValueOp):
            visited = [
                self._visit_filter(arg) if isinstance(arg, ir.Expr) else arg
                for arg in op.args
            ]
            unchanged = True
            for new, old in zip(visited, op.args):
                if new is not old:
                    unchanged = False
            if not unchanged:
                return expr._factory(type(op)(*visited))
            else:
                return expr
        else:
            raise NotImplementedError(type(op))

    def _rewrite_reduction_filter(self, expr):
        # Find the table that this reduction references.

        # TODO: what about reductions that reference a join that isn't visible
        # at this level? Means we probably have the wrong design, but will have
        # to revisit when it becomes a problem.
        aggregation, _ = L.reduction_to_aggregation(expr, default_name='tmp')
        return aggregation.to_array()

    def _visit_filter_Any(self, expr):
        # Rewrite semi/anti-join predicates in way that can hook into SQL
        # translation step
        transform = _AnyToExistsTransform(self.context, expr, self.table_set)
        return transform.get_result()

    _visit_filter_NotAny = _visit_filter_Any

    def _visit_filter_SummaryFilter(self, expr):
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
        # If expr is a ValueExpr, we must seek out the TableExprs that it
        # references, build their ASTs, and mark them in our QueryContext

        # For now, we need to make the simplifying assumption that a value
        # expression that is being translated only depends on a single table
        # expression.

        source_expr = self.query_expr

        # hm, is this the best place for this?
        root_op = source_expr.op()
        if isinstance(root_op, ops.Join) and not isinstance(
            root_op, ops.MaterializedJoin
        ):
            # Unmaterialized join
            source_expr = source_expr.materialize()

        if isinstance(root_op, ops.TableNode):
            self._collect(source_expr, toplevel=True)
            if self.table_set is None:
                raise com.InternalError('no table set')
        else:
            self.select_set = [source_expr]

    def _collect(self, expr, toplevel=False):
        op = expr.op()
        method = f'_collect_{type(op).__name__}'

        # Do not visit nodes twice
        if op in self.op_memo:
            return

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr, toplevel=toplevel)
        elif isinstance(op, (ops.PhysicalTable, ops.SQLQueryResult)):
            self._collect_PhysicalTable(expr, toplevel=toplevel)
        elif isinstance(op, ops.Join):
            self._collect_Join(expr, toplevel=toplevel)
        else:
            raise NotImplementedError(type(op))

        self.op_memo.add(op)

    def _collect_Distinct(self, expr, toplevel=False):
        if toplevel:
            self.distinct = True

        self._collect(expr.op().table, toplevel=toplevel)

    def _collect_Limit(self, expr, toplevel=False):
        if not toplevel:
            return

        op = expr.op()

        # Ignore "inner" limits, because they've been overrided by an exterior
        # one
        if self.limit is None:
            self.limit = {'n': op.n, 'offset': op.offset}

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
            subbed = self._sub(expr)
            sop = subbed.op()

            if isinstance(table.op(), ops.Join):
                can_sub = self._collect_Join(table)
            else:
                can_sub = False
                self._collect(table)

            selections = op.selections
            sort_keys = op.sort_keys
            filters = op.predicates

            if can_sub:
                selections = sop.selections
                filters = sop.predicates
                sort_keys = sop.sort_keys
                table = sop.table

            if len(selections) == 0:
                # select *
                selections = [table]

            self.sort_by = sort_keys
            self.select_set = selections
            self.table_set = table
            self.filters = filters

    def _collect_MaterializedJoin(self, expr, toplevel=False):
        op = expr.op()
        join = op.join

        if toplevel:
            subbed = self._sub(join)
            self.table_set = subbed
            self.select_set = [subbed]

        self._collect_Join(join, toplevel=False)

    def _convert_group_by(self, exprs):
        return list(range(len(exprs)))

    def _collect_Join(self, expr, toplevel=False):
        if toplevel:
            subbed = self._sub(expr)
            self.table_set = subbed
            self.select_set = [subbed]

        subtables = self._get_subtables(expr)

        # If any of the joined tables are non-blocking modified versions of the
        # same table, then it's not safe to continue walking down the tree (see
        # #667), and we should instead have inline views rather than attempting
        # to fuse things together into the same SELECT query.
        can_substitute = self._all_distinct_roots(subtables)
        if can_substitute:
            for table in subtables:
                self._collect(table, toplevel=False)

        return can_substitute

    def _collect_PhysicalTable(self, expr, toplevel=False):
        if toplevel:
            self.select_set = [expr]
            self.table_set = expr  # self._sub(expr)

    def _collect_SelfReference(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            self._collect(op.table, toplevel=toplevel)

    def _sub(self, what):
        if isinstance(what, list):
            return [L.substitute_parents(x, self.sub_memo) for x in what]
        else:
            return L.substitute_parents(what, self.sub_memo)

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
        subqueries = ExtractSubqueries.extract(self)
        self.subqueries = []
        for expr in subqueries:
            # See #173. Might have been extracted already in a parent context.
            if not self.context.is_extracted(expr):
                self.subqueries.append(expr)
                self.context.set_extracted(expr)
