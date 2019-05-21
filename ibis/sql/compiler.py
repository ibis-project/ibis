"""Core expression to SQL infrastructure.
"""

import abc

from collections import defaultdict
from itertools import chain

import six

from six import StringIO

import toolz

import ibis
import ibis.common as com
import ibis.util as util
import ibis.expr.types as ir
import ibis.expr.format as fmt
import ibis.expr.analysis as L
import ibis.expr.analytics as analytics
import ibis.expr.operations as ops
import ibis.sql.transforms as transforms

from ibis.compat import map


@six.add_metaclass(abc.ABCMeta)
class DML(object):
    @abc.abstractmethod
    def compile(self):
        pass


@six.add_metaclass(abc.ABCMeta)
class DDL(object):
    @abc.abstractmethod
    def compile(self):
        pass


class QueryAST(object):

    __slots__ = 'context', 'dml', 'setup_queries', 'teardown_queries'

    def __init__(
        self, context, dml, setup_queries=None, teardown_queries=None
    ):
        self.context = context
        self.dml = dml
        self.setup_queries = setup_queries
        self.teardown_queries = teardown_queries

    @property
    def queries(self):
        return [self.dml]

    def compile(self):
        compiled_setup_queries = [q.compile() for q in self.setup_queries]
        compiled_queries = [q.compile() for q in self.queries]
        compiled_teardown_queries = [
            q.compile() for q in self.teardown_queries
        ]
        return self.context.collapse(list(chain(
            compiled_setup_queries,
            compiled_queries,
            compiled_teardown_queries,
        )))


class SelectBuilder(object):

    """
    Transforms expression IR to a query pipeline (potentially multiple
    queries). There will typically be a primary SELECT query, perhaps with some
    subqueries and other DDL to ingest and tear down intermediate data sources.

    Walks the expression tree and catalogues distinct query units, builds
    select statements (and other DDL types, where necessary), and records
    relevant query unit aliases to be used when actually generating SQL.
    """

    def __init__(self, expr, context):
        self.expr = expr

        self.query_expr, self.result_handler = _adapt_expr(self.expr)

        self.sub_memo = {}

        self.context = context
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

        self.op_memo = util.IbisSet()

    def get_result(self):
        # make idempotent
        if self.queries:
            return self._wrap_result()

        select_query = self._build_result_query()

        self.queries.append(select_query)

        return select_query

    def _build_result_query(self):
        self._collect_elements()

        self._analyze_select_exprs()
        self._analyze_filter_exprs()
        self._analyze_subqueries()
        self._populate_context()

        klass = self._select_class

        return klass(self.table_set, self.select_set,
                     subqueries=self.subqueries,
                     where=self.filters,
                     group_by=self.group_by,
                     having=self.having,
                     limit=self.limit,
                     order_by=self.sort_by,
                     distinct=self.distinct,
                     result_handler=self.result_handler,
                     parent_expr=self.query_expr,
                     context=self.context)

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
            needs_alias = _foreign_ref_check(self, expr)
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

        method = '_visit_select_{0}'.format(type(op).__name__)
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

            minmax = self.table_set.aggregate([op.arg.min().name(min_name),
                                               op.arg.max().name(max_name)])
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

        method = '_visit_filter_{0}'.format(type(op).__name__)
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
        elif isinstance(op, (ops.Any, ops.BooleanValueOp,
                             ops.TableColumn, ops.Literal)):
            return expr
        elif isinstance(op, ops.ValueOp):
            visited = [self._visit_filter(arg)
                       if isinstance(arg, ir.Expr) else arg
                       for arg in op.args]
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
        transform = transforms.AnyToExistsTransform(self.context, expr,
                                                    self.table_set)
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
            backup_metric_name='__tmp__',
            parent_table=self.table_set)

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
        if (isinstance(root_op, ops.Join) and
                not isinstance(root_op, ops.MaterializedJoin)):
            # Unmaterialized join
            source_expr = source_expr.materialize()

        if isinstance(root_op, ops.TableNode):
            self._collect(source_expr, toplevel=True)
            if self.table_set is None:
                raise com.InternalError('no table set')
        else:
            # Expressions not depending on any table
            if isinstance(root_op, ops.ExpressionList):
                self.select_set = source_expr.exprs()
            else:
                self.select_set = [source_expr]

    def _collect(self, expr, toplevel=False):
        op = expr.op()
        method = '_collect_{0}'.format(type(op).__name__)

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
            self.limit = {
                'n': op.n,
                'offset': op.offset
            }

        self._collect(op.table, toplevel=toplevel)

    def _collect_Union(self, expr, toplevel=False):
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

        subtables = _get_subtables(expr)

        # If any of the joined tables are non-blocking modified versions of the
        # same table, then it's not safe to continue walking down the tree (see
        # #667), and we should instead have inline views rather than attempting
        # to fuse things together into the same SELECT query.
        can_substitute = _all_distinct_roots(subtables)
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
        subqueries = _extract_subqueries(self)
        self.subqueries = []
        for expr in subqueries:
            # See #173. Might have been extracted already in a parent context.
            if not self.context.is_extracted(expr):
                self.subqueries.append(expr)
                self.context.set_extracted(expr)


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


def _all_distinct_roots(subtables):
    bases = []
    for t in subtables:
        base = _blocking_base(t)
        for x in bases:
            if base.equals(x):
                return False
        bases.append(base)
    return True


def _blocking_base(expr):
    node = expr.op()
    if node.blocks() or isinstance(node, ops.Join):
        return expr
    else:
        for arg in expr.op().flat_args():
            if isinstance(arg, ir.TableExpr):
                return _blocking_base(arg)


def _extract_subqueries(select_stmt):
    helper = _ExtractSubqueries(select_stmt)
    return helper.get_result()


def _extract_noop(self, expr):
    return


class _ExtractSubqueries(object):

    # Helper class to make things a little easier

    def __init__(self, query, greedy=False):
        self.query = query
        self.greedy = greedy

        # Ordered set that uses object .equals to find keys
        self.observed_exprs = util.IbisMap()

        self.expr_counts = defaultdict(int)

    def get_result(self):
        if self.query.table_set is not None:
            self.visit(self.query.table_set)

        for clause in self.query.filters:
            self.visit(clause)

        to_extract = []

        # Read them inside-out, to avoid nested dependency issues
        for expr, key in zip(self.observed_exprs.keys,
                             self.observed_exprs.values):
            v = self.expr_counts[key]

            if self.greedy or v > 1:
                to_extract.append(expr)

        return to_extract

    def observe(self, expr):
        if self.seen(expr):
            key = self.observed_exprs.get(expr)
        else:
            # this key only needs to be unique because of the IbisMap
            key = id(expr.op())
            self.observed_exprs.set(expr, key)

        self.expr_counts[key] += 1

    def seen(self, expr):
        return expr in self.observed_exprs

    def visit(self, expr):
        node = expr.op()
        method = '_visit_{0}'.format(type(node).__name__)

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr)
        elif isinstance(node, ops.Join):
            self._visit_join(expr)
        elif isinstance(node, ops.PhysicalTable):
            self._visit_physical_table(expr)
        elif isinstance(node, ops.ValueOp):
            for arg in node.flat_args():
                if not isinstance(arg, ir.Expr):
                    continue
                self.visit(arg)
        else:
            raise NotImplementedError(type(node))

    def _visit_join(self, expr):
        node = expr.op()
        self.visit(node.left)
        self.visit(node.right)

    _visit_physical_table = _extract_noop

    def _visit_Exists(self, expr):
        node = expr.op()
        self.visit(node.foreign_table)
        for pred in node.predicates:
            self.visit(pred)

    _visit_ExistsSubquery = _visit_Exists
    _visit_NotExistsSubquery = _visit_Exists

    def _visit_Aggregation(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def _visit_Distinct(self, expr):
        self.observe(expr)

    def _visit_Limit(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def _visit_Union(self, expr):
        op = expr.op()
        self.visit(op.left)
        self.visit(op.right)
        self.observe(expr)

    def _visit_MaterializedJoin(self, expr):
        self.visit(expr.op().join)
        self.observe(expr)

    def _visit_Selection(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def _visit_SQLQueryResult(self, expr):
        self.observe(expr)

    def _visit_TableColumn(self, expr):
        table = expr.op().table
        if not self.seen(table):
            self.visit(table)

    def _visit_SelfReference(self, expr):
        self.visit(expr.op().table)


def _foreign_ref_check(query, expr):
    checker = _CorrelatedRefCheck(query, expr)
    return checker.get_result()


class _CorrelatedRefCheck(object):

    def __init__(self, query, expr):
        self.query = query
        self.ctx = query.context
        self.expr = expr

        qroots = self.query.table_set._root_tables()

        self.query_roots = util.IbisSet.from_list(qroots)

        # aliasing required
        self.foreign_refs = []

        self.has_foreign_root = False
        self.has_query_root = False

    def get_result(self):
        self._visit(self.expr)
        return self.has_query_root and self.has_foreign_root

    def _visit(self, expr, in_subquery=False, visit_cache=None,
               visit_table_cache=None):
        if visit_cache is None:
            visit_cache = set()

        if (id(expr), in_subquery) in visit_cache:
            return
        visit_cache.add((id(expr), in_subquery))

        node = expr.op()

        in_subquery = in_subquery or self._is_subquery(node)

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg, in_subquery=in_subquery,
                                  visit_cache=visit_cache,
                                  visit_table_cache=visit_table_cache)
            elif isinstance(arg, ir.Expr):
                self._visit(arg, in_subquery=in_subquery,
                            visit_cache=visit_cache,
                            visit_table_cache=visit_table_cache)

    def _is_subquery(self, node):
        # XXX
        if isinstance(node, (ops.TableArrayView,
                             transforms.ExistsSubquery,
                             transforms.NotExistsSubquery)):
            return True

        if isinstance(node, ops.TableColumn):
            return not self._is_root(node.table)

        return False

    def _visit_table(self, expr, in_subquery=False, visit_cache=None,
                     visit_table_cache=None):
        if visit_table_cache is None:
            visit_table_cache = set()

        if (id(expr), in_subquery) in visit_table_cache:
            return
        visit_table_cache.add((id(expr), in_subquery))

        node = expr.op()

        if isinstance(node, (ops.PhysicalTable, ops.SelfReference)):
            self._ref_check(node, in_subquery=in_subquery)

        for arg in node.flat_args():
            if isinstance(arg, ir.Expr):
                self._visit(arg, in_subquery=in_subquery,
                            visit_cache=visit_cache,
                            visit_table_cache=visit_table_cache)

    def _ref_check(self, node, in_subquery=False):
        is_aliased = self.ctx.has_ref(node)

        if self._is_root(node):
            if in_subquery:
                self.has_query_root = True
        else:
            if in_subquery:
                self.has_foreign_root = True
                if (not is_aliased and
                        self.ctx.has_ref(node, parent_contexts=True)):
                    self.ctx.make_alias(node)
            elif not self.ctx.has_ref(node):
                self.ctx.make_alias(node)

    def _is_root(self, what):
        if isinstance(what, ir.Expr):
            what = what.op()
        return what in self.query_roots


def _adapt_expr(expr):
    # Non-table expressions need to be adapted to some well-formed table
    # expression, along with a way to adapt the results to the desired
    # arity (whether array-like or scalar, for example)
    #
    # Canonical case is scalar values or arrays produced by some reductions
    # (simple reductions, or distinct, say)
    def as_is(x):
        return x

    if isinstance(expr, ir.TableExpr):
        return expr, as_is

    def _get_scalar(field):
        def scalar_handler(results):
            return results[field][0]
        return scalar_handler

    if isinstance(expr, ir.ScalarExpr):

        if L.is_scalar_reduction(expr):
            table_expr, name = L.reduction_to_aggregation(
                expr, default_name='tmp')
            return table_expr, _get_scalar(name)
        else:
            base_table = ir.find_base_table(expr)
            if base_table is None:
                # exprs with no table refs
                # TODO(phillipc): remove ScalarParameter hack
                if isinstance(expr.op(), ops.ScalarParameter):
                    name = expr.get_name()
                    assert name is not None, \
                        'scalar parameter {} has no name'.format(expr)
                    return expr, _get_scalar(name)
                return expr.name('tmp'), _get_scalar('tmp')

            raise NotImplementedError(repr(expr))

    elif isinstance(expr, ir.AnalyticExpr):
        return expr.to_aggregation(), as_is

    elif isinstance(expr, ir.ExprList):
        exprs = expr.exprs()

        is_aggregation = True
        any_aggregation = False

        for x in exprs:
            if not L.is_scalar_reduction(x):
                is_aggregation = False
            else:
                any_aggregation = True

        if is_aggregation:
            table = ir.find_base_table(exprs[0])
            return table.aggregate(exprs), as_is
        elif not any_aggregation:
            return expr, as_is
        else:
            raise NotImplementedError(expr._repr())

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

                table_expr = (base_table.projection([expr.name(name)])
                              .distinct())
                result_handler = _get_column(name)
            else:
                table_expr = base_table.projection([expr.name('tmp')])
                result_handler = _get_column('tmp')

        return table_expr, result_handler
    else:
        raise com.TranslationError('Do not know how to execute: {0}'
                                   .format(type(expr)))


class Union(DML):

    def __init__(self, tables, expr, context, distincts):
        self.context = context
        self.tables = tables
        self.distincts = distincts
        self.table_set = expr
        self.filters = []

    def _extract_subqueries(self):
        self.subqueries = _extract_subqueries(self)
        for subquery in self.subqueries:
            self.context.set_extracted(subquery)

    def format_subqueries(self):
        context = self.context
        subqueries = self.subqueries

        return ',\n'.join(
            '{} AS (\n{}\n)'.format(
                context.get_ref(expr),
                util.indent(context.get_compiled_expr(expr), 2)
            ) for expr in subqueries
        )

    def format_relation(self, expr):
        ref = self.context.get_ref(expr)
        if ref is not None:
            return 'SELECT *\nFROM {}'.format(ref)
        return self.context.get_compiled_expr(expr)

    @staticmethod
    def keyword(distinct):
        return 'UNION' if distinct else 'UNION ALL'

    def compile(self):
        self._extract_subqueries()

        extracted = self.format_subqueries()

        buf = []

        if extracted:
            buf.append('WITH {}'.format(extracted))

        # interleave correct keyword for the backend in between the formatted
        # UNION expressions
        buf.extend(
            toolz.interleave(
                (
                    map(self.format_relation, self.tables),
                    map(self.keyword, self.distincts)
                )
            )
        )
        return '\n'.join(buf)


def flatten_union(table):
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
        return toolz.concatv(
            flatten_union(op.left), [op.distinct], flatten_union(op.right)
        )
    return [table]


class QueryBuilder(object):

    select_builder = SelectBuilder
    union_class = Union

    def __init__(self, expr, context):
        self.expr = expr
        self.context = context

    def generate_setup_queries(self):
        return []

    def generate_teardown_queries(self):
        return []

    def get_result(self):
        op = self.expr.op()

        # collect setup and teardown queries
        setup_queries = self.generate_setup_queries()
        teardown_queries = self.generate_teardown_queries()

        # TODO: any setup / teardown DDL statements will need to be done prior
        # to building the result set-generating statements.
        if isinstance(op, ops.Union):
            query = self._make_union()
        else:
            query = self._make_select()

        return QueryAST(
            self.context,
            query,
            setup_queries=setup_queries,
            teardown_queries=teardown_queries,
        )

    def _make_union(self):
        # flatten unions so that we can codegen them all at once
        union_info = list(flatten_union(self.expr))

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
        return self.union_class(table_exprs, self.expr,
                                distincts=distincts,
                                context=self.context)

    def _make_select(self):
        builder = self.select_builder(self.expr, self.context)
        return builder.get_result()


class QueryContext(object):

    """Records bits of information used during ibis AST to SQL translation.

    Notably, table aliases (for subquery construction) and scalar query
    parameters are tracked here.
    """

    def __init__(
        self, indent=2, parent=None, memo=None, dialect=None, params=None
    ):
        self._table_refs = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent
        self.parent = parent

        self.always_alias = False

        self.query = None

        self._table_key_memo = {}
        self.memo = memo or fmt.FormatMemo()
        self.dialect = dialect
        self.params = params if params is not None else {}

    def _compile_subquery(self, expr):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx)

    def _to_sql(self, expr, ctx):
        raise NotImplementedError

    def collapse(self, queries):
        """Turn a sequence of queries into something executable.

        Parameters
        ----------
        queries : List[str]

        Returns
        -------
        query : str
        """
        return '\n\n'.join(queries)

    @property
    def top_context(self):
        if self.parent is None:
            return self
        else:
            return self.parent.top_context

    def set_always_alias(self):
        self.always_alias = True

    def get_compiled_expr(self, expr):
        this = self.top_context

        key = self._get_table_key(expr)
        if key in this.subquery_memo:
            return this.subquery_memo[key]

        op = expr.op()
        if isinstance(op, ops.SQLQueryResult):
            result = op.query
        else:
            result = self._compile_subquery(expr)

        this.subquery_memo[key] = result
        return result

    def make_alias(self, expr):
        i = len(self._table_refs)

        key = self._get_table_key(expr)

        # Get total number of aliases up and down the tree at this point; if we
        # find the table prior-aliased along the way, however, we reuse that
        # alias
        ctx = self
        while ctx.parent is not None:
            ctx = ctx.parent

            if key in ctx._table_refs:
                alias = ctx._table_refs[key]
                self.set_ref(expr, alias)
                return

            i += len(ctx._table_refs)

        alias = 't{:d}'.format(i)
        self.set_ref(expr, alias)

    def need_aliases(self, expr=None):
        return self.always_alias or len(self._table_refs) > 1

    def has_ref(self, expr, parent_contexts=False):
        key = self._get_table_key(expr)
        return self._key_in(key, '_table_refs',
                            parent_contexts=parent_contexts)

    def set_ref(self, expr, alias):
        key = self._get_table_key(expr)
        self._table_refs[key] = alias

    def get_ref(self, expr):
        """
        Get the alias being used throughout a query to refer to a particular
        table or inline view
        """
        return self._get_table_item('_table_refs', expr)

    def is_extracted(self, expr):
        key = self._get_table_key(expr)
        return key in self.top_context.extracted_subexprs

    def set_extracted(self, expr):
        key = self._get_table_key(expr)
        self.extracted_subexprs.add(key)
        self.make_alias(expr)

    def subcontext(self):
        return type(self)(indent=self.indent, parent=self, params=self.params)

    # Maybe temporary hacks for correlated / uncorrelated subqueries

    def set_query(self, query):
        self.query = query

    def is_foreign_expr(self, expr):
        from ibis.expr.analysis import ExprValidator

        # The expression isn't foreign to us. For example, the parent table set
        # in a correlated WHERE subquery
        if self.has_ref(expr, parent_contexts=True):
            return False

        exprs = [self.query.table_set] + self.query.select_set
        validator = ExprValidator(exprs)
        return not validator.validate(expr)

    def _get_table_item(self, item, expr):
        key = self._get_table_key(expr)
        top = self.top_context

        if self.is_extracted(expr):
            return getattr(top, item).get(key)

        return getattr(self, item).get(key)

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            table = table.op()

        k = id(table)
        if k in self._table_key_memo:
            return self._table_key_memo[k]
        else:
            val = table._repr()
            self._table_key_memo[k] = val
            return val

    def _key_in(self, key, memo_attr, parent_contexts=False):
        if key in getattr(self, memo_attr):
            return True

        ctx = self
        while parent_contexts and ctx.parent is not None:
            ctx = ctx.parent
            if key in getattr(ctx, memo_attr):
                return True

        return False


class ExprTranslator(object):

    """Class that performs translation of ibis expressions into executable
    SQL.
    """

    _rewrites = {}

    context_class = QueryContext

    def __init__(self, expr, context, named=False, permit_subquery=False):
        self.expr = expr
        self.permit_subquery = permit_subquery

        assert context is not None, \
            'context is None in {}'.format(type(self).__name__)
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

    def get_result(self):
        """
        Build compiled SQL expression from the bottom up and return as a string
        """
        translated = self.translate(self.expr)
        if self._needs_name(self.expr):
            # TODO: this could fail in various ways
            name = self.expr.get_name()
            translated = self.name(translated, name)
        return translated

    def _needs_name(self, expr):
        if not self.named:
            return False

        op = expr.op()
        if isinstance(op, ops.TableColumn):
            # This column has been given an explicitly different name
            if expr.get_name() != op.name:
                return True
            return False

        if expr.get_name() is ir.unnamed:
            return False

        return True

    def translate(self, expr):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._rewrites:  # even if type(op) is in self._registry
            expr = self._rewrites[type(op)](expr)
            op = expr.op()

        # TODO: use op MRO for subclasses instead of this isinstance spaghetti
        if isinstance(op, ops.ScalarParameter):
            return self._trans_param(expr)
        elif isinstance(op, ops.TableNode):
            # HACK/TODO: revisit for more complex cases
            return '*'
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr)
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )

    def _trans_param(self, expr):
        raw_value = self.context.params[expr.op()]
        literal = ibis.literal(raw_value, type=expr.type())
        return self.translate(literal)

    @classmethod
    def rewrites(cls, klass, f=None):
        def decorator(f):
            cls._rewrites[klass] = f

        if f is None:
            return decorator
        else:
            decorator(f)
            return f

    @classmethod
    def compiles(cls, klass, f=None):
        def decorator(f):
            cls._registry[klass] = f

        if f is None:
            return decorator
        else:
            decorator(f)
            return f


rewrites = ExprTranslator.rewrites


@rewrites(analytics.Bucket)
def _bucket(expr):
    import operator

    op = expr.op()
    stmt = ibis.case()

    if op.closed == 'left':
        l_cmp = operator.le
        r_cmp = operator.lt
    else:
        l_cmp = operator.lt
        r_cmp = operator.le

    user_num_buckets = len(op.buckets) - 1

    bucket_id = 0
    if op.include_under:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else r_cmp
        else:
            cmp = operator.le if op.closed == 'right' else operator.lt
        stmt = stmt.when(cmp(op.arg, op.buckets[0]), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if (op.close_extreme and
            ((op.closed == 'right' and j == 0) or
             (op.closed == 'left' and j == (user_num_buckets - 1)))):
            stmt = stmt.when((lower <= op.arg) & (op.arg <= upper),
                             bucket_id)
        else:
            stmt = stmt.when(l_cmp(lower, op.arg) & r_cmp(op.arg, upper),
                             bucket_id)
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else l_cmp
        else:
            cmp = operator.lt if op.closed == 'right' else operator.le

        stmt = stmt.when(cmp(op.buckets[-1], op.arg), bucket_id)
        bucket_id += 1

    return stmt.end().name(expr._name)


@rewrites(analytics.CategoryLabel)
def _category_label(expr):
    op = expr.op()

    stmt = op.args[0].case()
    for i, label in enumerate(op.labels):
        stmt = stmt.when(i, label)

    if op.nulls is not None:
        stmt = stmt.else_(op.nulls)

    return stmt.end().name(expr._name)


@rewrites(ops.Any)
def _any_expand(expr):
    arg = expr.op().args[0]
    return arg.sum() > 0


@rewrites(ops.NotAny)
def _notany_expand(expr):
    arg = expr.op().args[0]
    return arg.sum() == 0


@rewrites(ops.All)
def _all_expand(expr):
    arg = expr.op().args[0]
    t = ir.find_base_table(arg)
    return arg.sum() == t.count()


@rewrites(ops.NotAll)
def _notall_expand(expr):
    arg = expr.op().args[0]
    t = ir.find_base_table(arg)
    return arg.sum() < t.count()


class Dialect(object):

    """Dialects encode the properties of a particular flavor of SQL.

    For example, quoting behavior is a property that should be encoded by
    ``Dialect``. Each backend has its own dialect.
    """

    translator = ExprTranslator

    @classmethod
    def make_context(cls, params=None):
        if params is None:
            params = {}
        params = {expr.op(): value for expr, value in params.items()}
        return cls.translator.context_class(dialect=cls(), params=params)


class Select(DML):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(self, table_set, select_set, context,
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

    @property
    def translator(self):
        return self.context.dialect.translator

    def _translate(self, expr, named=False, permit_subquery=False):
        context = self.context
        translator = self.translator(expr, context=context,
                                     named=named,
                                     permit_subquery=permit_subquery)
        return translator.get_result()

    def equals(self, other, cache=None):
        if cache is None:
            cache = {}

        if (self, other) in cache:
            return cache[(self, other)]

        if id(self) == id(other):
            cache[(self, other)] = True
            return True

        if not isinstance(other, Select):
            cache[(self, other)] = False
            return False

        this_exprs = self._all_exprs()
        other_exprs = other._all_exprs()

        if self.limit != other.limit:
            cache[(self, other)] = False
            return False

        for x, y in zip(this_exprs, other_exprs):
            if not x.equals(y):
                cache[(self, other)] = False
                return False

        cache[(self, other)] = True
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
        query = '\n'.join(filter(
            None,
            [
                with_frag,
                select_frag,
                from_frag,
                where_frag,
                groupby_frag,
                order_frag,
                limit_frag
            ]
        ))
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

    @property
    def table_set_formatter(self):
        return TableSetFormatter

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
            clause = 'GROUP BY {}'.format(', '.join([
                str(x + 1) for x in self.group_by]))
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


class TableSetFormatter(object):

    _join_names = {
        ops.InnerJoin: 'INNER JOIN',
        ops.LeftJoin: 'LEFT OUTER JOIN',
        ops.RightJoin: 'RIGHT OUTER JOIN',
        ops.OuterJoin: 'FULL OUTER JOIN',
        ops.LeftAntiJoin: 'LEFT ANTI JOIN',
        ops.LeftSemiJoin: 'LEFT SEMI JOIN',
        ops.CrossJoin: 'CROSS JOIN'
    }

    def __init__(self, parent, expr, indent=2):
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
            raise NotImplementedError('Do not support joins between '
                                      'joins yet')

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
            raise NotImplementedError('not allowing joins on right '
                                      'side yet')
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

            if (not isinstance(op, ops.Equals) and
                    not self._non_equijoin_supported):
                raise com.TranslationError('Non-equality join predicates, '
                                           'i.e. non-equijoins, are not '
                                           'supported')

    def _get_join_type(self, op):
        return self._join_names[type(op)]

    def _quote_identifier(self, name):
        return name

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
                    return '{} {}'.format(ctx.get_ref(ref_expr), alias)
                else:
                    return alias

            subquery = ctx.get_compiled_expr(expr)
            result = '(\n{}\n)'.format(util.indent(subquery, self.indent))
            is_subquery = True

        if is_subquery or ctx.need_aliases(expr):
            result += ' {}'.format(ctx.get_ref(expr))

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
        for jtype, table, preds in zip(self.join_types, self.join_tables[1:],
                                       self.join_predicates):
            buf.write('\n')
            buf.write(util.indent('{} {}'.format(jtype, table), self.indent))

            fmt_preds = []
            npreds = len(preds)
            for pred in preds:
                new_pred = self._translate(pred)
                if npreds > 1:
                    new_pred = '({})'.format(new_pred)
                fmt_preds.append(new_pred)

            if len(fmt_preds):
                buf.write('\n')

                conj = ' AND\n{}'.format(' ' * 3)
                fmt_preds = util.indent('ON ' + conj.join(fmt_preds),
                                        self.indent * 2)
                buf.write(fmt_preds)

        return buf.getvalue()
