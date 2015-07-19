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

from collections import defaultdict

import ibis.common as com
import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.expr.types as ir

from ibis.sql.context import QueryContext
import ibis.sql.ddl as ddl
import ibis.sql.transforms as transforms
import ibis.util as util


def build_ast(expr, context=None):
    builder = QueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    ast = build_ast(expr, context)
    query = ast.queries[0]
    context = ast.context

    return query, context


def to_sql(expr, context=None):
    query, context = _get_query(expr, context)
    return query.compile(context)


# ---------------------------------------------------------------------


class QueryAST(object):

    def __init__(self, context, queries):
        self.context = context
        self.queries = queries


class QueryBuilder(object):

    def __init__(self, expr, context=None):
        self.expr = expr

        if context is None:
            context = QueryContext()

        self.context = context

    def get_result(self):
        op = self.expr.op()

        # TODO: any setup / teardown DDL statements will need to be done prior
        # to building the result set-generating statements.
        if isinstance(op, ops.Union):
            query = self._make_union()
        else:
            query = self._make_select()

        return QueryAST(self.context, [query])

    def _make_union(self):
        op = self.expr.op()
        return ddl.Union(op.left, op.right, distinct=op.distinct,
                         context=self.context)

    def _make_select(self):
        builder = SelectBuilder(self.expr, self.context)
        return builder.get_result()


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
        if len(self.queries) > 0:
            return self._wrap_result()

        # Generate other kinds of DDL statements that may be required to
        # execute the passed query. For example, loding
        setup_queries = self._generate_setup_queries()

        # Make DDL statements to be executed after the main primary select
        # statement(s)
        teardown_queries = self._generate_teardown_queries()

        select_query = self._build_result_query()

        self.queries.extend(setup_queries)
        self.queries.append(select_query)
        self.queries.extend(teardown_queries)

        return select_query

    def _generate_setup_queries(self):
        return []

    def _generate_teardown_queries(self):
        return []

    def _build_result_query(self):
        self._collect_elements()

        self._analyze_select_exprs()
        self._analyze_filter_exprs()
        self._analyze_subqueries()
        self._populate_context()

        return ddl.Select(self.context, self.table_set, self.select_set,
                          subqueries=self.subqueries,
                          where=self.filters,
                          group_by=self.group_by,
                          having=self.having,
                          limit=self.limit,
                          order_by=self.sort_by,
                          distinct=self.distinct,
                          result_handler=self.result_handler,
                          parent_expr=self.query_expr)

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
                if not isinstance(arg, ir.TableExpr):
                    continue
                self._make_table_aliases(arg)
        else:
            if not ctx.is_extracted(expr):
                ctx.make_alias(expr)

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

        if isinstance(op, ops.ValueNode):
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
            if ops.is_reduction(expr):
                return self._rewrite_reduction_filter(expr)

        if isinstance(op, ops.BinaryOp):
            left = self._visit_filter(op.left)
            right = self._visit_filter(op.right)
            unchanged = left is op.left and right is op.right
            if not unchanged:
                return type(expr)(type(op)(left, right))
            else:
                return expr
        elif isinstance(op, (ops.Any, ops.BooleanValueOp,
                             ops.TableColumn, ir.Literal)):
            return expr
        elif isinstance(op, ops.ValueNode):
            visited = [self._visit_filter(arg)
                       if isinstance(arg, ir.Expr) else arg
                       for arg in op.args]
            unchanged = True
            for new, old in zip(visited, op.args):
                if new is not old:
                    unchanged = False
            if not unchanged:
                return type(expr)(type(op)(*visited))
            else:
                return expr
        else:
            raise NotImplementedError(type(op))

    def _rewrite_reduction_filter(self, expr):
        # Find the table that this reduction references.

        # TODO: what about reductions that reference a join that isn't visible
        # at this level? Means we probably have the wrong design, but will have
        # to revisit when it becomes a problem.
        aggregation = _reduction_to_aggregation(expr, agg_name='tmp')
        return aggregation.to_array()

    def _visit_filter_Any(self, expr):
        # Rewrite semi/anti-join predicates in way that can hook into SQL
        # translation step
        transform = transforms.AnyToExistsTransform(self.context, expr,
                                                    self.table_set)
        return transform.get_result()
    _visit_filter_NotAny = _visit_filter_Any

    def _visit_filter_TopK(self, expr):
        # Top K is rewritten as an
        # - aggregation
        # - sort by
        # - limit
        # - left semi join with table set

        metric_name = '__tmp__'

        op = expr.op()

        metrics = [op.by.name(metric_name)]
        rank_set = (self.table_set.aggregate(metrics, by=[op.arg])
                    .sort_by([(metric_name, False)])
                    .limit(op.k))

        pred = (op.arg == getattr(rank_set, op.arg.get_name()))
        self.table_set = self.table_set.semi_join(rank_set, [pred])

        return None

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
            if isinstance(root_op, ir.ExpressionList):
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
        elif isinstance(op, (ops.Join, ops.MaterializedJoin)):
            self._collect_Join(expr, toplevel=toplevel)
        else:
            raise NotImplementedError(type(op))

        self.op_memo.add(op)

    def _collect_Aggregation(self, expr, toplevel=False):
        # The select set includes the grouping keys (if any), and these are
        # duplicated in the group_by set. SQL translator can decide how to
        # format these depending on the database. Most likely the
        # GROUP BY 1, 2, ... style
        if toplevel:
            subbed_expr = self._sub(expr)
            sub_op = subbed_expr.op()

            self.group_by = range(len(sub_op.by))
            self.having = sub_op.having
            self.select_set = sub_op.by + sub_op.agg_exprs
            self.table_set = sub_op.table

            self._collect(expr.op().table)

    def _collect_Distinct(self, expr, toplevel=False):
        if toplevel:
            self.distinct = True

        self._collect(expr.op().table, toplevel=toplevel)

    def _collect_Filter(self, expr, toplevel=False):
        op = expr.op()

        self.filters.extend(op.predicates)
        if toplevel:
            self.select_set = [op.table]
            self.table_set = op.table

        self._collect(op.table)

    def _collect_Limit(self, expr, toplevel=False):
        if not toplevel:
            return

        op = expr.op()
        self.limit = {
            'n': op.n,
            'offset': op.offset
        }
        self._collect(op.table, toplevel=toplevel)

    def _collect_SortBy(self, expr, toplevel=False):
        op = expr.op()

        self.sort_by = op.keys
        if toplevel:
            # HACK: yuck, need a better way to know if we should perform a
            # select * from a subquery here
            if not isinstance(op.table.op(), ops.Aggregation):
                self.select_set = [op.table]
                self.table_set = op.table
                toplevel = False

        self._collect(op.table, toplevel=toplevel)

    def _collect_Join(self, expr, toplevel=False):
        op = expr.op()

        if isinstance(op, ops.MaterializedJoin):
            expr = op.join
            op = expr.op()

        if toplevel:
            subbed = self._sub(expr)
            self.table_set = subbed
            self.select_set = [op.left, op.right]

        self._collect(op.left, toplevel=False)
        self._collect(op.right, toplevel=False)

    def _collect_Union(self, expr, toplevel=False):
        if not toplevel:
            return
        else:
            raise NotImplementedError

    def _collect_Projection(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            subbed = self._sub(expr)
            sop = subbed.op()

            self.select_set = sop.selections
            self.table_set = sop.table
            self._collect(op.table)

    def _collect_PhysicalTable(self, expr, toplevel=False):
        if toplevel:
            self.select_set = [expr]
            self.table_set = self._sub(expr)

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

        self.expr_counts = defaultdict(lambda: 0)

    def get_result(self):
        if self.query.table_set is not None:
            self.visit(self.query.table_set)

        for clause in self.query.filters:
            self.visit(clause)

        to_extract = []

        # Read them inside-out, to avoid nested dependency issues
        for expr, key in reversed(zip(self.observed_exprs.keys,
                                      self.observed_exprs.values)):
            v = self.expr_counts[key]

            if self.greedy or v > 1:
                to_extract.append(expr)

        return to_extract

    def observe(self, expr):
        if expr in self.observed_exprs:
            key = self.observed_exprs.get(expr)
        else:
            # this key only needs to be unique because of the IbisMap
            key = id(expr.op())
            self.observed_exprs.set(expr, key)

        self.expr_counts[key] += 1

    def _has_been_observed(self, expr):
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
        elif isinstance(node, ops.ValueNode):
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
    _visit_ExistsSubquery = _extract_noop
    _visit_NotExistsSubquery = _extract_noop

    def _visit_Aggregation(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_Distinct(self, expr):
        self.observe(expr)

    def _visit_Filter(self, expr):
        self.visit(expr.op().table)

    def _visit_Limit(self, expr):
        self.visit(expr.op().table)

    def _visit_Union(self, expr):
        self.observe(expr)

    def _visit_Projection(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_SQLQueryResult(self, expr):
        self.observe(expr)

    def _visit_TableColumn(self, expr):
        table = expr.op().table
        if not self._has_been_observed(table):
            self.visit(table)

    def _visit_SelfReference(self, expr):
        self.visit(expr.op().table)

    def _visit_SortBy(self, expr):
        self.observe(expr)
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

    def _visit(self, expr, in_subquery=False):
        node = expr.op()

        in_subquery = self._is_subquery(node)

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg, in_subquery=in_subquery)
            elif isinstance(arg, ir.Expr):
                self._visit(arg, in_subquery=in_subquery)
            else:
                continue

    def _is_subquery(self, node):
        # XXX
        if isinstance(node, ops.TableArrayView):
            return True

        if isinstance(node, ops.TableColumn):
            return not self._is_root(node.table)

        return False

    def _visit_table(self, expr, in_subquery=False):
        node = expr.op()

        if isinstance(node, (ops.PhysicalTable, ops.SelfReference)):
            self._ref_check(node, in_subquery=in_subquery)

        for arg in node.flat_args():
            if isinstance(arg, ir.Expr):
                self._visit(arg, in_subquery=in_subquery)

    def _ref_check(self, node, in_subquery=False):
        is_aliased = self.ctx.has_alias(node)

        if self._is_root(node):
            if in_subquery:
                self.has_query_root = True
        else:
            if in_subquery:
                self.has_foreign_root = True
                if (not is_aliased and
                        self.ctx.has_alias(node, parent_contexts=True)):
                    self.ctx.make_alias(node)

            elif not self.ctx.has_alias(node):
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

    def _scalar_reduce(x):
        return isinstance(x, ir.ScalarExpr) and ops.is_reduction(x)

    if isinstance(expr, ir.ScalarExpr):
        def scalar_handler(results):
            return results['tmp'][0]

        if _scalar_reduce(expr):
            table_expr = _reduction_to_aggregation(expr, agg_name='tmp')
            return table_expr, scalar_handler
        else:
            base_table = L.find_base_table(expr)
            if base_table is None:
                # expr with no table refs
                return expr.name('tmp'), scalar_handler
            else:
                raise NotImplementedError(expr._repr())

    elif isinstance(expr, ir.ExprList):
        exprs = expr.exprs()

        is_aggregation = True
        any_aggregation = False

        for x in exprs:
            if not _scalar_reduce(x):
                is_aggregation = False
            else:
                any_aggregation = True

        if is_aggregation:
            table = L.find_base_table(exprs[0])
            return table.aggregate(exprs), as_is
        elif not any_aggregation:
            return expr, as_is
        else:
            raise NotImplementedError(expr._repr())

    elif isinstance(expr, ir.ArrayExpr):
        op = expr.op()

        def _get_column(name):
            def column_handler(results):
                return results[name]
            return column_handler

        if isinstance(op, ops.TableColumn):
            table_expr = op.table
            result_handler = _get_column(op.name)
        else:
            # Something more complicated.
            base_table = L.find_source_table(expr)

            if isinstance(op, ops.DistinctArray):
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
        raise NotImplementedError


def _reduction_to_aggregation(expr, agg_name='tmp'):
    table = L.find_base_table(expr)
    return table.aggregate([expr.name(agg_name)])
