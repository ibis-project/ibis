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

# An Ibis analytical expression will typically consist of a primary SELECT
# statement, with zero or more supporting DDL queries. For example we would
# want to support converting a text file in HDFS to a Parquet-backed Impala
# table, with optional teardown if the user wants the intermediate converted
# table to be temporary.

from collections import defaultdict

import ibis.expr.analysis as L
import ibis.expr.operations as ops
import ibis.expr.types as ir

from ibis.sql.context import QueryContext
import ibis.sql.ddl as ddl
import ibis.sql.transforms as transforms


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


#----------------------------------------------------------------------


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

        self.op_memo = set()

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
        self._analyze_filter_clauses()
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

    #----------------------------------------------------------------------
    # Filter analysis / rewrites

    def _analyze_filter_clauses(self):
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

        method = '_visit_filter_{}'.format(type(op).__name__)
        if hasattr(self, method):
            f = getattr(self, method)
            return f(expr)

        unchanged = True
        if isinstance(expr, ir.ScalarExpr):
            if expr.is_reduction():
                return self._rewrite_reduction_filter(expr)

        if isinstance(op, ops.BinaryOp):
            left = self._visit_filter(op.left)
            right = self._visit_filter(op.right)
            unchanged = left is op.left and right is op.right
            if not unchanged:
                return type(expr)(type(op)(left, right))
            else:
                return expr
        elif isinstance(op, (ops.Any, ops.Between, ops.Contains,
                             ops.TableColumn, ops.Literal)):
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

    #----------------------------------------------------------------------
    # Analysis of table set

    def _collect_elements(self):
        # If expr is a ValueExpr, we must seek out the TableExprs that it
        # references, build their ASTs, and mark them in our QueryContext

        # For now, we need to make the simplifying assumption that a value
        # expression that is being translated only depends on a single table
        # expression.

        source_table = self.query_expr

        # hm, is this the best place for this?
        root_op = source_table.op()
        if (isinstance(root_op, ops.Join) and
                not isinstance(root_op, ops.MaterializedJoin)):
            # Unmaterialized join
            source_table = source_table.materialize()

        self._collect(source_table, toplevel=True)

    def _collect(self, expr, toplevel=False):
        op = expr.op()
        method = '_collect_{}'.format(type(op).__name__)

        # Do not visit nodes twice
        if id(op) in self.op_memo:
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

        self.op_memo.add(id(op))

    def _collect_Aggregation(self, expr, toplevel=False):
        # The select set includes the grouping keys (if any), and these are
        # duplicated in the group_by set. SQL translator can decide how to
        # format these depending on the database. Most likely the
        # GROUP BY 1, 2, ... style
        if toplevel:
            subbed_expr = self._sub(expr)
            sub_op = subbed_expr.op()

            self.group_by = sub_op.by
            self.having = sub_op.having
            self.select_set = self.group_by + sub_op.agg_exprs
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

    #----------------------------------------------------------------------
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

        # Keep track of table expressions that we find in the query structure
        self.observed_exprs = {}

        # Maintain order of observation
        self.observed_keys = []

        self.expr_counts = defaultdict(lambda: 0)

    def get_result(self):
        self.visit(self.query.table_set)

        for clause in self.query.filters:
            self.visit(clause)

        to_extract = []

        # Read them inside-out, to avoid nested dependency issues
        for k in reversed(self.observed_keys):
            v = self.expr_counts[k]

            if self.greedy or v > 1:
                to_extract.append(self.observed_exprs[k])

        return to_extract

    def observe(self, expr):
        key = self._key(expr)

        if key not in self.expr_counts:
            self.observed_keys.append(key)

        self.observed_exprs[key] = expr
        self.expr_counts[key] += 1

    def _key(self, expr):
        return id(expr.op())

    def _has_been_observed(self, expr):
        return self._key(expr) in self.observed_exprs

    def visit(self, expr):
        node = expr.op()
        method = '_visit_{}'.format(type(node).__name__)

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
        self.query_roots = set([id(x) for x in qroots])

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
        return id(what) in self.query_roots


def _adapt_expr(expr):
    # Non-table expressions need to be adapted to some well-formed table
    # expression, along with a way to adapt the results to the desired
    # arity (whether array-like or scalar, for example)
    #
    # Canonical case is scalar values or arrays produced by some reductions
    # (simple reductions, or distinct, say)
    if isinstance(expr, ir.TableExpr):
        handler = lambda x: x
        return expr, handler

    if isinstance(expr, ir.ScalarExpr) and expr.is_reduction():
        table_expr = _reduction_to_aggregation(expr, agg_name='tmp')

        def scalar_handler(results):
            return results['tmp'][0]

        return table_expr, scalar_handler
    elif isinstance(expr, ir.ArrayExpr):
        op = expr.op()

        if isinstance(op, (ops.TableColumn, ops.DistinctArray)):
            table_expr = op.table

            def column_handler(results):
                return results[op.name]
            result_handler = column_handler
        else:
            # Something more complicated.
            base_table = L.find_source_table(expr)
            table_expr = base_table.projection([expr.name('tmp')])

            def projection_handler(results):
                return results['tmp']

            result_handler = projection_handler

        return table_expr, result_handler
    else:
        raise NotImplementedError


def _reduction_to_aggregation(expr, agg_name='tmp'):
    table = L.find_base_table(expr)
    return table.aggregate([expr.name(agg_name)])
