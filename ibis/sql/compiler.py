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

import ibis.expr.base as ir
import ibis.util as util

from ibis.sql.context import QueryContext
from ibis.sql.select import Select


def build_ast(expr):
    builder = SelectBuilder(expr)
    return builder.get_result()


def _get_query(expr):
    ast = build_ast(expr)
    return ast.queries[0]


def to_sql(expr, context=None):
    query = _get_query(expr)
    return query.compile(context=context)


#----------------------------------------------------------------------


class QueryAST(object):

    def __init__(self, context, queries):
        self.context = context
        self.queries = queries


class SelectBuilder(object):

    """
    Transforms expression IR to a query pipeline (potentially multiple
    queries). There will typically be a primary SELECT query, perhaps with some
    subqueries and other DDL to ingest and tear down intermediate data sources.

    Walks the expression tree and catalogues distinct query units, builds
    select statements (and other DDL types, where necessary), and records
    relevant query unit aliases to be used when actually generating SQL.
    """

    def __init__(self, expr, context=None):
        self.expr = expr

        self.query_expr, self.result_handler = _adapt_expr(self.expr)

        if context is None:
            context = QueryContext()

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

        select_query = self._build_select()

        self.queries.extend(setup_queries)
        self.queries.append(select_query)
        self.queries.extend(teardown_queries)

        return self._wrap_result()

    def _wrap_result(self):
        return QueryAST(self.context, self.queries)

    def _build_select(self):
        self._collect_elements()

        # What's semantically contained in the filter predicates may need to be
        # rewritten. Not sure if this is the right place to do this, but a
        # starting point
        self._analyze_where_clauses()

        return Select(self.table_set, self.select_set,
                      where=self.filters,
                      group_by=self.group_by,
                      having=self.having,
                      limit=self.limit,
                      order_by=self.sort_by,
                      result_handler=self.result_handler)

    #----------------------------------------------------------------------
    # Filter analysis / rewrites

    def _analyze_where_clauses(self):
        # Various kinds of semantically valid WHERE clauses may need to be
        # rewritten into a form that we can actually translate into valid SQL.
        new_where = []
        for expr in self.filters:
            new_expr = self._visit_where(expr)

            # Transformations may result in there being no outputted filter
            # predicate
            if new_expr is not None:
                new_where.append(new_expr)

        self.filters = new_where

    def _visit_where(self, expr):
        # Dumping ground for analysis of WHERE expressions
        # - Subquery extraction
        # - Conversion to explicit semi/anti joins
        # - Rewrites to generate subqueries

        op = expr.op()

        method = '_visit_{}'.format(type(op).__name__)
        if hasattr(self, method):
            f = getattr(self, method)
            return f(expr)

        unchanged = True
        if isinstance(expr, ir.ScalarExpr):
            if expr.is_reduction():
                return self._rewrite_reduction(expr)

        if isinstance(op, ir.BinaryOp):
            left = self._visit_where(op.left)
            right = self._visit_where(op.right)
            unchanged = left is op.left and right is op.right
            if not unchanged:
                return type(expr)(type(op)(left, right))
            else:
                return expr
        elif isinstance(op, (ir.Between, ir.Contains,
                             ir.TableColumn, ir.Literal)):
            return expr
        else:
            raise NotImplementedError(type(op))

    def _rewrite_reduction(self, expr):
        # Find the table that this reduction references.

        # TODO: what about reductions that reference a join that isn't visible
        # at this level? Means we probably have the wrong design, but will have
        # to revisit when it becomes a problem.
        aggregation = _reduction_to_aggregation(expr, agg_name='tmp')
        return aggregation.to_array()

    def _visit_TopK(self, expr):
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
        filtered = self.table_set.semi_join(rank_set, [pred])

        # Now, fix up the now broken select set. Is this necessary?
        # new_select_set = []
        # for x in self.select_set:
        #     new_expr = ir.sub_for(x, [(self.table_set, filtered)])
        #     new_select_set.append(new_expr)
        # self.select_set = new_select_set

        self.table_set = filtered

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
        if (isinstance(root_op, ir.Join) and
            not isinstance(root_op, ir.MaterializedJoin)):
            # Unmaterialized join
            source_table = source_table.materialize()

        self._visit(source_table, toplevel=True)

    def _visit(self, expr, toplevel=False):
        op = expr.op()
        method = '_visit_{}'.format(type(op).__name__)

        # Do not visit nodes twice
        if id(op) in self.op_memo:
            return

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr, toplevel=toplevel)
        elif isinstance(op, (ir.PhysicalTable, ir.SQLQueryResult)):
            self._visit_PhysicalTable(expr, toplevel=toplevel)
        elif isinstance(op, ir.Join):
            self._visit_Join(expr, toplevel=toplevel)
        else:
            raise NotImplementedError(type(op))

        self.op_memo.add(id(op))

    def _visit_Aggregation(self, expr, toplevel=False):
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

            self._visit(expr.op().table)

    def _sub(self, what):
        if isinstance(what, list):
            return [ir.substitute_parents(x, self.sub_memo) for x in what]
        else:
            return ir.substitute_parents(what, self.sub_memo)

    def _visit_Filter(self, expr, toplevel=False):
        op = expr.op()

        self.filters.extend(op.predicates)
        if toplevel:
            self.select_set = [op.table]
            self.table_set = op.table

        self._visit(op.table)

    def _visit_Limit(self, expr, toplevel=False):
        if not toplevel:
            return

        op = expr.op()
        self.limit = {
            'n': op.n,
            'offset': op.offset
        }
        self._visit(op.table, toplevel=toplevel)

    def _visit_Join(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            subbed = self._sub(expr)
            self.table_set = subbed
            self.select_set = [op.left, op.left]

        self._visit(op.left, toplevel=toplevel)
        self._visit(op.right, toplevel=toplevel)

    def _visit_Projection(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            subbed = self._sub(expr)
            sop = subbed.op()

            self.select_set = sop.selections
            self.table_set = sop.table
            self._visit(op.table)

    def _visit_PhysicalTable(self, expr, toplevel=False):
        if toplevel:
            self.select_set = [expr]
            self.table_set = self._sub(expr)

    def _visit_SelfReference(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            self._visit(op.table, toplevel=toplevel)

    def _visit_SortBy(self, expr, toplevel=False):
        op = expr.op()
        if toplevel:
            self.sort_by = op.keys
            self._visit(op.table, toplevel=toplevel)

    def _generate_setup_queries(self):
        return []

    def _generate_teardown_queries(self):
        return []



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

        if isinstance(op, ir.TableColumn):
            table_expr = op.table
            def column_handler(results):
                return results[op.name]
            result_handler = column_handler
        else:
            # Something more complicated.
            base_table = _find_source_table(expr)
            table_expr = base_table.projection([expr.name('tmp')])
            def projection_handler(results):
                return results['tmp']

            result_handler = projection_handler

        return table_expr, result_handler
    else:
        raise NotImplementedError


def _reduction_to_aggregation(expr, agg_name='tmp'):
    table = _find_base_table(expr)
    return table.aggregate([expr.name(agg_name)])


def _find_base_table(expr):
    if isinstance(expr, ir.TableExpr):
        return expr

    for arg in expr.op().args:
        if isinstance(arg, ir.Expr):
            r = _find_base_table(arg)
            if isinstance(r, ir.TableExpr):
                return r


def _find_source_table(expr):
    # A more complex version of _find_base_table.
    # TODO: Revisit/refactor this all at some point
    node = expr.op()

    # First table expression observed for each argument that the expr
    # depends on
    first_tables = []
    def push_first(arg):
        if isinstance(arg, (tuple, list)):
            [push_first(x) for x in arg]
            return

        if not isinstance(arg, ir.Expr):
            return
        if isinstance(arg, ir.TableExpr):
            first_tables.append(arg)
        else:
            collect(arg.op())

    def collect(node):
        for arg in node.args:
            push_first(arg)

    collect(node)
    options = util.unique_by_key(first_tables, id)

    if len(options) > 1:
        raise NotImplementedError

    return options[0]
