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

from ibis.compat import lzip
import ibis.common as com
import ibis.expr.analysis as L
import ibis.expr.analytics as analytics

import ibis.expr.operations as ops
import ibis.expr.types as ir

import ibis.sql.transforms as transforms
import ibis.util as util
import ibis


# ---------------------------------------------------------------------


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

        # GH #667; this may reference a filtered version of self.table_set
        arg = L.substitute_parents(op.arg)

        pred = (arg == getattr(rank_set, op.arg.get_name()))
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
            # Expressions not depending on any table
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
        if not toplevel:
            return
        else:
            raise NotImplementedError

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
            self.select_set = sub_op.by + sub_op.agg_exprs
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

    def _walk(expr):
        op = expr.op()
        if isinstance(op, ops.Join):
            _walk(op.left)
            _walk(op.right)
        else:
            subtables.append(expr)
    _walk(expr)

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

        self.expr_counts = defaultdict(lambda: 0)

    def get_result(self):
        if self.query.table_set is not None:
            self.visit(self.query.table_set)

        for clause in self.query.filters:
            self.visit(clause)

        to_extract = []

        # Read them inside-out, to avoid nested dependency issues
        for expr, key in reversed(lzip(self.observed_exprs.keys,
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

    def _visit_Exists(self, expr):
        node = expr.op()
        self.visit(node.foreign_table)
        for pred in node.predicates:
            self.visit(pred)

    _visit_ExistsSubquery = _visit_Exists
    _visit_NotExistsSubquery = _visit_Exists

    def _visit_Aggregation(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_Distinct(self, expr):
        self.observe(expr)

    def _visit_Limit(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_Union(self, expr):
        self.observe(expr)

    def _visit_MaterializedJoin(self, expr):
        self.observe(expr)
        self.visit(expr.op().join)

    def _visit_Selection(self, expr):
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

        in_subquery = in_subquery or self._is_subquery(node)

        for arg in node.flat_args():
            if isinstance(arg, ir.TableExpr):
                self._visit_table(arg, in_subquery=in_subquery)
            elif isinstance(arg, ir.Expr):
                self._visit(arg, in_subquery=in_subquery)
            else:
                continue

    def _is_subquery(self, node):
        # XXX
        if isinstance(node, (ops.TableArrayView,
                             transforms.ExistsSubquery,
                             transforms.NotExistsSubquery)):
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

        if L.is_scalar_reduce(expr):
            table_expr, name = L.reduction_to_aggregation(
                expr, default_name='tmp')
            return table_expr, _get_scalar(name)
        else:
            base_table = ir.find_base_table(expr)
            if base_table is None:
                # expr with no table refs
                return expr.name('tmp'), _get_scalar('tmp')
            else:
                raise NotImplementedError(expr._repr())

    elif isinstance(expr, ir.AnalyticExpr):
        return expr.to_aggregation(), as_is

    elif isinstance(expr, ir.ExprList):
        exprs = expr.exprs()

        is_aggregation = True
        any_aggregation = False

        for x in exprs:
            if not L.is_scalar_reduce(x):
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

    elif isinstance(expr, ir.ArrayExpr):
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
        raise com.TranslationError('Do not know how to execute: {0}'
                                   .format(type(expr)))


class QueryBuilder(object):

    select_builder = SelectBuilder

    def __init__(self, expr, context=None):
        self.expr = expr

        if context is None:
            context = self._make_context()

        self.context = context

    @property
    def _make_context(self):
        raise NotImplementedError

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
        return self._union_class(op.left, op.right,
                                 distinct=op.distinct,
                                 context=self.context)

    def _make_select(self):
        builder = self.select_builder(self.expr, self.context)
        return builder.get_result()


class QueryContext(object):

    """
    Records bits of information used during ibis AST to SQL translation
    """

    def __init__(self, indent=2, parent=None):
        self._table_refs = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent
        self.parent = parent

        self.always_alias = False

        self.query = None

        self._table_key_memo = {}

    def _compile_subquery(self, expr, isolated=False):
        sub_ctx = self.subcontext(isolated=isolated)
        return self._to_sql(expr, sub_ctx)

    def _to_sql(self, expr, ctx):
        raise NotImplementedError

    @property
    def top_context(self):
        if self.parent is None:
            return self
        else:
            return self.parent.top_context

    def set_always_alias(self):
        self.always_alias = True

    def get_compiled_expr(self, expr, isolated=False):
        this = self.top_context

        key = self._get_table_key(expr)
        if key in this.subquery_memo:
            return this.subquery_memo[key]

        op = expr.op()
        if isinstance(op, ops.SQLQueryResult):
            result = op.query
        else:
            result = self._compile_subquery(expr, isolated=isolated)

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

        alias = 't%d' % i
        self.set_ref(expr, alias)

    def need_aliases(self):
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

    def subcontext(self, isolated=False):
        if not isolated:
            return type(self)(indent=self.indent, parent=self)
        else:
            return type(self)(indent=self.indent)

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

    _rewrites = {}

    def __init__(self, expr, context=None, named=False, permit_subquery=False):
        self.expr = expr
        self.permit_subquery = permit_subquery

        if context is None:
            context = self._context_class()
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

    @property
    def _context_class(self):
        raise NotImplementedError

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

        if type(op) in self._rewrites and type(op) not in self._registry:
            expr = self._rewrites[type(op)](expr)
            op = expr.op()

        # TODO: use op MRO for subclasses instead of this isinstance spaghetti
        if isinstance(op, ir.Parameter):
            return self._trans_param(expr)
        elif isinstance(op, ops.TableNode):
            # HACK/TODO: revisit for more complex cases
            return '*'
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr)
        else:
            raise com.TranslationError('No translator rule for {0}'
                                       .format(type(op)))

    def _trans_param(self, expr):
        raise NotImplementedError

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


class DDL(object):
    pass


class Select(DDL):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(self, table_set, select_set,
                 subqueries=None, where=None, group_by=None, having=None,
                 order_by=None, limit=None,
                 distinct=False, indent=2,
                 result_handler=None, parent_expr=None,
                 context=None):
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

    translator = None

    def _translate(self, expr, context=None, named=False,
                   permit_subquery=False):

        if context is None:
            context = self.context

        translator = self.translator(expr, context=context,
                                     named=named,
                                     permit_subquery=permit_subquery)
        return translator.get_result()

    def equals(self, other):
        if not isinstance(other, Select):
            return False

        this_exprs = self._all_exprs()
        other_exprs = other._all_exprs()

        if self.limit != other.limit:
            return False

        for x, y in zip(this_exprs, other_exprs):
            if not x.equals(y):
                return False

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


class TableSetFormatter(object):

    def __init__(self, parent, expr, indent=2):
        self.parent = parent
        self.context = parent.context
        self.expr = expr
        self.indent = indent

        self.join_tables = []
        self.join_types = []
        self.join_predicates = []

    def _translate(self, expr):
        return self.parent._translate(expr, context=self.context)

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


class Union(DDL):

    def __init__(self, left_table, right_table, distinct=False,
                 context=None):
        self.context = context
        self.left = left_table
        self.right = right_table

        self.distinct = distinct
