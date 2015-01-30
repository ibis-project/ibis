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
from io import BytesIO

import ibis.expr.base as ir
import ibis.common as com
import ibis.util as util

from ibis.sql.exprs import ExprTranslator


def build_ast(expr):
    builder = QueryASTBuilder(expr)
    return builder.get_result()


def translate_expr(expr, context=None, named=False):
    translator = ExprTranslator(expr, context=context, named=named)
    return translator.get_result()


def _get_query(expr):
    ast = build_ast(expr)
    return ast.queries[0]


def to_sql(expr, context=None):
    query = _get_query(expr)
    return query.compile(context=context)


#----------------------------------------------------------------------
# The QueryContext (temporary name) will store useful information like table
# alias names for converting value expressions to SQL.


class QueryContext(object):

    """

    """

    def __init__(self, indent=2):
        self.table_aliases = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent

    @property
    def top_context(self):
        return self

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            table = table.op()
        return id(table)

    def is_extracted(self, expr):
        key = self._get_table_key(expr)
        return key in self.top_context.extracted_subexprs

    def set_extracted(self, expr):
        key = self._get_table_key(expr)
        self.extracted_subexprs.add(key)
        self.make_alias(expr)

    def get_formatted_query(self, expr):
        this = self.top_context

        key = self._get_table_key(expr)
        if key in this.subquery_memo:
            return this.subquery_memo[key]

        result = to_sql(expr, context=self.subcontext())
        this.subquery_memo[key] = result
        return result

    def make_alias(self, table_expr):
        i = len(self.table_aliases)
        alias = 't%d' % i
        self.set_alias(table_expr, alias)

    def has_alias(self, table_expr):
        key = self._get_table_key(table_expr)
        return key in self.table_aliases

    def need_aliases(self):
        return len(self.table_aliases) > 1

    def set_alias(self, table_expr, alias):
        key = self._get_table_key(table_expr)
        self.table_aliases[key] = alias

    def get_alias(self, table_expr):
        """
        Get the alias being used throughout a query to refer to a particular
        table or inline view
        """
        key = self._get_table_key(table_expr)

        top = self.top_context
        if self is top:
            if self.is_extracted(table_expr):
                return top.table_aliases.get(key)

        return self.table_aliases.get(key)

    def subcontext(self):
        return SubContext(self)


class SubContext(QueryContext):

    def __init__(self, parent):
        self.parent = parent
        super(SubContext, self).__init__(indent=parent.indent)

    @property
    def top_context(self):
        return self.parent.top_context


#----------------------------------------------------------------------


class Select(object):

    """
    A SELECT statement which, after execution, might yield back to the user a
    table, array/list, or scalar value, depending on the expression that
    generated it
    """

    def __init__(self, table_set, select_set, where=None, group_by=None,
                 order_by=None, limit=None, having=None,
                 parent_expr=None, indent=2):
        self.select_set = select_set
        self.table_set = table_set

        self.where = where or []

        # Group keys and post-predicates for aggregations
        self.group_by = group_by or []
        self.having = having or []
        self.order_by = order_by or []

        self.limit = limit
        self.parent_expr = parent_expr
        self.subqueries = []

        self.indent = indent

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

    def compile(self, context=None, semicolon=False):
        """
        This method isn't yet idempotent; calling multiple times may yield
        unexpected results
        """
        if context is None:
            context = QueryContext()

        self._extract_subqueries(context)

        self.populate_context(context)

        # If any subqueries, translate them and add to beginning of query as
        # part of the WITH section
        with_frag = self.format_subqueries(context)

        # SELECT
        select_frag = self.format_select_set(context)

        # FROM, JOIN, UNION
        from_frag = self.format_table_set(context)

        # WHERE
        where_frag = self.format_where(context)

        # GROUP BY and HAVING
        groupby_frag = self.format_group_by(context)

        # ORDER BY and LIMIT
        order_frag = self.format_postamble(context)

        # Glue together the query fragments and return
        query = _join_not_none('\n', [with_frag, select_frag, from_frag,
                                      where_frag, groupby_frag, order_frag])

        return query

    def populate_context(self, context):
        # Populate aliases for the distinct relations used to output this
        # select statement.

        # Urgh, hack for now
        op = self.table_set.op()

        if isinstance(op, ir.Join):
            roots = self.table_set._root_tables()
            for table in roots:
                if context.is_extracted(table):
                    continue
                context.make_alias(table)
        else:
            context.make_alias(self.table_set)

    def _extract_subqueries(self, context):
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
        self.subqueries = _extract_subqueries(self)
        for expr in self.subqueries:
            context.set_extracted(expr)

    def format_subqueries(self, context):
        if len(self.subqueries) == 0:
            return

        buf = BytesIO()
        buf.write('WITH ')

        for i, expr in enumerate(self.subqueries):
            if i > 0:
                buf.write(',\n')
            formatted = util.indent(context.get_formatted_query(expr), 2)
            alias = context.get_alias(expr)
            buf.write('{} AS (\n{}\n)'.format(alias, formatted))

        return buf.getvalue()

    def format_select_set(self, context):
        # TODO:
        formatted = []
        for expr in self.select_set:
            if isinstance(expr, ir.ValueExpr):
                expr_str = translate_expr(expr, context=context, named=True)
            elif isinstance(expr, ir.TableExpr):
                # A * selection, possibly prefixed
                if context.need_aliases():
                    expr_str = '{}.*'.format(context.get_alias(expr))
                else:
                    expr_str = '*'
            formatted.append(expr_str)

        buf = BytesIO()
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
                if i: buf.write(',')
                buf.write(' ')
                buf.write(val)
                tokens += 1
                line_length += len(val) + 2

        return 'SELECT{}'.format(buf.getvalue())

    def format_table_set(self, ctx):
        fragment = 'FROM '

        helper = _TableSetFormatter(ctx, self.table_set)
        fragment += helper.get_result()

        return fragment

    def format_group_by(self, context):
        if len(self.group_by) == 0:
            # There is no aggregation, nothing to see here
            return None

        # Verify that the group by exprs match the first few tokens in the
        # select set
        for i, expr in enumerate(self.group_by):
            if expr is not self.select_set[i]:
                raise com.InternalError('Select was improperly formed')

        lines = []
        if len(self.group_by) > 0:
            clause = 'GROUP BY {}'.format(', '.join([
                str(x + 1) for x in range(len(self.group_by))]))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = translate_expr(expr, context=context)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_where(self, context):
        if len(self.where) == 0:
            return None

        buf = BytesIO()
        buf.write('WHERE ')
        fmt_preds = [translate_expr(pred, context=context)
                     for pred in self.where]
        conj = ' AND\n{}'.format(' ' * 6)
        buf.write(conj.join(fmt_preds))
        return buf.getvalue()

    def format_postamble(self, context):
        buf = BytesIO()
        lines = 0

        if len(self.order_by) > 0:
            buf.write('ORDER BY ')
            formatted = []
            for key in self.order_by:
                translated = translate_expr(key.expr, context=context)
                if not key.ascending:
                    translated += ' DESC'
                formatted.append(translated)
            buf.write(', '.join(formatted))
            lines += 1

        if self.limit is not None:
            if lines:
                buf.write('\n')
            n, offset = self.limit['n'], self.limit['offset']
            buf.write('LIMIT {}'.format(n))
            if offset is not None:
                buf.write(' OFFSET {}'.format(offset))
            lines += 1

        if not lines:
            return None

        return buf.getvalue()

    def adapt_result(self, result):
        if isinstance(self.parent_expr, ir.TableExpr):
            result_type = 'table'
        elif isinstance(self.parent_expr, ir.ArrayExpr):
            result_type = 'array'
        elif isinstance(self.parent_expr, ir.ScalarExpr):
            aresult_type = 'scalar'
        pass


class _TableSetFormatter(object):
    _join_names = {
        ir.InnerJoin: 'INNER JOIN',
        ir.LeftJoin: 'LEFT OUTER JOIN',
        ir.RightJoin: 'RIGHT OUTER JOIN',
        ir.OuterJoin: 'FULL OUTER JOIN',
        ir.LeftAntiJoin: 'LEFT ANTI JOIN',
        ir.LeftSemiJoin: 'LEFT SEMI JOIN',
        ir.CrossJoin: 'CROSS JOIN'
    }

    def __init__(self, context, expr, indent=2):
        self.context = context
        self.expr = expr
        self.indent = indent

        self.join_tables = []
        self.join_types = []
        self.join_predicates = []

    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ir.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        # TODO: Now actually format the things
        buf = BytesIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(self.join_types, self.join_tables[1:],
                                       self.join_predicates):
            buf.write('\n')
            buf.write(util.indent('{} {}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = [translate_expr(pred, context=self.context)
                             for pred in preds]
                conj = ' AND\n{}'.format(' ' * 3)
                fmt_preds = util.indent('ON ' + conj.join(fmt_preds),
                                        self.indent * 2)
                buf.write(fmt_preds)

        return buf.getvalue()

    def _walk_join_tree(self, op):
        left = op.left.op()
        right = op.right.op()

        if util.all_of([left, right], ir.Join):
            raise NotImplementedError('Do not support joins between '
                                      'joins yet')

        self._validate_join_predicates(op.predicates)

        jname = self._join_names[type(op)]

        # Impala requires this
        if len(op.predicates) == 0:
            jname = self._join_names[ir.CrossJoin]

        # Read off tables and join predicates left-to-right in
        # depth-first order
        if isinstance(left, ir.Join):
            self._walk_join_tree(left)
            self.join_tables.append(self._format_table(op.right))
            self.join_types.append(jname)
            self.join_predicates.append(op.predicates)
        elif isinstance(right, ir.Join):
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

    def _format_table(self, expr):
        return _format_table(self.context, expr)

    # Placeholder; revisit when supporting other databases
    _non_equijoin_supported = False
    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()

            if (not isinstance(op, ir.Equals) and
                not self._non_equijoin_supported):
                raise com.TranslationError(
                    'Non-equality join predicates, '
                    'i.e. non-equijoins, are not supported')


def _format_table(ctx, expr, indent=2):
    # TODO: This could probably go in a class and be significantly nicer

    ref_expr = expr
    op = ref_op = expr.op()
    if isinstance(op, ir.SelfReference):
        ref_expr = op.table
        ref_op = ref_expr.op()

    if isinstance(ref_op, ir.PhysicalTable):
        name = op.name
        if name is None:
            raise com.RelationError('Table did not have a name: {!r}'
                                    .format(expr))
        result = name
    else:
        # A subquery
        if ctx.is_extracted(ref_expr):
            # Was put elsewhere, e.g. WITH block, we just need to grab its
            # alias
            alias = ctx.get_alias(expr)

            # HACK: self-references have to be treated more carefully here
            if isinstance(op, ir.SelfReference):
                return '{} {}'.format(ctx.get_alias(ref_expr), alias)
            else:
                return alias

        subquery = ctx.get_formatted_query(expr)
        result = '(\n{}\n)'.format(util.indent(subquery, indent))

    if ctx.need_aliases():
        result += ' {}'.format(ctx.get_alias(expr))

    return result


def _extract_subqueries(select_stmt):
    helper = _ExtractSubqueries(select_stmt)
    return helper.get_result()


class _ExtractSubqueries(object):

    # Helper class to make things a little easier

    def __init__(self, query, greedy=False):
        self.query = query
        self.greedy = greedy

        # Keep track of table expressions that we find in the query structure
        self.observed_exprs = {}
        self.expr_counts = defaultdict(lambda: 0)

    def get_result(self):
        self.visit(self.query.table_set)

        to_extract = []
        for k, v in self.expr_counts.items():
            if self.greedy or v > 1:
                to_extract.append(self.observed_exprs[k])

        return to_extract

    def observe(self, expr):
        key = id(expr.op())
        self.observed_exprs[key] = expr
        self.expr_counts[key] += 1

    def visit(self, expr):
        node = expr.op()
        method = '_visit_{}'.format(type(node).__name__)

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr)
        elif isinstance(node, ir.Join):
            self._visit_join(expr)
        elif isinstance(node, ir.PhysicalTable):
            self._visit_physical_table(expr)
        else:
            raise NotImplementedError(type(node))

    def _visit_join(self, expr):
        node = expr.op()
        self.visit(node.left)
        self.visit(node.right)

    def _visit_physical_table(self, expr):
        return

    def _visit_Aggregation(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_Filter(self, expr):
        pass

    def _visit_Limit(self, expr):
        self.visit(expr.op().table)

    def _visit_Projection(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)

    def _visit_SelfReference(self, expr):
        self.visit(expr.op().table)

    def _visit_SortBy(self, expr):
        self.observe(expr)
        self.visit(expr.op().table)


def _find_base_table(expr):
    if isinstance(expr, ir.TableExpr):
        return expr

    for arg in expr.op().args:
        if isinstance(arg, ir.Expr):
            r = _find_base_table(arg)
            if isinstance(r, ir.TableExpr):
                return r


def _join_not_none(sep, pieces):
    pieces = [x for x in pieces if x is not None]
    return sep.join(pieces)


class QueryAST(object):

    def __init__(self, context, queries):
        self.context = context
        self.queries = queries


class QueryASTBuilder(object):

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

        return Select(self.table_set, self.select_set, where=self.filters,
                      group_by=self.group_by,
                      having=self.having, limit=self.limit,
                      order_by=self.sort_by,
                      parent_expr=self.expr)

    def _analyze_where_clauses(self):
        # Various kinds of semantically valid WHERE clauses may need to be
        # rewritten into a form that we can actually translate into valid SQL.
        new_where = []
        for expr in self.filters:
            new_expr = self._visit_where(expr)

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
        elif isinstance(op, (ir.Between, ir.TableColumn, ir.Literal)):
            return expr
        else:
            raise NotImplementedError(type(op))

    def _rewrite_reduction(self, expr):
        # Find the table that this reduction references.

        # TODO: what about reductions that reference a join that isn't visible
        # at this level? Means we probably have the wrong design, but will have
        # to revisit when it becomes a problem.
        table = _find_base_table(expr)
        aggregation = table.aggregate([expr.name('tmp')])
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

    def _collect_elements(self):
        # If expr is a ValueExpr, we must seek out the TableExprs that it
        # references, build their ASTs, and mark them in our QueryContext

        # For now, we need to make the simplifying assumption that a value
        # expression that is being translated only depends on a single table
        # expression.

        source_table = self._get_source_table_expr()

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
        elif isinstance(op, ir.PhysicalTable):
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

    def _get_source_table_expr(self):
        if isinstance(self.expr, ir.TableExpr):
            return self.expr

        node = self.expr.op()

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
        return util.unique_by_key(first_tables, id)

    def _generate_setup_queries(self):
        return []

    def _generate_teardown_queries(self):
        return []
