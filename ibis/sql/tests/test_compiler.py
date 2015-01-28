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

import unittest

from ibis.sql.compiler import ExprTranslator, QueryContext, build_ast, to_sql
from ibis.expr.tests.mocks import MockConnection
import ibis.common as com
import ibis.expr.base as ir


# We are only testing Impala SQL dialect for the time being. At some point if
# we choose to support more SQL dialects we can refactor the test suite to
# check each supported database.


class TestASTBuilder(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_rewrite_expr_with_parent(self):
        table = self.con.table('test1')

        table2 = table[table['f'] > 0]

        expr = table2['c'] == 2

        result = ir.substitute_parents(expr)
        expected = table['c'] == 2
        assert result.equals(expected)

        # Substitution not fully possible if we depend on a new expr in a
        # projection
        table4 = table[['c', (table['c'] * 2).name('foo')]]
        expr = table4['c'] == table4['foo']
        result = ir.substitute_parents(expr)
        expected = table['c'] == table4['foo']
        assert result.equals(expected)

    def test_rewrite_past_projection(self):
        table = self.con.table('test1')

        # Rewrite past a projection
        table3 = table[['c', 'f']]
        expr = table3['c'] == 2

        result = ir.substitute_parents(expr)
        expected = table['c'] == 2
        assert result.equals(expected)

        # Unsafe to rewrite past projection
        table5 = table[(table.f * 2).name('c'), table.f]
        expr = table5['c'] == 2
        result = ir.substitute_parents(expr)
        assert result is expr

    def test_rewrite_join_projection_without_other_ops(self):
        # Drop out filters and other commutative table operations. Join
        # predicates are "lifted" to reference the base, unmodified join roots

        # Star schema with fact table
        table = self.con.table('star1')
        table2 = self.con.table('star2')
        table3 = self.con.table('star3')

        filtered = table[table['f'] > 0]

        pred1 = table['foo_id'] == table2['foo_id']
        pred2 = filtered['bar_id'] == table3['bar_id']

        j1 = filtered.left_join(table2, [pred1])
        j2 = j1.inner_join(table3, [pred2])

        # Project out the desired fields
        view = j2[[filtered, table2['value1'], table3['value2']]]

        # Construct the thing we expect to obtain
        ex_pred2 = table['bar_id'] == table3['bar_id']
        ex_expr = (table.left_join(table2, [pred1])
                   .inner_join(table3, [ex_pred2]))

        rewritten_proj = ir.substitute_parents(view)
        op = rewritten_proj.op()
        assert op.table.equals(ex_expr)

        # Ensure that filtered table has been substituted with the base table
        assert op.selections[0] is table

    def test_ast_with_projection_join_filter(self):
        table = self.con.table('test1')
        table2 = self.con.table('test2')

        filter_pred = table['f'] > 0

        table3 = table[filter_pred]

        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])
        result = joined[[table3, table2['value']]]

        ast = build_ast(result)
        stmt = ast.queries[0]

        def foo():
            table3 = table[filter_pred]
            joined = table2.inner_join(table3, [join_pred])
            result = joined[[table3, table2['value']]]
            return result

        assert len(stmt.select_set) == 2
        assert len(stmt.where) == 1
        assert stmt.where[0] is filter_pred

        # Check that the join has been rebuilt to only include the root tables
        tbl = stmt.table_set
        tbl_node = tbl.op()
        assert isinstance(tbl_node, ir.InnerJoin)
        assert tbl_node.left is table2
        assert tbl_node.right is table

        # table expression substitution has been made in the predicate
        assert tbl_node.predicates[0].equals(table['g'] == table2['key'])

    def test_ast_with_aggregation_join_filter(self):
        table = self.con.table('test1')
        table2 = self.con.table('test2')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]
        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])

        met1 = (table3['f'] - table2['value']).mean().name('foo')
        result = joined.aggregate([met1, table3['f'].sum().name('bar')],
                                  by=[table3['g'], table2['key']])

        ast = build_ast(result)
        stmt = ast.queries[0]

        # hoisted metrics
        ex_metrics = [(table['f'] - table2['value']).mean().name('foo'),
                      table['f'].sum().name('bar')]
        ex_by = [table['g'], table2['key']]

        # hoisted join and aggregate
        expected_table_set = \
            table2.inner_join(table, [table['g'] == table2['key']])
        assert stmt.table_set.equals(expected_table_set)

        # Check various exprs
        for res, ex in zip(stmt.select_set, ex_by + ex_metrics):
            assert res.equals(ex)

        for res, ex in zip(stmt.group_by, ex_by):
            assert res.equals(ex)

        # Check we got the filter
        assert len(stmt.where) == 1
        assert stmt.where[0].equals(filter_pred)

    def test_ast_non_materialized_join(self):
        pass

    def test_nonequijoin_unsupported(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        joined = t1.inner_join(t2, [t1.f < t2.value1])[[t1]]
        self.assertRaises(com.TranslationError, to_sql, joined)

    def test_simple_scalar_aggregates(self):
        # Things like table.column.{sum, mean, ...}()
        pass

    def test_simple_count_distinct(self):
        pass

    def test_input_source_from_sql(self):
        pass

    def test_input_source_from_textfile(self):
        pass

    def test_sort_by(self):
        table = self.con.table('star1')

        what = table.sort_by('f')
        result = to_sql(what)
        expected = """SELECT *
FROM star1
ORDER BY f"""
        assert result == expected

        what = table.sort_by(('f', 0))
        result = to_sql(what)
        expected = """SELECT *
FROM star1
ORDER BY f DESC"""
        assert result == expected

        what = table.sort_by(['c', ('f', 0)])
        result = to_sql(what)
        expected = """SELECT *
FROM star1
ORDER BY c, f DESC"""
        assert result == expected

    def test_limit(self):
        table = self.con.table('star1').limit(10)
        result = to_sql(table)
        expected = """SELECT *
FROM star1
LIMIT 10"""
        assert result == expected

        table = self.con.table('star1').limit(10, offset=5)
        result = to_sql(table)
        expected = """SELECT *
FROM star1
LIMIT 10 OFFSET 5"""
        assert result == expected

        # Put the limit in a couple places in the stack
        table = self.con.table('star1')
        table = table[table.f > 0].limit(10)
        result = to_sql(table)

        expected = """SELECT *
FROM star1
WHERE f > 0
LIMIT 10"""

        assert result == expected

        table = self.con.table('star1')

        # Semantically, this should produce a subquery
        table = table.limit(10)
        table = table[table.f > 0]

        result2 = to_sql(table)

        expected2 = """SELECT *
FROM (
  SELECT *
  FROM star1
  LIMIT 10
)
WHERE f > 0"""

        assert result2 == expected2

    def test_join_with_limited_table(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        limited = t1.limit(100)
        joined = (limited.inner_join(t2, [limited.foo_id == t2.foo_id])
                  [[limited]])

        result = to_sql(joined)
        expected = """SELECT t0.*
FROM (
  SELECT *
  FROM star1
  LIMIT 100
) t0
  INNER JOIN star2 t1
    ON t0.foo_id = t1.foo_id"""

        assert result == expected

    def test_sort_by_on_limit_yield_subquery(self):
        # x.limit(...).sort_by(...)
        # is different from
        # x.sort_by(...).limit(...)
        pass

    def test_top_convenience(self):
        # x.top(10, by=field)
        # x.top(10, by=[field1, field2])
        pass

    def test_scalar_aggregate_expr(self):
        # Things like (table.a - table2.b.mean()).sum(), requiring subquery
        # extraction
        pass

    def test_filter_in_between_joins(self):
        # With filter predicates involving only a single
        pass

    def test_self_aggregate_in_predicate(self):
        # Per ibis#43
        pass


def _get_query(expr):
    ast = build_ast(expr)
    return ast.queries[0]


class TestSelectSQL(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()

    def test_nameless_table(self):
        # Ensure that user gets some kind of sensible error
        nameless = ir.table([('key', 'string')])
        self.assertRaises(com.RelationError, to_sql, nameless)

        with_name = ir.table([('key', 'string')], name='baz')
        result = to_sql(with_name)
        assert result == 'SELECT *\nFROM baz'

    def test_physical_table_reference_translate(self):
        # If an expression's table leaves all reference database tables, verify
        # we translate correctly
        table = self.con.table('alltypes')

        query = _get_query(table)
        sql_string = query.compile()
        expected = "SELECT *\nFROM alltypes"
        assert sql_string == expected

    def test_simple_join_formatting(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        pred = t1['foo_id'] == t2['foo_id']
        pred2 = t1['bar_id'] == t2['foo_id']
        cases = [
            (t1.inner_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.foo_id = t1.foo_id"""),
            (t1.left_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  LEFT OUTER JOIN star2 t1
    ON t0.foo_id = t1.foo_id"""),
            (t1.outer_join(t2, [pred])[[t1]],
             """SELECT t0.*
FROM star1 t0
  FULL OUTER JOIN star2 t1
    ON t0.foo_id = t1.foo_id"""),
            # multiple predicates
            (t1.inner_join(t2, [pred, pred2])[[t1]],
             """SELECT t0.*
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.foo_id = t1.foo_id AND
       t0.bar_id = t1.foo_id"""),
        ]

        for expr, expected_sql in cases:
            result_sql = to_sql(expr)
            assert result_sql == expected_sql

    def test_multiple_join_cases(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')
        t3 = self.con.table('star3')

        predA = t1['foo_id'] == t2['foo_id']
        predB = t1['bar_id'] == t3['bar_id']

        what = (t1.left_join(t2, [predA])
                .inner_join(t3, [predB])
                .projection([t1, t2['value1'], t3['value2']]))
        result_sql = to_sql(what)
        expected_sql = """SELECT t0.*, t1.value1, t2.value2
FROM star1 t0
  LEFT OUTER JOIN star2 t1
    ON t0.foo_id = t1.foo_id
  INNER JOIN star3 t2
    ON t0.bar_id = t2.bar_id"""
        assert result_sql == expected_sql

    def test_join_between_joins(self):
        t1 = ir.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value1', 'double'),
        ], 'first')

        t2 = ir.table([
            ('key1', 'string'),
            ('value2', 'double'),
        ], 'second')

        t3 = ir.table([
            ('key2', 'string'),
            ('key3', 'string'),
            ('value3', 'double'),
        ], 'third')

        t4 = ir.table([
            ('key3', 'string'),
            ('value4', 'double')
        ], 'fourth')

        left = t1.inner_join(t2, [('key1', 'key1')])[t1, t2.value2]
        right = t3.inner_join(t4, [('key3', 'key3')])[t3, t4.value4]

        joined = left.inner_join(right, [('key2', 'key2')])

        # At one point, the expression simplification was resulting in bad refs
        # here (right.value3 referencing the table inside the right join)
        exprs = [left, right.value3, right.value4]
        projected = joined.projection(exprs)

        result = to_sql(projected)
        expected = """SELECT t0.*, t1.value3, t1.value4
FROM (
  SELECT t0.*, t1.value2
  FROM first t0
    INNER JOIN second t1
      ON t0.key1 = t1.key1
) t0
  INNER JOIN (
    SELECT t0.*, t1.value4
    FROM third t0
      INNER JOIN fourth t1
        ON t0.key3 = t1.key3
  ) t1
    ON t0.key2 = t1.key2"""
        assert result == expected

    def test_self_reference_simple(self):
        t1 = self.con.table('star1')

        result_sql = to_sql(t1.view())
        expected_sql = "SELECT *\nFROM star1"
        assert result_sql == expected_sql

    def test_join_self_reference(self):
        t1 = self.con.table('star1')
        t2 = t1.view()

        result = t1.inner_join(t2, [t1.foo_id == t2.bar_id])[[t1]]

        result_sql = to_sql(result)
        expected_sql = """SELECT t0.*
FROM star1 t0
  INNER JOIN star1 t1
    ON t0.foo_id = t1.bar_id"""
        assert result_sql == expected_sql

    def test_where_simple_comparisons(self):
        t1 = self.con.table('star1')

        what = t1.filter([t1.f > 0, t1.c < t1.f * 2])

        result = to_sql(what)
        expected = """SELECT *
FROM star1
WHERE f > 0 AND
      c < (f * 2)"""
        assert result == expected

    def test_where_in_array_literal(self):
        # e.g.
        # where string_col in (v1, v2, v3)
        raise unittest.SkipTest

    def test_where_with_join(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        # This also tests some cases of predicate pushdown
        what = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                [t1, t2.value1]
                .filter([t1.f > 0, t2.value3 < 1000]))

        what2 = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                 .filter([t1.f > 0, t2.value3 < 1000])
                 [t1, t2.value1])

        expected_sql = """SELECT t0.*, t1.value1
FROM star1 t0
  INNER JOIN star2 t1
    ON t0.foo_id = t1.foo_id
WHERE t0.f > 0 AND
      t1.value3 < 1000"""

        result_sql = to_sql(what)
        assert result_sql == expected_sql

        result2_sql = to_sql(what2)
        assert result2_sql == expected_sql

    def test_where_no_pushdown_possible(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        joined = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                [t1, (t1.f - t2.value1).name('diff')])

        filtered = joined[joined.diff > 1]

        # TODO: I'm not sure if this is exactly what we want
        expected_sql = """SELECT *
FROM (
  SELECT t0.*, t0.f - t1.value1 AS diff
  FROM star1 t0
    INNER JOIN star2 t1
      ON t0.foo_id = t1.foo_id
  WHERE t0.f > 0 AND
        t1.value3 < 1000
)
WHERE diff > 1"""

        raise unittest.SkipTest

        result_sql = to_sql(filtered)
        assert result_sql == expected_sql

    def test_where_correlation_subquery(self):
        pass

    def test_where_uncorrelated_subquery(self):
        pass

    def test_simple_aggregate_query(self):
        t1 = self.con.table('star1')

        cases = [
            (t1.aggregate([t1['f'].sum().name('total')],
                          [t1['foo_id']]),
             """SELECT foo_id, sum(f) AS total
FROM star1
GROUP BY 1"""),
            (t1.aggregate([t1['f'].sum().name('total')],
                          ['foo_id', 'bar_id']),
             """SELECT foo_id, bar_id, sum(f) AS total
FROM star1
GROUP BY 1, 2""")
        ]
        for expr, expected_sql in cases:
            result_sql = to_sql(expr)
            assert result_sql == expected_sql

    def test_aggregate_having(self):
        # Filtering post-aggregation predicate
        t1 = self.con.table('star1')

        total = t1.f.sum().name('total')
        metrics = [total]

        expr = t1.aggregate(metrics, by=['foo_id'],
                            having=[total > 10])
        result = to_sql(expr)
        expected = """SELECT foo_id, sum(f) AS total
FROM star1
GROUP BY 1
HAVING sum(f) > 10"""
        assert result == expected

        expr = t1.aggregate(metrics, by=['foo_id'],
                            having=[t1.count() > 100])
        result = to_sql(expr)
        expected = """SELECT foo_id, sum(f) AS total
FROM star1
GROUP BY 1
HAVING count(*) > 100"""
        assert result == expected

    def test_expr_template_field_name_binding(self):
        # Given an expression with no concrete links to actual database tables,
        # indicate a mapping between the distinct unbound table leaves of the
        # expression and some database tables with compatible schemas but
        # potentially different column names
        pass

    def test_no_aliases_needed(self):
        table = ir.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('value', 'double')
        ])

        expr = table.aggregate([table['value'].sum().name('total')],
                               by=['key1', 'key2'])

        query = _get_query(expr)
        context = QueryContext()
        query.populate_context(context)
        assert not context.need_aliases()

    def test_context_aliases_multiple_join(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')
        t3 = self.con.table('star3')

        expr = (t1.left_join(t2, [t1['foo_id'] == t2['foo_id']])
                .inner_join(t3, [t1['bar_id'] == t3['bar_id']])
                [[t1, t2['value1'], t3['value2']]])

        query = _get_query(expr)
        context = QueryContext()
        query.populate_context(context)

        assert context.get_alias(t1) == 't0'
        assert context.get_alias(t2) == 't1'
        assert context.get_alias(t3) == 't2'

    def test_fuse_projections(self):
        table = ir.table([
            ('foo', 'int32'),
            ('bar', 'int64'),
            ('value', 'double')
        ], name='table')

        # Cases where we project in both cases using the base table reference
        f1 = (table['foo'] + table['bar']).name('baz')
        pred = table['value'] > 0

        table2 = table[table, f1]
        table2_filtered = table2[pred]

        f2 = (table2['foo'] * 2).name('qux')
        f3 = (table['foo'] * 2).name('qux')

        table3 = table2.projection([table2, f3])

        # fusion works even if there's a filter
        table3_filtered = table2_filtered.projection([table2, f3])

        expected = table[table, f1, f3]
        expected2 = table[pred][table, f1, f3]

        assert table3.equals(expected)
        assert table3_filtered.equals(expected2)

        ex_sql = """SELECT *, foo + bar AS baz, foo * 2 AS qux
FROM table"""

        ex_sql2 = """SELECT *, foo + bar AS baz, foo * 2 AS qux
FROM table
WHERE value > 0"""

        table3_sql = to_sql(table3)
        table3_filt_sql = to_sql(table3_filtered)

        assert table3_sql == ex_sql
        assert table3_filt_sql == ex_sql2

        # Use the intermediate table refs
        table3 = table2.projection([table2, f2])

        # fusion works even if there's a filter
        table3_filtered = table2_filtered.projection([table2, f2])

        expected = table[table, f1, f3]
        expected2 = table[pred][table, f1, f3]

        assert table3.equals(expected)
        assert table3_filtered.equals(expected2)

    def test_aggregate_projection_subquery(self):
        t = self.con.table('alltypes')

        proj = t[t.f > 0][t, (t.a + t.b).name('foo')]

        def agg(x):
            return x.aggregate([x.foo.sum().name('foo total')], by=['g'])

        # predicate gets pushed down
        filtered = proj[proj.g == 'bar']

        result = to_sql(filtered)
        expected = """SELECT *, a + b AS foo
FROM alltypes
WHERE f > 0 AND
      g = 'bar'"""
        assert result == expected

        agged = agg(filtered)
        result = to_sql(agged)
        expected = """SELECT g, sum(foo) AS `foo total`
FROM (
  SELECT *, a + b AS foo
  FROM alltypes
  WHERE f > 0 AND
        g = 'bar'
)
GROUP BY 1"""
        assert result == expected

        # different pushdown case. Does Impala support this?
        agged2 = agg(proj[proj.foo < 10])

        result = to_sql(agged2)
        expected = """SELECT g, sum(foo) AS `foo total`
FROM (
  SELECT *, a + b AS foo
  FROM alltypes
  WHERE f > 0 AND
        foo < 10
)
GROUP BY 1"""
        assert result == expected

    def test_subquery_aliased(self):
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        agged = t1.aggregate([t1.f.sum().name('total')], by=['foo_id'])
        what = (agged.inner_join(t2, [agged.foo_id == t2.foo_id])
                [agged, t2.value1])

        result = to_sql(what)
        expected = """SELECT t0.*, t1.value1
FROM (
  SELECT foo_id, sum(f) AS total
  FROM star1
  GROUP BY 1
) t0
  INNER JOIN star2 t1
    ON t0.foo_id = t1.foo_id"""
        assert result == expected

    def test_double_nested_subquery_no_aliases(self):
        # We don't require any table aliasing anywhere
        t = ir.table([
            ('key1', 'string'),
            ('key2', 'string'),
            ('key3', 'string'),
            ('value', 'double')
        ], 'foo_table')

        agg1 = t.aggregate([t.value.sum().name('total')],
                           by=['key1', 'key2', 'key3'])
        agg2 = agg1.aggregate([agg1.total.sum().name('total')],
                              by=['key1', 'key2'])
        agg3 = agg2.aggregate([agg2.total.sum().name('total')],
                              by=['key1'])

        result = to_sql(agg3)
        expected = """SELECT key1, sum(total) AS total
FROM (
  SELECT key1, key2, sum(total) AS total
  FROM (
    SELECT key1, key2, key3, sum(value) AS total
    FROM foo_table
    GROUP BY 1, 2, 3
  )
  GROUP BY 1, 2
)
GROUP BY 1"""
        assert result == expected

    def test_aggregate_projection_alias_bug(self):
        # Observed in use
        t1 = self.con.table('star1')
        t2 = self.con.table('star2')

        what = (t1.inner_join(t2, [t1.foo_id == t2.foo_id])
                [[t1, t2.value1]])

        what = what.aggregate([what.value1.sum().name('total')],
                              by=[t1.foo_id])

        # TODO: Not fusing the aggregation with the projection yet
        result = to_sql(what)
        expected = """SELECT foo_id, sum(value1) AS total
FROM (
  SELECT t0.*, t1.value1
  FROM star1 t0
    INNER JOIN star2 t1
      ON t0.foo_id = t1.foo_id
)
GROUP BY 1"""
        assert result == expected

    def test_aggregate_fuse_with_projection(self):
        # see above test case
        pass

    def test_subquery_used_for_self_join(self):
        # There could be cases that should look in SQL like
        # WITH t0 as (some subquery)
        # select ...
        # from t0 t1
        #   join t0 t2
        #     on t1.kind = t2.subkind
        # ...
        # However, the Ibis code will simply have an expression (projection or
        # aggregation, say) built on top of the subquery expression, so we need
        # to extract the subquery unit (we see that it appears multiple times
        # in the tree).
        t = self.con.table('alltypes')

        agged = t.aggregate([t.f.sum().name('total')], by=['g', 'a', 'b'])
        view = agged.view()
        metrics = [(agged.total - view.total).max().name('metric')]
        reagged = (agged.inner_join(view, [agged.a == view.b])
                   .aggregate(metrics, by=[agged.g]))

        result = to_sql(reagged)
        expected = """WITH t0 AS (
  SELECT g, a, b, sum(f) AS total
  FROM alltypes
  GROUP BY 1, 2, 3
)
SELECT t0.g, max(t0.total - t1.total) AS metric
FROM t0
  INNER JOIN t0 t1
    ON t0.a = t1.b
GROUP BY 1"""
        assert result == expected

    def test_extract_subquery_nested_lower(self):
        # We may have a join between two tables requiring subqueries, and
        # buried inside these there may be a common subquery. Let's test that
        # we find it and pull it out to the top level to avoid repeating
        # ourselves.
        pass

    def test_subquery_in_filter_predicate(self):
        # E.g. comparing against some scalar aggregate value. See Ibis #43
        t1 = self.con.table('star1')

        pred = t1.f > t1.f.mean()
        expr = t1[pred]

        # This brought out another expression rewriting bug, since the filtered
        # table isn't found elsewhere in the expression.
        pred2 = t1.f > t1[t1.foo_id == 'foo'].f.mean()
        expr2 = t1[pred2]

        result = to_sql(expr)
        expected = """SELECT *
FROM star1
WHERE f > (
  SELECT avg(f) AS tmp
  FROM star1
)"""
        assert result == expected

        result = to_sql(expr2)
        expected = """SELECT *
FROM star1
WHERE f > (
  SELECT avg(f) AS tmp
  FROM star1
  WHERE foo_id = 'foo'
)"""
        assert result == expected

    def test_filter_subquery_derived_reduction(self):
        t1 = self.con.table('star1')

        # Reduction can be nested inside some scalar expression
        pred3 = t1.f > t1[t1.foo_id == 'foo'].f.mean().log()
        pred4 = t1.f > (t1[t1.foo_id == 'foo'].f.mean().log() + 1)

        expr3 = t1[pred3]
        result = to_sql(expr3)
        expected = """SELECT *
FROM star1
WHERE f > (
  SELECT log(avg(f)) AS tmp
  FROM star1
  WHERE foo_id = 'foo'
)"""
        assert result == expected

        expr4 = t1[pred4]

        result = to_sql(expr4)
        expected = """SELECT *
FROM star1
WHERE f > (
  SELECT log(avg(f)) + 1 AS tmp
  FROM star1
  WHERE foo_id = 'foo'
)"""
        assert result == expected

    def test_subquery_in_where_from_another_table(self):
        # TODO: this will currently break at the IR level
        pass


class TestASTTransformations(unittest.TestCase):

    pass





class TestValueExprs(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.table = self.con.table('alltypes')

        self.int_cols = ['a', 'b', 'c', 'd']
        self.bool_cols = ['h']
        self.float_cols = ['e', 'f']

    def _translate(self, expr, named=False, context=None):
        translator = ExprTranslator(expr, context=context, named=named)
        return translator.get_result()

    def _check_literals(self, cases):
        for value, expected in cases:
            lit_expr = ir.literal(value)
            result = self._translate(lit_expr)
            assert result == expected

    def _check_expr_cases(self, cases, context=None, named=False):
        for expr, expected in cases:
            result = self._translate(expr, named=named, context=context)
            assert result == expected

    def test_string_literals(self):
        cases = [
            ('simple', "'simple'"),
            ('I can\'t', "'I can\\'t'"),
            ('An "escape"', "'An \"escape\"'")
        ]

        for value, expected in cases:
            lit_expr = ir.literal(value)
            result = self._translate(lit_expr)
            assert result == expected

    def test_number_boolean_literals(self):
        cases = [
            (5, '5'),
            (1.5, '1.5'),
            (True, 'TRUE'),
            (False, 'FALSE')
        ]
        self._check_literals(cases)

    def test_column_ref_table_aliases(self):
        context = QueryContext()

        table1 = ir.table([
            ('key1', 'string'),
            ('value1', 'double')
        ])

        table2 = ir.table([
            ('key2', 'string'),
            ('value and2', 'double')
        ])

        context.set_alias(table1, 't0')
        context.set_alias(table2, 't1')

        expr = table1['value1'] - table2['value and2']

        result = self._translate(expr, context=context)
        expected = 't0.value1 - t1.`value and2`'
        assert result == expected

    def test_column_ref_quoting(self):
        schema = [('has a space', 'double')]
        table = ir.table(schema)
        self._translate(table['has a space'], '`has a space`')

    def test_named_expressions(self):
        a, b, g = self.table.get_columns(['a', 'b', 'g'])

        cases = [
            (g.cast('double').name('g_dub'), 'CAST(g AS double) AS g_dub'),
            (g.name('has a space'), 'g AS `has a space`'),
            (((a - b) * a).name('expr'), '(a - b) * a AS expr')
        ]

        return self._check_expr_cases(cases, named=True)

    def test_binary_infix_operators(self):
        # For each function, verify that the generated code is what we expect
        a, b, h = self.table.get_columns(['a', 'b', 'h'])
        bool_col = a > 0

        cases = [
            (a + b, 'a + b'),
            (a - b, 'a - b'),
            (a * b, 'a * b'),
            (a / b, 'a / b'),
            (a ** b, 'a ^ b'),
            (a < b, 'a < b'),
            (a <= b, 'a <= b'),
            (a > b, 'a > b'),
            (a >= b, 'a >= b'),
            (a == b, 'a = b'),
            (a != b, 'a != b'),
            (h & bool_col, 'h AND (a > 0)'),
            (h | bool_col, 'h OR (a > 0)'),
            # xor is brute force
            (h ^ bool_col, '(h OR (a > 0)) AND NOT (h AND (a > 0))')
        ]
        self._check_expr_cases(cases)

    def test_binary_infix_parenthesization(self):
        a, b, c = self.table.get_columns(['a', 'b', 'c'])

        cases = [
            ((a + b) + c, '(a + b) + c'),
            (a.log() + c, 'log(a) + c'),
            (b + (-(a + c)), 'b + (-(a + c))')
        ]

        self._check_expr_cases(cases)

    def test_isnull_notnull(self):
        cases = [
            (self.table['g'].isnull(), 'g IS NULL'),
            (self.table['a'].notnull(), 'a IS NOT NULL'),
            ((self.table['a'] + self.table['b']).isnull(), 'a + b IS NULL')
        ]
        self._check_expr_cases(cases)

    def test_casts(self):
        a, d, g = self.table.get_columns(['a', 'd', 'g'])
        cases = [
            (a.cast('int16'), 'CAST(a AS smallint)'),
            (a.cast('int32'), 'CAST(a AS int)'),
            (a.cast('int64'), 'CAST(a AS bigint)'),
            (a.cast('float'), 'CAST(a AS float)'),
            (a.cast('double'), 'CAST(a AS double)'),
            (a.cast('string'), 'CAST(a AS string)'),
            (d.cast('int8'), 'CAST(d AS tinyint)'),
            (g.cast('double'), 'CAST(g AS double)')
        ]
        self._check_expr_cases(cases)

    def test_negate(self):
        cases = [
            (-self.table['a'], '-a'),
            (-self.table['f'], '-f'),
            (-self.table['h'], 'NOT h')
        ]
        self._check_expr_cases(cases)
