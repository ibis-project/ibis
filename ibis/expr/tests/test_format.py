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

import ibis

from ibis.compat import unittest
from ibis.expr.format import ExprFormatter
from ibis.expr.tests.mocks import MockConnection


class TestExprFormatting(unittest.TestCase):
    # Uncertain about how much we want to commit to unit tests around the
    # particulars of the output at the moment.

    def setUp(self):
        self.schema = [
            ('a', 'int8'),
            ('b', 'int16'),
            ('c', 'int32'),
            ('d', 'int64'),
            ('e', 'float'),
            ('f', 'double'),
            ('g', 'string'),
            ('h', 'boolean')
        ]
        self.schema_dict = dict(self.schema)
        self.table = ibis.table(self.schema)
        self.con = MockConnection()

    def test_format_table_column(self):
        # GH #507
        result = repr(self.table.f)
        assert 'Column[array(double)]' in result

    def test_format_projection(self):
        # This should produce a ref to the projection
        proj = self.table[['c', 'a', 'f']]
        repr(proj['a'])

    def test_table_type_output(self):
        foo = ibis.table(
            [
                ('job', 'string'),
                ('dept_id', 'string'),
                ('year', 'int32'),
                ('y', 'double')
            ], 'foo')

        expr = foo.dept_id == foo.view().dept_id
        result = repr(expr)
        assert 'SelfReference[table]' in result
        assert 'UnboundTable[table]' in result

    def test_memoize_aggregate_correctly(self):
        table = self.table

        agg_expr = (table['c'].sum() / table['c'].mean() - 1).name('analysis')
        agg_exprs = [table['a'].sum().name('sum(a)'),
                     table['b'].mean().name('mean(b)'), agg_expr]

        result = table.aggregate(agg_exprs, by=['g'])

        formatter = ExprFormatter(result)
        formatted = formatter.get_result()

        alias = formatter.memo.get_alias(table.op())
        assert formatted.count(alias) == 7

    def test_aggregate_arg_names(self):
        # Not sure how to test this *well*

        t = self.table

        by_exprs = [t.g.name('key1'), t.f.round().name('key2')]
        agg_exprs = [t.c.sum().name('c'), t.d.mean().name('d')]

        expr = self.table.group_by(by_exprs).aggregate(agg_exprs)
        result = repr(expr)
        assert 'metrics' in result
        assert 'by' in result

    def test_format_multiple_join_with_projection(self):
        # Star schema with fact table
        table = ibis.table([
            ('c', 'int32'),
            ('f', 'double'),
            ('foo_id', 'string'),
            ('bar_id', 'string'),
        ], 'one')

        table2 = ibis.table([
            ('foo_id', 'string'),
            ('value1', 'double')
        ], 'two')

        table3 = ibis.table([
            ('bar_id', 'string'),
            ('value2', 'double')
        ], 'three')

        filtered = table[table['f'] > 0]

        pred1 = filtered['foo_id'] == table2['foo_id']
        pred2 = filtered['bar_id'] == table3['bar_id']

        j1 = filtered.left_join(table2, [pred1])
        j2 = j1.inner_join(table3, [pred2])

        # Project out the desired fields
        view = j2[[filtered, table2['value1'], table3['value2']]]

        # it works!
        repr(view)

    def test_memoize_database_table(self):
        table = self.con.table('test1')
        table2 = self.con.table('test2')

        filter_pred = table['f'] > 0
        table3 = table[filter_pred]
        join_pred = table3['g'] == table2['key']

        joined = table2.inner_join(table3, [join_pred])

        met1 = (table3['f'] - table2['value']).mean().name('foo')
        result = joined.aggregate([met1, table3['f'].sum().name('bar')],
                                  by=[table3['g'], table2['key']])

        formatted = repr(result)
        assert formatted.count('test1') == 1
        assert formatted.count('test2') == 1

    def test_memoize_filtered_table(self):
        airlines = ibis.table([('dest', 'string'),
                               ('origin', 'string'),
                               ('arrdelay', 'int32')], 'airlines')

        dests = ['ORD', 'JFK', 'SFO']
        t = airlines[airlines.dest.isin(dests)]
        delay_filter = t.dest.topk(10, by=t.arrdelay.mean())

        result = repr(delay_filter)
        assert result.count('Selection') == 1

    def test_memoize_insert_sort_key(self):
        table = self.con.table('airlines')

        t = table['arrdelay', 'dest']
        expr = (t.group_by('dest')
                .mutate(dest_avg=t.arrdelay.mean(),
                        dev=t.arrdelay - t.arrdelay.mean()))

        worst = (expr[expr.dev.notnull()]
                 .sort_by(ibis.desc('dev'))
                 .limit(10))

        result = repr(worst)
        assert result.count('airlines') == 1

    def test_named_value_expr_show_name(self):
        expr = self.table.f * 2
        expr2 = expr.name('baz')

        # it works!
        repr(expr)

        result2 = repr(expr2)

        # not really committing to a particular output yet
        assert 'baz' in result2

    def test_memoize_filtered_tables_in_join(self):
        # related: GH #667
        purchases = ibis.table([('region', 'string'),
                                ('kind', 'string'),
                                ('user', 'int64'),
                                ('amount', 'double')], 'purchases')

        metric = purchases.amount.sum().name('total')
        agged = (purchases.group_by(['region', 'kind'])
                 .aggregate(metric))

        left = agged[agged.kind == 'foo']
        right = agged[agged.kind == 'bar']

        cond = left.region == right.region
        joined = (left.join(right, cond)
                  [left, right.total.name('right_total')])

        result = repr(joined)

        # Join, and one for each aggregation
        assert result.count('predicates') == 3
