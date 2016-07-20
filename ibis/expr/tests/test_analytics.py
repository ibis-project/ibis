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

from ibis.expr.tests.mocks import MockConnection
from ibis.compat import unittest
import ibis.expr.types as ir
import ibis

from ibis.tests.util import assert_equal


class TestAnalytics(unittest.TestCase):

    def setUp(self):
        self.con = MockConnection()
        self.alltypes = self.con.table('functional_alltypes')

    def test_category_project(self):
        t = self.alltypes

        tier = t.double_col.bucket([0, 50, 100]).name('tier')
        expr = t[tier, t]

        assert isinstance(expr.tier, ir.CategoryArray)

    def test_bucket(self):
        d = self.alltypes.double_col
        bins = [0, 10, 50, 100]

        expr = d.bucket(bins)
        assert isinstance(expr, ir.CategoryArray)
        assert expr.op().nbuckets == 3

        expr = d.bucket(bins, include_over=True)
        assert expr.op().nbuckets == 4

        expr = d.bucket(bins, include_over=True, include_under=True)
        assert expr.op().nbuckets == 5

    def test_bucket_error_cases(self):
        d = self.alltypes.double_col

        self.assertRaises(ValueError, d.bucket, [])
        self.assertRaises(ValueError, d.bucket, [1, 2], closed='foo')

        # it works!
        d.bucket([10], include_under=True, include_over=True)

        self.assertRaises(ValueError, d.bucket, [10])
        self.assertRaises(ValueError, d.bucket, [10], include_under=True)
        self.assertRaises(ValueError, d.bucket, [10], include_over=True)

    def test_histogram(self):
        d = self.alltypes.double_col

        self.assertRaises(ValueError, d.histogram, nbins=10, binwidth=5)
        self.assertRaises(ValueError, d.histogram)
        self.assertRaises(ValueError, d.histogram, 10, closed='foo')

    def test_topk_analysis_bug(self):
        # GH #398
        airlines = ibis.table([('dest', 'string'),
                               ('origin', 'string'),
                               ('arrdelay', 'int32')],
                              'airlines')
        dests = ['ORD', 'JFK', 'SFO']
        t = airlines[airlines.dest.isin(dests)]
        delay_filter = t.origin.topk(10, by=t.arrdelay.mean())

        filtered = t.filter([delay_filter])

        post_pred = filtered.op().predicates[1]
        assert delay_filter.to_filter().equals(post_pred)

    def test_topk_function_late_bind(self):
        # GH #520
        airlines = ibis.table([('dest', 'string'),
                               ('origin', 'string'),
                               ('arrdelay', 'int32')],
                              'airlines')
        expr1 = airlines.dest.topk(5, by=lambda x: x.arrdelay.mean())
        expr2 = airlines.dest.topk(5, by=airlines.arrdelay.mean())

        assert_equal(expr1.to_aggregation(), expr2.to_aggregation())
