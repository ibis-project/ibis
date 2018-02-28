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


import pytest

import ibis
import ibis.common as com

from ibis import window
from ibis.tests.util import assert_equal

pytest.importorskip('hdfs')
pytest.importorskip('sqlalchemy')
pytest.importorskip('impala.dbapi')

from ibis.impala.compiler import to_sql  # noqa: E402
from ibis.impala.tests.common import ImpalaE2E  # noqa: E402


@pytest.yield_fixture(scope='module')
def con(request):
    ImpalaE2E.setUpClass()

    try:
        yield ImpalaE2E.con
    finally:
        ImpalaE2E.tearDownClass()


def assert_sql_equal(expr, expected):
    result = to_sql(expr)
    assert result == expected


@pytest.mark.impala
def test_aggregate_in_projection(con):
    t = con.table('alltypes')
    proj = t[t, (t.f / t.f.sum()).name('normed_f')]

    expected = """\
SELECT *, `f` / sum(`f`) OVER () AS `normed_f`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(proj, expected)


@pytest.mark.impala
def test_add_default_order_by(con):
    t = con.table('alltypes')

    first = t.f.first().name('first')
    last = t.f.last().name('last')
    lag = t.f.lag().name('lag')
    diff = (t.f.lead() - t.f).name('fwd_diff')
    lag2 = t.f.lag().over(window(order_by=t.d)).name('lag2')
    grouped = t.group_by('g')
    proj = grouped.mutate([lag, diff, first, last, lag2])
    expected = """\
SELECT *, lag(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `lag`,
       lead(`f`) OVER (PARTITION BY `g` ORDER BY `f`) - `f` AS `fwd_diff`,
       first_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `first`,
       last_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `last`,
       lag(`f`) OVER (PARTITION BY `g` ORDER BY `d`) AS `lag2`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(proj, expected)


@pytest.mark.impala
@pytest.mark.parametrize(
    ['window', 'frame'],
    [
        (window(preceding=0),
         'range between current row and unbounded following'),

        (window(following=0),
         'range between unbounded preceding and current row'),

        (window(preceding=5),
         'rows between 5 preceding and unbounded following'),
        (window(preceding=5, following=0),
         'rows between 5 preceding and current row'),
        (window(preceding=5, following=2),
         'rows between 5 preceding and 2 following'),
        (window(following=2),
         'rows between unbounded preceding and 2 following'),
        (window(following=2, preceding=0),
         'rows between current row and 2 following'),
        (window(preceding=5),
         'rows between 5 preceding and unbounded following'),
        (window(following=[5, 10]),
         'rows between 5 following and 10 following'),
        (window(preceding=[10, 5]),
         'rows between 10 preceding and 5 preceding'),

        # # cumulative windows
        (ibis.cumulative_window(),
         'range between unbounded preceding and current row'),

        # # trailing windows
        (ibis.trailing_window(10),
         'rows between 10 preceding and current row'),
    ]
)
def test_window_frame_specs(con, window, frame):
    t = con.table('alltypes')

    ex_template = """\
SELECT sum(`d`) OVER (ORDER BY `f` {0}) AS `foo`
FROM ibis_testing.`alltypes`"""

    w2 = window.order_by(t.f)
    expr = t.projection([t.d.sum().over(w2).name('foo')])
    expected = ex_template.format(frame.upper())
    assert_sql_equal(expr, expected)


@pytest.mark.impala
def test_cumulative_functions(con):
    t = con.table('alltypes')

    w = ibis.window(order_by=t.d)
    exprs = [
        (t.f.cumsum().over(w), t.f.sum().over(w)),
        (t.f.cummin().over(w), t.f.min().over(w)),
        (t.f.cummax().over(w), t.f.max().over(w)),
        (t.f.cummean().over(w), t.f.mean().over(w)),
    ]

    for cumulative, static in exprs:
        actual = cumulative.name('foo')
        expected = static.over(ibis.cumulative_window()).name('foo')

        expr1 = t.projection(actual)
        expr2 = t.projection(expected)

        assert to_sql(expr1) == to_sql(expr2)


@pytest.mark.impala
def test_nested_analytic_function(con):
    t = con.table('alltypes')

    w = window(order_by=t.f)
    expr = (t.f - t.f.lag()).lag().over(w).name('foo')
    result = t.projection([expr])
    expected = """\
SELECT lag(`f` - lag(`f`) OVER (ORDER BY `f`)) \
OVER (ORDER BY `f`) AS `foo`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(result, expected)


@pytest.mark.impala
def test_rank_functions(con):
    t = con.table('alltypes')

    proj = t[t.g, t.f.rank().name('minr'),
             t.f.dense_rank().name('denser')]
    expected = """\
SELECT `g`, (rank() OVER (ORDER BY `f`) - 1) AS `minr`,
       (dense_rank() OVER (ORDER BY `f`) - 1) AS `denser`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(proj, expected)


@pytest.mark.impala
def test_multiple_windows(con):
    t = con.table('alltypes')

    w = window(group_by=t.g)

    expr = t.f.sum().over(w) - t.f.sum()
    proj = t.projection([t.g, expr.name('result')])

    expected = """\
SELECT `g`, sum(`f`) OVER (PARTITION BY `g`) - sum(`f`) OVER () AS `result`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(proj, expected)


@pytest.mark.impala
def test_order_by_desc(con):
    t = con.table('alltypes')

    w = window(order_by=ibis.desc(t.f))

    proj = t[t.f, ibis.row_number().over(w).name('revrank')]
    expected = """\
SELECT `f`, (row_number() OVER (ORDER BY `f` DESC) - 1) AS `revrank`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(proj, expected)

    expr = (t.group_by('g')
            .order_by(ibis.desc(t.f))
            [t.d.lag().name('foo'), t.a.max()])
    expected = """\
SELECT lag(`d`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `foo`,
       max(`a`) OVER (PARTITION BY `g` ORDER BY `f` DESC) AS `max`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(expr, expected)


@pytest.mark.impala
def test_row_number_does_not_require_order_by(con):
    t = con.table('alltypes')

    expr = t.group_by(t.g).mutate(ibis.row_number().name('foo'))
    expected = """\
SELECT *, (row_number() OVER (PARTITION BY `g`) - 1) AS `foo`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(expr, expected)

    expr = (t.group_by(t.g)
            .order_by(t.f)
            .mutate(ibis.row_number().name('foo')))

    expected = """\
SELECT *, (row_number() OVER (PARTITION BY `g` ORDER BY `f`) - 1) AS `foo`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(expr, expected)


@pytest.mark.impala
def test_row_number_properly_composes_with_arithmetic(con):
    t = con.table('alltypes')
    w = ibis.window(order_by=t.f)
    expr = t.mutate(new=ibis.row_number().over(w) / 2)

    expected = """\
SELECT *, (row_number() OVER (ORDER BY `f`) - 1) / 2 AS `new`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(expr, expected)


@pytest.mark.impala
@pytest.mark.parametrize(
    ['column', 'op'],
    [
        ('f', 'approx_nunique'),
        ('f', 'approx_median'),
        ('g', 'group_concat'),
    ]
)
def test_unsupported_aggregate_functions(con, column, op):
    t = con.table('alltypes')
    w = ibis.window(order_by=t.d)
    expr = getattr(t[column], op)()
    proj = t.projection([expr.over(w).name('foo')])
    with pytest.raises(com.TranslationError):
        to_sql(proj)


@pytest.mark.impala
def test_propagate_nested_windows(con):
    # GH #469
    t = con.table('alltypes')

    w = ibis.window(group_by=t.g, order_by=t.f)

    col = (t.f - t.f.lag()).lag()

    # propagate down here!
    result = col.over(w)
    ex_expr = (t.f - t.f.lag().over(w)).lag().over(w)
    assert_equal(result, ex_expr)

    expr = t.projection(col.over(w).name('foo'))
    expected = """\
SELECT lag(`f` - lag(`f`) OVER (PARTITION BY `g` ORDER BY `f`)) \
OVER (PARTITION BY `g` ORDER BY `f`) AS `foo`
FROM ibis_testing.`alltypes`"""
    assert_sql_equal(expr, expected)


@pytest.mark.xfail
def test_math_on_windowed_expr():
    # Window clause may not be found at top level of expression
    assert False


@pytest.mark.xfail
def test_group_by_then_different_sort_orders():
    assert False
