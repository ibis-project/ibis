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

import ibis.config as config
from ibis.tests.expr.mocks import MockBackend


@pytest.fixture
def con():
    return MockBackend()


def test_interactive_execute_on_repr(con):
    table = con.table('functional_alltypes')
    expr = table.bigint_col.sum()
    with config.option_context('interactive', True):
        repr(expr)

    assert len(con.executed_queries) > 0


def test_repr_png_is_none_in_interactive(con):
    table = con.table('functional_alltypes')

    with config.option_context('interactive', True):
        assert table._repr_png_() is None


def test_repr_png_is_not_none_in_not_interactive(con):
    pytest.importorskip('ibis.expr.visualize')

    table = con.table('functional_alltypes')

    with config.option_context('interactive', False), config.option_context(
        'graphviz_repr', True
    ):
        assert table._repr_png_() is not None


def test_default_limit(con):
    table = con.table('functional_alltypes')

    with config.option_context('interactive', True):
        repr(table)

    expected = """\
SELECT *
FROM functional_alltypes
LIMIT {}""".format(
        config.options.sql.default_limit
    )

    assert con.executed_queries[0] == expected


def test_respect_set_limit(con):
    table = con.table('functional_alltypes').limit(10)

    with config.option_context('interactive', True):
        repr(table)

    expected = """\
SELECT *
FROM functional_alltypes
LIMIT 10"""

    assert con.executed_queries[0] == expected


def test_disable_query_limit(con):
    table = con.table('functional_alltypes')

    with config.option_context('interactive', True):
        with config.option_context('sql.default_limit', None):
            repr(table)

    expected = """\
SELECT *
FROM functional_alltypes"""

    assert con.executed_queries[0] == expected


def test_interactive_non_compilable_repr_not_fail(con):
    # #170
    table = con.table('functional_alltypes')

    expr = table.string_col.topk(3)

    # it works!
    with config.option_context('interactive', True):
        repr(expr)


def test_histogram_repr_no_query_execute(con):
    t = con.table('functional_alltypes')
    tier = t.double_col.histogram(10).name('bucket')
    expr = t.group_by(tier).size()
    with config.option_context('interactive', True):
        expr._repr()
    assert con.executed_queries == []


def test_compile_no_execute(con):
    t = con.table('functional_alltypes')
    t.double_col.sum().compile()
    assert con.executed_queries == []


def test_isin_rule_supressed_exception_repr_not_fail(con):
    with config.option_context('interactive', True):
        t = con.table('functional_alltypes')
        bool_clause = t['string_col'].notin(['1', '4', '7'])
        expr = t[bool_clause]['string_col'].value_counts()
        repr(expr)
