import pytest

import ibis

pytestmark = pytest.mark.spark
pytest.importorskip('pyspark')


@pytest.fixture(scope='session')
def expr():
    schema = ibis.schema([('c1', 'int64'), ('c2', 'string')])
    t = ibis.table(schema, name='tbl')
    expr = t.projection(['c2', (t.c1 + 1).name('c3')]).mutate(c4=t.c2.length())
    return expr


def test_compile(client, expr):
    expected = 'SELECT `c2`, `c1` + 1 AS `c3`, length(`c2`) AS `c4`\nFROM tbl'
    assert client.compile(expr) == ibis.spark.compile(expr) == expected


def test_verify(expr):
    assert ibis.spark.verify(expr)
