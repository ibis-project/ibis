import pytest

import ibis

pytestmark = pytest.mark.spark


@pytest.fixture(scope='session')
def expr():
    schema = ibis.schema(
        [
            ('foo', 'int64'),
            ('bar', 'string')
        ]
    )
    t = ibis.table(
        schema, name='tbl'
    )
    expr = t.projection(
        ['bar', (t.foo + 1).name('new_foo')]
    ).mutate(
        bar_len=t.bar.length()
    )
    return expr


def test_compile(client, expr):
    assert client.compile(expr) == ibis.spark.compile(expr)


def test_verify(expr):
    assert ibis.spark.verify(expr)
