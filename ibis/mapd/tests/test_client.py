from ibis.tests.util import assert_equal

import ibis
import ibis.expr.types as ir
import pandas as pd
import pytest


pytestmark = pytest.mark.mapd
pytest.importorskip('pymapd')


def test_table(alltypes):
    assert isinstance(alltypes, ir.TableExpr)


def test_array_execute(alltypes):
    d = alltypes.limit(10).double_col
    s = d.execute()
    assert isinstance(s, pd.Series)
    assert len(s) == 10


def test_literal_execute(alltypes):
    expr = alltypes[alltypes, ibis.literal('1234').name('lit')].limit(1)
    result = expr.execute()
    assert result.lit[0] == '1234'


def test_simple_aggregate_execute(alltypes):
    d = alltypes.double_col.sum().name('sum1')
    v = d.execute()
    assert isinstance(v, float)


def test_list_tables(con):
    assert len(con.list_tables()) > 0
    assert len(con.list_tables(like='functional_alltypes')) == 1


def test_compile_verify(alltypes):
    supported_expr = alltypes.double_col.sum()
    assert supported_expr.verify()


def test_database_layer(con, alltypes):
    db = con.database()
    t = db.functional_alltypes

    assert_equal(t, alltypes)

    assert db.list_tables() == con.list_tables()


def test_compile_toplevel():
    t = ibis.table([('foo', 'double')], name='t0')

    # it works!
    expr = t.foo.sum()
    result = ibis.mapd.compile(expr)
    expected = 'SELECT sum("foo") AS "sum"\nFROM t0'  # noqa
    assert str(result) == expected
