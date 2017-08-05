# Copyright 2015 Cloudera Inc.
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

import os
import uuid

import pytest

import numpy as np

import pandas.util.testing as tm

import ibis
import ibis.expr.types as ir
import ibis.common as com
from ibis.util import guid


sa = pytest.importorskip('sqlalchemy')


pytestmark = pytest.mark.sqlite


def test_file_not_exist_and_create():
    path = '__ibis_tmp_{}.db'.format(guid())

    with pytest.raises(com.IbisError):
        ibis.sqlite.connect(path)

    con = ibis.sqlite.connect(path, create=True)
    assert os.path.exists(path)
    con.con.dispose()
    os.remove(path)


def test_table(con):
    table = con.table('functional_alltypes')
    assert isinstance(table, ir.TableExpr)


def test_array_execute(alltypes, df):
    expr = alltypes.double_col
    result = expr.execute()
    expected = df.double_col
    tm.assert_series_equal(result, expected)


def test_literal_execute(con):
    expr = ibis.literal('1234')
    result = con.execute(expr)
    assert result == '1234'


def test_simple_aggregate_execute(alltypes, df):
    expr = alltypes.double_col.sum()
    result = expr.execute()
    expected = df.double_col.sum()
    np.testing.assert_allclose(result, expected)


def test_list_tables(con):
    assert con.list_tables()
    assert len(con.list_tables(like='functional')) == 1


def test_compile_verify(alltypes):
    unsupported_expr = alltypes.string_col.approx_nunique()
    assert not unsupported_expr.verify()

    supported_expr = alltypes.double_col.sum()
    assert supported_expr.verify()


def test_attach_file(dbpath):
    client = ibis.sqlite.connect()

    client.attach('foo', dbpath)
    client.attach('bar', dbpath)

    foo_tables = client.list_tables(database='foo')
    bar_tables = client.list_tables(database='bar')

    assert foo_tables == bar_tables


def test_database_layer(con, db):
    assert db.list_tables() == con.list_tables()


def test_compile_toplevel():
    t = ibis.table([('foo', 'double')], name='t0')

    # it works!
    expr = t.foo.sum()
    result = ibis.sqlite.compile(expr)
    expected = """\
SELECT sum(t0.foo) AS sum 
FROM t0 AS t0"""  # noqa
    assert str(result) == expected


def test_create_and_drop_table(con):
    t = con.table('functional_alltypes')
    name = str(uuid.uuid4())
    con.create_table(name, t.limit(5))
    new_table = con.table(name)
    tm.assert_frame_equal(new_table.execute(), t.limit(5).execute())
    con.drop_table(name)
    assert name not in con.list_tables()
