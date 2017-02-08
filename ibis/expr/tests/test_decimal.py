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

import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.expr.operations as ops

from ibis.expr.tests.mocks import MockConnection

import pytest


@pytest.fixture
def con():
    return MockConnection()


@pytest.fixture
def lineitem(con):
    return con.table('tpch_lineitem')


def test_type_metadata(lineitem):
    col = lineitem.l_extendedprice
    assert isinstance(col, ir.DecimalColumn)

    assert col._precision == 12
    assert col._scale == 2


def test_cast_scalar_to_decimal():
    val = api.literal('1.2345')

    casted = val.cast('decimal(15,5)')
    assert isinstance(casted, ir.DecimalScalar)
    assert casted._precision == 15
    assert casted._scale == 5


def test_decimal_aggregate_function_behavior(lineitem):
    # From the Impala documentation: "The result of an aggregate function
    # such as MAX(), SUM(), or AVG() on DECIMAL values is promoted to a
    # scale of 38, with the same precision as the underlying column. Thus,
    # the result can represent the largest possible value at that
    # particular precision."
    col = lineitem.l_extendedprice
    functions = ['sum', 'mean', 'max', 'min']

    for func_name in functions:
        result = getattr(col, func_name)()
        assert isinstance(result, ir.DecimalScalar)
        assert result._precision == col._precision
        assert result._scale == 38


def test_where(lineitem):
    table = lineitem

    q = table.l_quantity
    expr = api.where(table.l_discount > 0,
                     q * table.l_discount, api.null)

    assert isinstance(expr, ir.DecimalColumn)

    expr = api.where(table.l_discount > 0,
                     (q * table.l_discount).sum(), api.null)
    assert isinstance(expr, ir.DecimalColumn)

    expr = api.where(table.l_discount.sum() > 0,
                     (q * table.l_discount).sum(), api.null)
    assert isinstance(expr, ir.DecimalScalar)


def test_fillna(lineitem):
    expr = lineitem.l_extendedprice.fillna(0)
    assert isinstance(expr, ir.DecimalColumn)

    expr = lineitem.l_extendedprice.fillna(lineitem.l_quantity)
    assert isinstance(expr, ir.DecimalColumn)


def test_precision_scale(lineitem):
    col = lineitem.l_extendedprice

    p = col.precision()
    s = col.scale()

    assert isinstance(p, ir.IntegerValue)
    assert isinstance(p.op(), ops.DecimalPrecision)

    assert isinstance(s, ir.IntegerValue)
    assert isinstance(s.op(), ops.DecimalScale)


@pytest.mark.xfail(raises=AssertionError, reason='NYI')
def test_invalid_precision_scale_combo():
    assert False


def test_decimal_str(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    assert str(t) == 'decimal({0:d}, {1:d})'.format(t.precision, t.scale)


def test_decimal_repr(lineitem):
    col = lineitem.l_extendedprice
    t = col.type()
    assert repr(t) == 'Decimal(precision={0:d}, scale={1:d})'.format(
        t.precision,
        t.scale,
    )
