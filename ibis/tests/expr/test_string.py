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
from __future__ import annotations

import pytest

import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import literal
from ibis.tests.util import assert_equal


def test_lower_upper(table):
    lresult = table.g.lower()
    uresult = table.g.upper()

    assert isinstance(lresult, ir.StringColumn)
    assert isinstance(uresult, ir.StringColumn)

    assert isinstance(lresult.op(), ops.Lowercase)
    assert isinstance(uresult.op(), ops.Uppercase)

    lit = literal("FoO")

    lresult = lit.lower()
    uresult = lit.upper()
    assert isinstance(lresult, ir.StringScalar)
    assert isinstance(uresult, ir.StringScalar)


def test_substr(table):
    lit = literal("FoO")

    result = table.g.substr(2, 4)
    lit_result = lit.substr(0, 2)

    assert isinstance(result, ir.StringColumn)
    assert isinstance(lit_result, ir.StringScalar)

    op = result.op()
    assert isinstance(op, ops.Substring)

    start, length = op.args[1:]

    assert start == literal(2).op()
    assert length == literal(4).op()


def test_left_right(table):
    result = table.g.left(5)
    expected = table.g.substr(0, 5)
    assert result.equals(expected)

    result = table.g.right(5)
    op = result.op()
    assert isinstance(op, ops.StrRight)
    assert op.args[1] == literal(5).op()


def test_length(table):
    lit = literal("FoO")
    result = table.g.length()
    lit_result = lit.length()

    assert isinstance(result, ir.IntegerColumn)
    assert isinstance(lit_result, ir.IntegerScalar)
    assert isinstance(result.op(), ops.StringLength)


def test_join(table):
    dash = literal("-")

    expr = dash.join([table.f.cast("string"), table.g])
    assert isinstance(expr, ir.StringColumn)

    expr = dash.join([literal("ab"), literal("cd")])
    assert isinstance(expr, ir.StringScalar)


def test_contains(table):
    expr = table.g.contains("foo")
    assert isinstance(expr.op(), ops.StringContains)

    with pytest.raises(TypeError):
        "foo" in table.g  # noqa: B015


@pytest.mark.parametrize(
    ("left_slice", "right_start", "right_stop"),
    [(slice(None, 3), 0, 3), (slice(2, 6), 2, 4)],
)
def test_getitem_slice(table, left_slice, right_start, right_stop):
    case = table.g[left_slice]
    expected = table.g.substr(right_start, right_stop)
    assert_equal(case, expected)


def test_add_radd(table, string_col):
    string_col = table[string_col]
    assert isinstance(literal("foo") + "bar", ir.StringScalar)
    assert isinstance("bar" + literal("foo"), ir.StringScalar)
    assert isinstance(string_col + "bar", ir.StringColumn)
    assert isinstance("bar" + string_col, ir.StringColumn)


def test_startswith(table):
    assert isinstance(table.g.startswith("foo"), ir.BooleanColumn)
    assert isinstance(literal("bar").startswith("foo"), ir.BooleanScalar)


def test_endswith(table):
    assert isinstance(table.g.endswith("foo"), ir.BooleanColumn)
    assert isinstance(literal("bar").endswith("foo"), ir.BooleanScalar)
