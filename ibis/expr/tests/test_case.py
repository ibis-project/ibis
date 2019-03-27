import pytest

import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis

from ibis.tests.util import assert_equal


def test_ifelse(table):
    bools = table.g.isnull()
    result = bools.ifelse("foo", "bar")
    assert isinstance(result, ir.StringColumn)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_ifelse_literal():
    assert False


def test_simple_case_expr(table):
    case1, result1 = "foo", table.a
    case2, result2 = "bar", table.c
    default_result = table.b

    expr1 = table.g.lower().cases(
        [(case1, result1),
         (case2, result2)],
        default=default_result
    )

    expr2 = (table.g.lower().case()
             .when(case1, result1)
             .when(case2, result2)
             .else_(default_result)
             .end())

    assert_equal(expr1, expr2)
    assert isinstance(expr1, ir.IntegerColumn)


def test_multiple_case_expr(table):
    case1 = table.a == 5
    case2 = table.b == 128
    case3 = table.c == 1000

    result1 = table.f
    result2 = table.b * 2
    result3 = table.e

    default = table.d

    expr = (ibis.case()
            .when(case1, result1)
            .when(case2, result2)
            .when(case3, result3)
            .else_(default)
            .end())

    op = expr.op()
    assert isinstance(expr, ir.FloatingColumn)
    assert isinstance(op, ops.SearchedCase)
    assert op.default is default


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_simple_case_no_default():
    # TODO: this conflicts with the null else cases below. Make a decision
    # about what to do, what to make the default behavior based on what the
    # user provides. SQL behavior is to use NULL when nothing else
    # provided. The .replace convenience API could use the field values as
    # the default, getting us around this issue.
    assert False


def test_simple_case_null_else(table):
    expr = table.g.case().when("foo", "bar").end()
    op = expr.op()

    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default, ir.ValueExpr)
    assert isinstance(op.default.op(), ops.NullLiteral)


def test_multiple_case_null_else(table):
    expr = ibis.case().when(table.g == "foo", "bar").end()
    op = expr.op()

    assert isinstance(expr, ir.StringColumn)
    assert isinstance(op.default, ir.ValueExpr)
    assert isinstance(op.default.op(), ops.NullLiteral)


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_case_type_precedence():
    assert False


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_no_implicit_cast_possible():
    assert False


def test_case_mixed_type():
    t0 = ibis.table(
        [('one', 'string'),
         ('two', 'double'),
         ('three', 'int32')], name='my_data')

    expr = (
        t0.three
          .case()
          .when(0, 'low')
          .when(1, 'high')
          .else_('null')
          .end()
          .name('label'))
    result = t0[expr]
    assert result['label'].type().equals(dt.string)
