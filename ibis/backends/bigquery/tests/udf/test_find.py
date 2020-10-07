import ast

import pytest

from ibis.compat import PY38
from ibis.util import is_iterable

from .udf.find import find_names

if PY38:
    # ref: https://github.com/ibis-project/ibis/issues/2098
    # note: UDF is already skipt on CI
    pytestmark = [pytest.mark.bigquery, pytest.mark.udf]
else:
    pytestmark = pytest.mark.bigquery


def parse_expr(expr):
    body = parse_stmt(expr)
    return body.value


def parse_stmt(stmt):
    (body,) = ast.parse(stmt).body
    return body


def eq(left, right):
    if type(left) != type(right):
        return False

    if is_iterable(left) and is_iterable(right):
        return all(map(eq, left, right))

    if not isinstance(left, ast.AST) and not isinstance(right, ast.AST):
        return left == right

    assert hasattr(left, '_fields') and hasattr(right, '_fields')
    return left._fields == right._fields and all(
        eq(getattr(left, left_name), getattr(right, right_name))
        for left_name, right_name in zip(left._fields, right._fields)
    )


def var(id):
    return ast.Name(id=id, ctx=ast.Load())


def store(id):
    return ast.Name(id=id, ctx=ast.Store())


def test_find_BinOp():
    expr = parse_expr('a + 1')
    found = find_names(expr)
    assert len(found) == 1
    assert eq(found[0], var('a'))


def test_find_dup_names():
    expr = parse_expr('a + 1 * a')
    found = find_names(expr)
    assert len(found) == 1
    assert eq(found[0], var('a'))


def test_find_Name():
    expr = parse_expr('b')
    found = find_names(expr)
    assert len(found) == 1
    assert eq(found[0], var('b'))


def test_find_Tuple():
    expr = parse_expr('(a, (b, 1), (((c,),),))')
    found = find_names(expr)
    assert len(found) == 3
    assert eq(found, [var('a'), var('b'), var('c')])


def test_find_Compare():
    expr = parse_expr('a < b < c == e + (f, (gh,))')
    found = find_names(expr)
    assert len(found) == 6
    assert eq(
        found, [var('a'), var('b'), var('c'), var('e'), var('f'), var('gh')]
    )


def test_find_ListComp():
    expr = parse_expr('[i for i in range(n) if i < 2]')
    found = find_names(expr)
    assert all(isinstance(f, ast.Name) for f in found)
    assert eq(found, [var('i'), store('i'), var('n')])
