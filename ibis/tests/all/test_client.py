import pytest

import ibis
import ibis.tests.util as tu
from ibis.compat import parse_version


@tu.skipif_unsupported
def test_version(backend, con):
    expected_type = (type(parse_version('1.0')),
                     type(parse_version('1.0-legacy')))
    assert isinstance(con.version, expected_type)


@pytest.mark.parametrize(('expr_fn', 'expected'), [
    (lambda t: t.string_col, [('string_col', 'string')]),
    (lambda t: t[t.string_col, t.bigint_col],
     [('string_col', 'string'), ('bigint_col', 'int64')])
])
def test_query_schema(backend, con, alltypes, expr_fn, expected):
    if not hasattr(con, '_build_ast'):
        pytest.skip(
            '{} backend has no _build_ast method'.format(
                type(backend).__name__
            )
        )

    expr = expr_fn(alltypes)

    # we might need a public API for it
    ast = con._build_ast(expr, backend.make_context())
    query = con.sync_query(con, ast.queries[0])

    # test single columns
    expected = ibis.schema(expected)
    assert query.schema().equals(expected)
