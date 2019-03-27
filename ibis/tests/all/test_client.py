import pytest

import ibis
import ibis.tests.util as tu
import ibis.expr.datatypes as dt
from ibis.compat import parse_version


@tu.skipif_unsupported
def test_version(backend, con):
    expected_type = (type(parse_version('1.0')),
                     type(parse_version('1.0-legacy')))
    assert isinstance(con.version, expected_type)


@pytest.mark.parametrize(('expr_fn', 'expected'), [
    (lambda t: t.string_col, [('string_col', dt.String)]),
    (lambda t: t[t.string_col, t.bigint_col],
     [('string_col', dt.String), ('bigint_col', dt.Int64)])
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
    query = con.query_class(con, ast)
    schema = query.schema()

    # clickhouse columns has been defined as non-nullable
    # whereas other backends don't support non-nullable columns yet
    expected = ibis.schema([(name, dtype(nullable=schema[name].nullable))
                            for name, dtype in expected])
    assert query.schema().equals(expected)
