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
    (lambda t: t[t.string_col, t.float_col],
     [('string_col', 'string'), ('float_col', 'float')])
])
def test_query_schema(backend, con, alltypes, expr_fn, expected):
    if not hasattr(con, '_build_ast'):
        pytest.skip()
    if backend.name == 'bigquery':
        pytest.skip()

    expr = expr_fn(alltypes)
    # we might need a public API for it
    query = con.sync_query(con, con._build_ast(expr).queries[0])

    # test single columns
    expected = ibis.schema(expected)
    assert query.schema().equals(expected)
