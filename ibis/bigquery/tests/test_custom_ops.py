import pytest

from ibis.bigquery import compiler as bq_comp


pytestmark = pytest.mark.bigquery
pytest.importorskip('google.cloud.bigquery')


def test_approx_quantile(alltypes):
    N = 10
    expr = bq_comp.approx_quantile(alltypes.float_col, N)
    result = expr.compile()
    expected = """\
SELECT APPROX_QUANTILES(`float_col`, {}) AS `tmp`
FROM testing.functional_alltypes""".format(N)
    assert result == expected
