import pytest
import ibis


pytestmark = pytest.mark.mapd
pytest.importorskip('pymapd')


def test_literal_execute(client):
    expected = '1234'
    expr = ibis.literal(expected)
    result = client.execute(expr)
    assert result == expected

