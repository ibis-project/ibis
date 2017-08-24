import pytest
import ibis

pytest.importorskip('multipledispatch')

from ibis.pandas.client import PandasTable  # noqa: E402


def test_client_table(t):
    assert isinstance(t.op(), ibis.expr.operations.DatabaseTable)
    assert isinstance(t.op(), PandasTable)
    assert 'PandasTable' in repr(t)
