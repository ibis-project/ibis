import pytest

pytest.importorskip('multipledispatch')

from ibis.pandas.execution import execute, execute_node  # noqa: E402
from multipledispatch.conflict import ambiguities  # noqa: E402

pytestmark = pytest.mark.pandas


@pytest.mark.parametrize('func', [execute, execute_node])
def test_no_execute_ambiguities(func):
    assert not ambiguities(func.funcs)
