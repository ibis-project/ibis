import pytest

from ibis.config import options


def test_sql_config():
    assert options.sql.default_limit == 10_000

    with pytest.raises(TypeError):
        options.sql.default_limit = -1

    options.sql.default_limit = 100
    assert options.sql.default_limit == 100
    options.sql.default_limit = 10_000
