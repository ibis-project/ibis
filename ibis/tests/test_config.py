import pytest

from ibis.config import options


def test_sql_config():
    try:
        assert options.sql.default_limit is None

        with pytest.raises(TypeError):
            options.sql.default_limit = -1

        options.sql.default_limit = 100
        assert options.sql.default_limit == 100
    finally:
        options.sql.default_limit = None
