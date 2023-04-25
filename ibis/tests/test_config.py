import pytest

from ibis.config import options


def test_sql_config(monkeypatch):
    assert options.sql.default_limit is None

    with pytest.raises(TypeError):
        options.sql.default_limit = -1

    monkeypatch.setattr(options.sql, 'default_limit', 100)
    assert options.sql.default_limit == 100
