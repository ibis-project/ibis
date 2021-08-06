def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


def test_list_tables(backend):
    tables = backend.api.list_tables()
    assert isinstance(tables, list)
    assert 'awards_players' in tables
    assert 'batting' in tables
    assert 'functional_alltypes' in tables
    assert all(isinstance(table, str) for table in tables)
