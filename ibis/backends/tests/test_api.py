def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


def test_list_databases(con):
    # every backend has a different set of databases, not testing the
    # exact names for now
    databases = con.list_databases()
    # assert isinstance(databases, list)
    assert databases == []
    assert all(isinstance(database, str) for database in databases)
