def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


def test_current_database(con):
    current_database = con.current_database
    assert isinstance(current_database, str) or current_database is None
