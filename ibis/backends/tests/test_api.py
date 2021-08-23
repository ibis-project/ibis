import pytest


def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


@pytest.mark.xfail_unsupported
def test_set_database(con):
    current_database = con.current_database
    if current_database is None:
        # Backend does not support multiple databases
        return
    databases = con.list_databases()
    if len(databases) < 2:
        # If there are no more databases, we set the database
        # again to the current database. While not a perfect
        # test, at least this should call the `set_database` code
        another_database = current_database
    else:
        another_database = databases.remove(current_database)

    con.set_database(another_database)
    assert con.current_database == another_database
