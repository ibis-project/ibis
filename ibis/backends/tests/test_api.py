def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()


def test_version(backend):
    assert isinstance(backend.api.version, str)


def test_database_consistency(con):
    # every backend has a different set of databases, not testing the
    # exact names for now
    databases = con.list_databases()
    assert isinstance(databases, list)
    assert len(databases) >= 1
    assert all(isinstance(database, str) for database in databases)

    current_database = con.current_database
    assert isinstance(current_database, str)
    assert current_database in databases

    if len(databases) == 1:
        new_database = current_database
    else:
        new_database = next(db for db in databases if db != current_database)

    try:
        con.set_database(new_database)
    except NotImplementedError:
        pass
    else:
        assert con.current_database == new_database
        assert con.list_databases() == databases

    # restoring the original database, in case the same connection is used
    # in other tests
    try:
        con.set_database(current_database)
    except NotImplementedError:
        pass


def test_list_tables(con):
    tables = con.list_tables()
    assert isinstance(tables, list)
    assert 'functional_alltypes' in tables
    assert 'batting' in tables
    # impala doesn't seem to use 'awards_players'
    # assert 'awards_players' in tables
    assert all(isinstance(table, str) for table in tables)
