

def test_version(backend, con):
    assert isinstance(con.version, tuple)
