def test_backend_name(backend):
    # backend is the TestConf for the backend
    assert backend.api.name == backend.name()
