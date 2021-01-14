def test_backend_name(backend):
    assert backend.api.Backend.name == backend.name()
