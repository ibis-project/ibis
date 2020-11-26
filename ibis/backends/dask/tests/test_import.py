# Trivial test to make sure CI runs and package is importable


def test_import():
    import ibis.backends.dask  # noqa: F401
