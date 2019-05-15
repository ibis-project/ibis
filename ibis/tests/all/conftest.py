import operator

import pytest

from ibis.tests.backends import Backend


def subclasses(cls):
    """Get all child classes of `cls` not including `cls`, transitively."""
    assert isinstance(cls, type), "cls is not a class, type: {}".format(
        type(cls)
    )
    children = set(cls.__subclasses__())
    return children.union(*map(subclasses, children))


ALL_BACKENDS = sorted(subclasses(Backend), key=operator.attrgetter("__name__"))


def pytest_runtest_call(item):
    """Dynamically add an xfail marker for specific backends."""
    for marker in list(item.iter_markers(name="xfail_backends")):
        backend_types, = marker.args
        if isinstance(item.funcargs["backend"], tuple(backend_types)):
            item.add_marker(pytest.mark.xfail(**marker.kwargs))

    for marker in list(item.iter_markers(name="xpass_backends")):
        backend_types, = marker.args
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        if not isinstance(backend, tuple(backend_types)):
            item.add_marker(pytest.mark.xfail(**marker.kwargs))


pytestmark = pytest.mark.backend

params_backend = [
    pytest.param(backend, marks=getattr(pytest.mark, backend.__name__.lower()))
    for backend in ALL_BACKENDS
]


@pytest.fixture(params=params_backend, scope='session')
def backend(request, data_directory):
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes()


@pytest.fixture(scope='session')
def sorted_alltypes(alltypes):
    return alltypes.sort_by('id')


@pytest.fixture(scope='session')
def batting(backend):
    return backend.batting()


@pytest.fixture(scope='session')
def awards_players(backend):
    return backend.awards_players()


@pytest.fixture(scope='session')
def geo(backend):
    return backend.geo()


@pytest.fixture
def analytic_alltypes(alltypes):
    return alltypes


@pytest.fixture(scope='session')
def df(alltypes):
    return alltypes.execute()


@pytest.fixture(scope='session')
def sorted_df(df):
    return df.sort_values('id').reset_index(drop=True)


@pytest.fixture(scope='session')
def batting_df(batting):
    return batting.execute(limit=None)


@pytest.fixture(scope='session')
def awards_players_df(awards_players):
    return awards_players.execute(limit=None)


@pytest.fixture(scope='session')
def geo_df(geo):
    # Currently geo is implemented just for MapD
    if geo is not None:
        return geo.execute(limit=None)
