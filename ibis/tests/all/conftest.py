import operator

import pytest

import ibis.common as com
from ibis.tests.backends import Backend, Spark


def subclasses(cls):
    """Get all child classes of `cls` not including `cls`, transitively."""
    assert isinstance(cls, type), "cls is not a class, type: {}".format(
        type(cls)
    )
    children = set(cls.__subclasses__())
    return children.union(*map(subclasses, children))


ALL_BACKENDS = sorted(subclasses(Backend), key=operator.attrgetter("__name__"))


def pytest_runtest_call(item):
    """Dynamically add various custom markers."""
    nodeid = item.nodeid
    for marker in list(item.iter_markers(name="only_on_backends")):
        backend_types, = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        if not isinstance(backend, backend_types):
            pytest.skip(nodeid)

    for marker in list(item.iter_markers(name="skip_backends")):
        backend_types, = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        if isinstance(backend, backend_types):
            pytest.skip(nodeid)

    for marker in list(item.iter_markers(name="skip_missing_feature")):
        backend = item.funcargs["backend"]
        features, = marker.args
        missing_features = [
            feature for feature in features if not getattr(backend, feature)
        ]
        if missing_features:
            pytest.mark.skip(
                ('Backend {} is missing features {} needed to run {}').format(
                    type(backend).__name__, ', '.join(missing_features), nodeid
                )
            )

    for marker in list(item.iter_markers(name="xfail_backends")):
        backend_types, = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        item.add_marker(
            pytest.mark.xfail(
                condition=isinstance(backend, backend_types),
                reason='Backend {} does not pass this test'.format(
                    type(backend).__name__
                ),
                **marker.kwargs,
            )
        )

    for marker in list(item.iter_markers(name="xpass_backends")):
        backend_types, = map(tuple, marker.args)
        backend = item.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        item.add_marker(
            pytest.mark.xfail(
                condition=not isinstance(backend, backend_types),
                reason='{} does not pass this test'.format(
                    type(backend).__name__
                ),
                **marker.kwargs,
            )
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """Dynamically add an xfail marker for specific backends."""
    outcome = yield
    try:
        outcome.get_result()
    except (
        com.OperationNotDefinedError,
        com.UnsupportedOperationError,
        com.UnsupportedBackendType,
        NotImplementedError,
    ) as e:
        markers = list(pyfuncitem.iter_markers(name="xfail_unsupported"))
        assert (
            len(markers) == 1
        ), "More than one xfail_unsupported marker found on test {}".format(
            pyfuncitem
        )
        marker, = markers
        backend = pyfuncitem.funcargs["backend"]
        assert isinstance(backend, Backend), "backend has type {!r}".format(
            type(backend).__name__
        )
        pytest.xfail(reason='{}: {}'.format(type(backend).__name__, e))


pytestmark = pytest.mark.backend

params_backend = [
    pytest.param(backend, marks=getattr(pytest.mark, backend.__name__.lower()))
    for backend in ALL_BACKENDS
]


@pytest.fixture(params=params_backend, scope='session')
def backend(request, data_directory, spark_client_testing):
    if request.param is Spark:
        Spark.client_testing = spark_client_testing
    return request.param(data_directory)


@pytest.fixture(scope='session')
def con(backend):
    return backend.connection


@pytest.fixture(scope='session')
def alltypes(backend):
    return backend.functional_alltypes


@pytest.fixture(scope='session')
def sorted_alltypes(alltypes):
    return alltypes.sort_by('id')


@pytest.fixture(scope='session')
def batting(backend):
    return backend.batting


@pytest.fixture(scope='session')
def awards_players(backend):
    return backend.awards_players


@pytest.fixture(scope='session')
def geo(backend):
    return backend.geo


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
    return None
