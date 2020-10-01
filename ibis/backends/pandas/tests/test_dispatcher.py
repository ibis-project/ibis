import pytest
from multipledispatch import Dispatcher
from multipledispatch.conflict import AmbiguityWarning

from ..dispatcher import TwoLevelDispatcher


class A1(object):
    pass


class A2(A1):
    pass


class A3(A2):
    pass


class B1(object):
    pass


class B2(B1):
    pass


class B3(B2):
    pass


@pytest.fixture(scope='module')
def foo_dispatchers():

    foo = TwoLevelDispatcher('foo', doc='Test dispatcher foo')
    foo_m = Dispatcher('foo_m', doc='Control dispatcher foo_m')

    @foo.register(A1, B1)
    @foo_m.register(A1, B1)
    def foo0(x, y):
        return 0

    @foo.register(A1, B2)
    @foo_m.register(A1, B2)
    def foo1(x, y):
        return 1

    @foo.register(A2, B1)
    @foo_m.register(A2, B1)
    def foo2(x, y):
        return 2

    @foo.register(A2, B2)
    @foo_m.register(A2, B2)
    def foo3(x, y):
        return 3

    @foo.register((A1, A2),)
    @foo_m.register((A1, A2),)
    def foo4(x):
        return 4

    return foo, foo_m


@pytest.fixture(scope='module')
def foo(foo_dispatchers):
    return foo_dispatchers[0]


@pytest.fixture(scope='module')
def foo_m(foo_dispatchers):
    return foo_dispatchers[1]


def test_cache(foo, mocker):
    """Test that cache is properly set after calling with args."""

    spy = mocker.spy(foo, 'dispatch')
    a1, b1 = A1(), B1()

    assert (A1, B1) not in foo._cache
    foo(a1, b1)
    assert (A1, B1) in foo._cache
    foo(a1, b1)
    spy.assert_called_once_with(A1, B1)


def test_dispatch(foo, mocker):
    """Test that calling dispatcher with a signature that is registered
    does not trigger a linear search through dispatch_iter."""

    spy = mocker.spy(foo, 'dispatch_iter')

    # This should not trigger a linear search
    foo(A1(), B1())
    assert not spy.called, (
        "Calling dispatcher with registered signature should "
        "not trigger linear search"
    )

    foo(A3(), B3())
    spy.assert_called_once_with(A3, B3)


@pytest.mark.parametrize(
    'args',
    [
        (A1(), B1()),
        (A1(), B2()),
        (A1(), B3()),
        (A2(), B1()),
        (A2(), B2()),
        (A2(), B3()),
        (A3(), B1()),
        (A3(), B2()),
        (A3(), B3()),
        (A1(),),
        (A2(),),
        (A3(),),
    ],
)
def test_registered(foo_dispatchers, args):
    foo, foo_m = foo_dispatchers
    assert foo(*args) == foo_m(*args)


def test_ordering(foo, foo_m):
    assert foo.ordering == foo_m.ordering


def test_funcs(foo, foo_m):
    assert foo.funcs == foo_m.funcs


@pytest.mark.parametrize(
    'args', [(B1(),), (B2(),), (A1(), A1()), (A1(), A2(), A3())]
)
def test_unregistered(foo, args):
    with pytest.raises(
        NotImplementedError, match="Could not find signature for foo.*"
    ):
        foo(*args)


def test_ambiguities_warning():
    bar = TwoLevelDispatcher('bar')

    bar.register(A1, B1)(lambda a, b: 0)
    bar.register(A1, B2)(lambda a, b: 1)
    bar.register(A2, B1)(lambda a, b: 2)

    with pytest.warns(AmbiguityWarning, match=".*Consider.*\n\n.*(A2, B2).*"):
        bar.reorder()


def test_ambiguities_no_warning():
    bar = TwoLevelDispatcher('bar')

    bar.register(A1, B1)(lambda a, b: 0)
    bar.register(A1, B2)(lambda a, b: 1)
    bar.register(A2, B1)(lambda a, b: 2)
    bar.register(A2, B2)(lambda a, b: 3)

    with pytest.warns(None) as warnings:
        bar.reorder()

    assert len(warnings) == 0
