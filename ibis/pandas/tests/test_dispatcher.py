import pytest

from ibis.pandas.dispatcher import TwoLevelDispatcher


@pytest.fixture
def foo():
    dispatcher = TwoLevelDispatcher('foo', 'test_dispatcher')

    @dispatcher.register(int, int)
    def foo0(x, y):
        return 0

    @dispatcher.register(int, float)
    def foo1(x, y):
        return 1

    @dispatcher.register(float, int)
    def foo2(x, y):
        return 2

    @dispatcher.register(float, float)
    def foo3(x, y):
        return 3

    @dispatcher.register((list, tuple),)
    def foo4(x):
        return 4

    return dispatcher


def test_basic(foo):
    assert foo(0, 0) == 0
    assert foo(0, 0.0) == 1
    assert foo(0.0, 0) == 2
    assert foo(0.0, 0.0) == 3
    assert foo(list()) == 4
    assert foo(tuple()) == 4
