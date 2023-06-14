from __future__ import annotations

import pytest

from ibis.common.graph import Graph, Node, bfs, dfs, toposort
from ibis.common.grounds import Annotable, Concrete
from ibis.common.patterns import InstanceOf, TupleOf


class MyNode(Node):
    def __init__(self, name, children):
        self.name = name
        self.children = children

    @property
    def __args__(self):
        return (self.children,)

    @property
    def __argnames__(self):
        return ('children',)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __eq__(self, other):
        return self.name == other.name


C = MyNode(name="C", children=[])
D = MyNode(name="D", children=[])
E = MyNode(name="E", children=[])
B = MyNode(name="B", children=[D, E])
A = MyNode(name="A", children=[B, C])


def test_bfs():
    assert list(bfs(A).keys()) == [A, B, C, D, E]

    with pytest.raises(
        TypeError, match="must be an instance of ibis.common.graph.Node"
    ):
        bfs(1)


def test_construction():
    assert Graph(A) == bfs(A)


def test_graph_nodes():
    g = Graph(A)
    assert g.nodes() == {A, B, C, D, E}


def test_graph_repr():
    g = Graph(A)
    assert repr(g) == f"Graph({dict(g)})"


def test_dfs():
    assert list(dfs(A).keys()) == [D, E, B, C, A]

    with pytest.raises(
        TypeError, match="must be an instance of ibis.common.graph.Node"
    ):
        dfs(1)


def test_invert():
    g = dfs(A)
    assert g == {D: (), E: (), B: (D, E), C: (), A: (B, C)}

    i = g.invert()
    assert i == {D: (B,), E: (B,), B: (A,), C: (A,), A: ()}

    j = i.invert()
    assert j == g


def test_toposort():
    assert list(toposort(A).keys()) == [C, D, E, B, A]


def test_toposort_cycle_detection():
    C = MyNode(name="C", children=[])
    A = MyNode(name="A", children=[C])
    B = MyNode(name="B", children=[A])
    A.children.append(B)

    # A depends on B which depends on A
    with pytest.raises(ValueError, match="cycle detected in the graph"):
        toposort(A)


def test_nested_children():
    a = MyNode(name="a", children=[])
    b = MyNode(name="b", children=[a])
    c = MyNode(name="c", children=[])
    d = MyNode(name="d", children=[])
    e = MyNode(name="e", children=[[b, c], d])

    assert e.__children__() == (b, c, d)


def test_example():
    class Example(Annotable, Node):
        def __hash__(self):
            return hash((self.__class__, self.__args__))

    class Literal(Example):
        value = InstanceOf(object)

    class BoolLiteral(Literal):
        value = InstanceOf(bool)

    class And(Example):
        operands = TupleOf(InstanceOf(BoolLiteral))

    class Or(Example):
        operands = TupleOf(InstanceOf(BoolLiteral))

    class Collect(Example):
        arguments = TupleOf(TupleOf(InstanceOf(Example)) | InstanceOf(Example))

    a = BoolLiteral(True)
    b = BoolLiteral(False)
    c = BoolLiteral(True)
    d = BoolLiteral(False)

    and_ = And((a, b, c, d))
    or_ = Or((a, c))
    collect = Collect([and_, (or_, or_)])

    graph = bfs(collect)

    expected = {
        collect: (and_, or_, or_),
        or_: (a, c),
        and_: (a, b, c, d),
        a: (),
        b: (),
        # c and d are identical with a and b
    }
    assert graph == expected


def test_concrete_with_traversable_children():
    class Bool(Concrete, Node):
        pass

    class Value(Bool):
        value = InstanceOf(bool)

    class Either(Bool):
        left = InstanceOf(Bool)
        right = InstanceOf(Bool)

    class All(Bool):
        arguments = TupleOf(InstanceOf(Bool))
        strict = InstanceOf(bool)

    T, F = Value(True), Value(False)

    node = All((T, F), strict=True)
    assert node.__args__ == ((T, F), True)
    assert node.__children__() == (T, F)

    node = Either(T, F)
    assert node.__args__ == (T, F)
    assert node.__children__() == (T, F)

    node = All((T, Either(T, Either(T, F))), strict=False)
    assert node.__args__ == ((T, Either(T, Either(T, F))), False)
    assert node.__children__() == (T, Either(T, Either(T, F)))

    copied = node.copy(arguments=(T, F))
    assert copied == All((T, F), strict=False)
