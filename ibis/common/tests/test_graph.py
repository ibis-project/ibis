from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from ibis.common.collections import frozendict
from ibis.common.graph import (
    Graph,
    Node,
    _flatten_collections,
    _recursive_lookup,
    bfs,
    dfs,
    toposort,
)
from ibis.common.grounds import Annotable, Concrete
from ibis.common.patterns import Eq, If, InstanceOf, Object, TupleOf, _


class MyNode(Node):
    __match_args__ = ("name", "children")
    __slots__ = ("name", "children")

    def __init__(self, name, children):
        self.name = name
        self.children = children

    @property
    def __args__(self):
        return (self.children,)

    @property
    def __argnames__(self):
        return ("children",)

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
    assert bfs(e) == {
        e: (b, c, d),
        b: (a,),
        c: (),
        d: (),
        a: (),
    }


@pytest.mark.parametrize("func", [bfs, dfs, Graph.from_bfs, Graph.from_dfs])
def test_traversals_with_filter(func):
    graph = func(A, filter=If(lambda x: x.name != "B"))
    assert graph == {A: (C,), C: ()}

    graph = func(A, filter=If(lambda x: x.name != "D"))
    assert graph == {E: (), B: (E,), A: (B, C), C: ()}


@pytest.mark.parametrize("func", [bfs, dfs, Graph.from_bfs, Graph.from_dfs])
def test_traversal_with_filtering_out_root(func):
    graph = func(A, filter=If(lambda x: x.name != "A"))
    assert graph == {}


def test_replace_with_filtering_out_root():
    rule = InstanceOf(MyNode) >> MyNode(name="new", children=[])
    result = A.replace(rule, filter=If(lambda x: x.name != "A"))
    assert result == A


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
    assert bfs(node) == {node: (T, F), T: (), F: ()}

    node = Either(T, F)
    assert node.__args__ == (T, F)
    assert bfs(node) == {node: (T, F), T: (), F: ()}

    node = All((T, Either(T, Either(T, F))), strict=False)
    assert node.__args__ == ((T, Either(T, Either(T, F))), False)
    assert bfs(node) == {
        node: (T, Either(T, Either(T, F))),
        T: (),
        F: (),
        Either(T, Either(T, F)): (T, Either(T, F)),
        Either(T, F): (T, F),
    }

    copied = node.copy(arguments=(T, F))
    assert copied == All((T, F), strict=False)


class MySequence(Sequence):
    def __init__(self, *items):
        self.items = items

    def __getitem__(self, index):
        raise AssertionError("must not be called")  # pragma: no cover

    def __len__(self):
        return len(self.items)


class MyMapping(Mapping):
    def __init__(self, **items):
        self.items = items

    def __getitem__(self, key):
        raise AssertionError("must not be called")  # pragma: no cover

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


def test_flatten_collections():
    # test that flatten collections doesn't recurse into arbitrary mappings
    # and sequences, just the commonly used builtin ones: list, tuple, dict

    result = _flatten_collections([0.0, A, B, [C, D, (E, 6)], "7", MySequence(8, A)])
    assert list(result) == [A, B, C, D, E]

    result = _flatten_collections(
        {
            "a": 0.0,
            "b": A,
            "c": (MyMapping(d=B, e=3), frozendict(f=C)),
            "d": [5, "6", {"e": (D, 8.9)}],
        }
    )
    assert list(result) == [A, C, D]


def test_recursive_lookup():
    results = {A: "A", B: "B", C: "C", D: "D"}

    assert _recursive_lookup((A, B, "a", {"b": C}), results) == (
        "A",
        "B",
        "a",
        {"b": "C"},
    )
    assert _recursive_lookup({"a": B, "c": D}, results) == {
        "a": "B",
        "c": "D",
    }
    assert _recursive_lookup([A, B, "c"], results) == ("A", "B", "c")
    assert _recursive_lookup(A, results) == "A"

    my_seq = MySequence(A, "b", "c")
    my_map = MyMapping(a="a", b=B, c="c")
    assert _recursive_lookup((A, my_seq, [B, A], my_map), results) == (
        "A",
        my_seq,
        ("B", "A"),
        my_map,
    )


def test_node_match():
    result = A.match(If(_.name == "C"))
    assert result == {C}

    result = A.match(Object(MyNode, name=Eq("D")))
    assert result == {D}

    result = A.match(If(_.children))
    assert result == {A, B}
