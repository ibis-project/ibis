from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from ibis.common.collections import frozendict
from ibis.common.graph import (
    Graph,
    Node,
    _coerce_finder,
    _coerce_replacer,
    _flatten_collections,
    _recursive_lookup,
    bfs,
    bfs_while,
    dfs,
    dfs_while,
    traverse,
)
from ibis.common.grounds import Annotable, Concrete
from ibis.common.patterns import Eq, If, InstanceOf, Object, TupleOf, _, pattern


class MyNode(Node):
    __match_args__ = ("name", "children")
    __slots__ = ("children", "name")

    def __init__(self, name, children):
        self.name = name
        self.children = children

    @property
    def __args__(self):
        return (self.name, self.children)

    @property
    def __argnames__(self):
        return ("name", "children")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __eq__(self, other):
        return self.name == other.name

    def copy(self, name=None, children=None):
        return self.__class__(name or self.name, children or self.children)


C = MyNode(name="C", children=[])
D = MyNode(name="D", children=[])
E = MyNode(name="E", children=[])
B = MyNode(name="B", children=[D, E])
A = MyNode(name="A", children=[B, C])
F = MyNode(name="F", children=[{C: D, E: None}])


def test_bfs():
    assert list(bfs(A).keys()) == [A, B, C, D, E]
    assert list(bfs([D, E, B])) == [D, E, B]
    assert bfs(1) == {}


def test_construction():
    assert Graph(A) == bfs(A)


def test_graph_nodes():
    assert Graph(A).nodes() == {A, B, C, D, E}
    assert Graph(F).nodes() == {F, C, D, E}


def test_graph_repr():
    g = Graph(A)
    assert repr(g) == f"Graph({dict(g)})"


def test_dfs():
    assert list(dfs(A).keys()) == [D, E, B, C, A]
    assert list(dfs([D, E, B])) == [D, E, B]
    assert dfs(1) == {}


def test_invert():
    g = dfs(A)
    assert g == {D: (), E: (), B: (D, E), C: (), A: (B, C)}

    i = g.invert()
    assert i == {D: (B,), E: (B,), B: (A,), C: (A,), A: ()}

    j = i.invert()
    assert j == g


def test_toposort():
    g, dependents = Graph(A).toposort()
    assert list(g.keys()) == [C, D, E, B, A]
    assert dependents == Graph(A).invert()


def test_toposort_cycle_detection():
    C = MyNode(name="C", children=[])
    A = MyNode(name="A", children=[C])
    B = MyNode(name="B", children=[A])
    A.children.append(B)

    # A depends on B which depends on A
    g = Graph(A)
    with pytest.raises(ValueError, match="cycle detected in the graph"):
        g.toposort()


def test_nested_children():
    a = MyNode(name="a", children=[])
    b = MyNode(name="b", children=[a])
    c = MyNode(name="c", children=[])
    d = MyNode(name="d", children=[])
    e = MyNode(name="e", children=[[b, c], {"d": d}])
    assert bfs(e) == {
        e: (b, c, d),
        b: (a,),
        c: (),
        d: (),
        a: (),
    }

    assert a.__children__ == ()
    assert b.__children__ == (a,)
    assert c.__children__ == ()
    assert d.__children__ == ()
    assert e.__children__ == (b, c, d)


@pytest.mark.parametrize("func", [bfs_while, dfs_while, Graph.from_bfs, Graph.from_dfs])
def test_traversals_with_filter(func):
    graph = func(A, filter=lambda x: x.name != "B")
    assert graph == {A: (C,), C: ()}

    graph = func(A, filter=lambda x: x.name != "D")
    assert graph == {E: (), B: (E,), A: (B, C), C: ()}


@pytest.mark.parametrize("func", [bfs_while, dfs_while, Graph.from_bfs, Graph.from_dfs])
def test_traversal_with_filtering_out_root(func):
    graph = func(A, filter=lambda x: x.name != "A")
    assert graph == {}


def test_replace_with_filtering_out_root():
    rule = InstanceOf(MyNode) >> MyNode(name="new", children=[])
    result = A.replace(rule, filter=If(lambda x: x.name != "A"))
    assert result == A


def test_replace_with_mapping():
    new_E = MyNode(name="e", children=[])
    new_D = MyNode(name="d", children=[])
    new_B = MyNode(name="B", children=[new_D, new_E])
    new_A = MyNode(name="A", children=[new_B, C])

    subs = {
        E: new_E,
        D: new_D,
    }
    result = A.replace(subs)
    assert result == new_A


@pytest.mark.parametrize("kind", ["pattern", "mapping", "function"])
def test_replace_doesnt_recreate_unchanged_nodes(kind):
    A1 = MyNode(name="A1", children=[])
    A2 = MyNode(name="A2", children=[A1])
    B1 = MyNode(name="B1", children=[])
    B2 = MyNode(name="B2", children=[B1])
    C = MyNode(name="C", children=[A2, B2])

    B3 = MyNode(name="B3", children=[])

    if kind == "pattern":
        replacer = pattern(MyNode)(name="B2") >> B3
    elif kind == "mapping":
        replacer = {B2: B3}
    else:

        def replacer(node, children):
            if node is B2:
                return B3
            return node.__recreate__(children) if children else node

    res = C.replace(replacer)

    assert res is not C
    assert res.name == "C"
    assert len(res.children) == 2
    assert res.children[0] is A2
    assert res.children[1] is B3


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
        [0.0, A, (MyMapping(d=B, e=3), frozendict(f=C)), [5, "6", {"e": (D, 8.9)}]]
    )
    assert list(result) == [A, C, D]

    # test that dictionary keys are also flattened
    result = _flatten_collections([0.0, {A: B, C: [D]}, frozendict({E: 6})])
    assert list(result) == [A, B, C, D, E]


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

    # test that dictionary nodes as dictionary keys are also looked up
    assert _recursive_lookup({A: B, C: D}, results) == {"A": "B", "C": "D"}


def test_coerce_finder():
    f = _coerce_finder(int)
    assert f(1) is True
    assert f("1") is False

    f = _coerce_finder((int, str))
    assert f(1) is True
    assert f("1") is True
    assert f(1.0) is False

    f = _coerce_finder(InstanceOf(bool))
    assert f(True) is True
    assert f(False) is True
    assert f(1) is False

    f = _coerce_finder(lambda x: x == 1)
    assert f(1) is True
    assert f(2) is False


def test_coerce_replacer():
    r = _coerce_replacer(lambda _, children: D if children else C)
    assert r(C, {"children": []}) is D
    assert r(C, None) is C

    r = _coerce_replacer({C: D, D: E})
    assert r(C, {}) == D
    assert r(D, {}) == E
    assert r(A, {"name": "A", "children": [B, C]}) == A

    r = _coerce_replacer(InstanceOf(MyNode) >> _.copy(name=_.name.lower()))
    assert r(C, {"name": "C", "children": []}) == MyNode(name="c", children=[])
    assert r(D, {"name": "D", "children": []}) == MyNode(name="d", children=[])


def test_node_find_using_type():
    class FooNode(MyNode):
        pass

    class BarNode(MyNode):
        pass

    C = BarNode(name="C", children=[])
    D = FooNode(name="D", children=[])
    E = BarNode(name="E", children=[])
    B = FooNode(name="B", children=[D, E])
    A = MyNode(name="A", children=[B, C])

    result = A.find(MyNode)
    assert result == [A, B, C, D, E]

    result = A.find(FooNode)
    assert result == [B, D]

    result = A.find(BarNode)
    assert result == [C, E]

    result = A.find((FooNode, BarNode))
    assert result == [B, C, D, E]

    result = A.find(int)
    assert result == []


def test_node_find_using_pattern():
    result = A.find(If(_.name == "C"))
    assert result == [C]

    result = A.find(Object(MyNode, name=Eq("D")))
    assert result == [D]

    result = A.find(If(_.children))
    assert result == [A, B]


def test_node_find_below():
    lowercase = MyNode(name="lowercase", children=[])
    root = MyNode(name="root", children=[A, B, lowercase])
    result = root.find_below(MyNode)
    assert result == [A, B, lowercase, C, D, E]

    result = root.find_below(lambda x: x.name.islower(), filter=lambda x: x != root)
    assert result == [lowercase]


def test_node_find_topmost_using_type():
    class FooNode(MyNode):
        pass

    G = FooNode(name="G", children=[A, B])
    E = MyNode(name="E", children=[G, G, A])

    result = E.find_topmost(FooNode)
    assert result == [G]

    result = E.find_topmost((FooNode, MyNode))
    assert result == [E]


def test_node_find_topmost_using_pattern():
    G = MyNode(name="G", children=[A, B])
    E = MyNode(name="E", children=[G, G, A])

    result = E.find_topmost(Object(MyNode, name="G") | Object(MyNode, name="B"))
    expected = [G, B]
    assert result == expected


def test_node_find_topmost_dont_traverse_the_same_node_twice():
    G = MyNode(name="G", children=[A, B])
    E = MyNode(name="E", children=[G, G, A])

    result = E.find_topmost(If(_.name == "G"))
    expected = [G]
    assert result == expected


def test_map_clear():
    Z = MyNode(name="Z", children=[A, A])
    Y = MyNode(name="Y", children=[A])
    X = MyNode(name="X", children=[Z, Y])
    result_sequence = {}

    def record_result_keys(node, results, **_):
        result_sequence[node] = tuple(results.keys())
        return node

    expected_result_sequence = {
        C: (),
        D: (C,),
        E: (C, D),
        B: (C, D, E),
        A: (C, B),
        Z: (A,),
        Y: (A, Z),
        X: (Z, Y),
    }
    result = X.map_clear(record_result_keys)
    assert result == X
    assert result_sequence == expected_result_sequence


def test_traverse():
    def walker(node):
        return True, node

    result = list(traverse(walker, A))
    assert result == [A, B, D, E, C]
