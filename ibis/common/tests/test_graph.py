import pytest

from ibis.common.graph import Graph, Traversable, bfs, dfs, toposort


class Node(Traversable):
    def __init__(self, name, children):
        self.name = name
        self.children = children

    @property
    def __children__(self):
        return self.children

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self):
        return hash((self.__class__, self.name))

    def __eq__(self, other):
        return self.name == other.name


C = Node(name="C", children=[])
D = Node(name="D", children=[])
E = Node(name="E", children=[])
B = Node(name="B", children=[D, E])
A = Node(name="A", children=[B, C])


def test_bfs():
    assert list(bfs(A).keys()) == [A, B, C, D, E]

    with pytest.raises(TypeError, match="must be an instance of Traversable"):
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

    with pytest.raises(TypeError, match="must be an instance of Traversable"):
        dfs(1)


def test_invert():
    g = dfs(A)
    assert g == {D: [], E: [], B: [D, E], C: [], A: [B, C]}

    i = g.invert()
    assert i == {D: [B], E: [B], B: [A], C: [A], A: []}

    j = i.invert()
    assert j == g


def test_toposort():
    assert list(toposort(A).keys()) == [C, D, E, B, A]


def test_toposort_cycle_detection():
    C = Node(name="C", children=[])
    A = Node(name="A", children=[C])
    B = Node(name="B", children=[A])
    A.children.append(B)

    # A depends on B which depends on A
    with pytest.raises(ValueError, match="cycle detected in the graph"):
        toposort(A)
