from __future__ import annotations

from typing import Any, Optional

import pytest
from typing_extensions import Self

from ibis.common.collections import frozendict
from ibis.common.deferred import _
from ibis.common.graph import Graph, Node
from ibis.common.grounds import Concrete
from ibis.common.patterns import Between, Object


class MyNode(Concrete, Node):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]
    e: Optional[Self] = None
    f: tuple[Self, ...] = ()
    g: Any = None


def generate_node(depth, g=None):
    # generate a nested node object with the given depth
    if depth == 0:
        return MyNode(10, "20", c=(30, 40), d=frozendict(e=50, f=60))
    return MyNode(
        depth,
        "2",
        c=(3, 4),
        d=frozendict(e=5, f=6),
        e=generate_node(0),
        f=(generate_node(depth - 1), generate_node(0)),
        g=g,
    )


@pytest.mark.parametrize("depth", [0, 1, 10])
def test_generate_node(depth):
    n = generate_node(depth)
    assert isinstance(n, MyNode)
    assert len(Graph.from_bfs(n).nodes()) == depth + 1


def test_bfs(benchmark):
    node = generate_node(500)
    benchmark(Graph.from_bfs, node)


def test_dfs(benchmark):
    node = generate_node(500)
    benchmark(Graph.from_dfs, node)


def test_replace_pattern(benchmark):
    node = generate_node(500)
    pattern = Object(MyNode, a=Between(lower=100)) >> _.copy(a=_.a + 1)
    benchmark(node.replace, pattern)


def test_replace_mapping(benchmark):
    node = generate_node(500)
    subs = {generate_node(1): generate_node(0)}
    benchmark(node.replace, subs)


def test_equality_caching(benchmark):
    node = generate_node(150)
    other = generate_node(150)
    assert node == other
    assert other == node
    assert node is not other
    benchmark.pedantic(node.__eq__, args=[other], iterations=100, rounds=200)
