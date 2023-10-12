from __future__ import annotations

from typing import Optional

import pytest
from typing_extensions import Self  # noqa: TCH002

from ibis.common.collections import frozendict
from ibis.common.graph import Graph, Node
from ibis.common.grounds import Concrete


class MyNode(Concrete, Node):
    a: int
    b: str
    c: tuple[int, ...]
    d: frozendict[str, int]
    e: Optional[Self] = None
    f: tuple[Self, ...] = ()


def generate_node(depth):
    # generate a nested node object with the given depth
    if depth == 0:
        return MyNode(10, "20", c=(30, 40), d=frozendict(e=50, f=60))
    return MyNode(
        1,
        "2",
        c=(3, 4),
        d=frozendict(e=5, f=6),
        e=generate_node(0),
        f=(generate_node(depth - 1), generate_node(0)),
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
