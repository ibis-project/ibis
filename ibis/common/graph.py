"""Various traversal utilities for the expression graph."""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Any, Callable, Dict, Sequence


def _flatten_collections(node, filter):
    """Flatten collections of nodes into a single iterator.

    We treat common collection types inherently traversable (e.g. list, tuple, dict)
    but as undesired in a graph representation, so we traverse them implicitly.

    Parameters
    ----------
    node : Any
        Flattaneble object unless it's an instance of the types passed as filter.
    filter : type
        Type to filter out for the traversal, e.g. Traversable.

    Returns
    -------
    Iterator : Any
    """
    if isinstance(node, filter):
        yield node
    elif isinstance(node, (str, bytes)):
        pass
    elif isinstance(node, Sequence):
        for item in node:
            yield from _flatten_collections(item, filter)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _flatten_collections(key, filter)
            yield from _flatten_collections(value, filter)


class Traversable(Hashable):

    __slots__ = ()

    @property
    @abstractmethod
    def __children__(self) -> Sequence:
        ...


def children(node, filter=Traversable):
    # TODO(kszucs): perhaps this should be a set instead of a tuple
    return tuple(_flatten_collections(node.__children__, filter))


class Graph(Dict[Traversable, Sequence[Traversable]]):
    def __init__(self, mapping=(), /, **kwargs):
        if isinstance(mapping, Traversable):
            mapping = self.from_bfs(mapping)
        super().__init__(mapping, **kwargs)

    @classmethod
    def from_bfs(cls, root: Traversable, filter=Traversable) -> Graph:
        if not isinstance(root, Traversable):
            raise TypeError('node must be an instance of Traversable')

        queue = deque([root])
        graph = cls()

        while queue:
            if (node := queue.popleft()) not in graph:
                dependencies = children(node, filter)
                graph[node] = dependencies
                queue.extend(dependencies)

        return graph

    @classmethod
    def from_dfs(cls, root: Traversable, filter=Traversable) -> Graph:
        if not isinstance(root, Traversable):
            raise TypeError('node must be an instance of Traversable')

        stack = deque([root])
        graph = dict()

        while stack:
            if (node := stack.pop()) not in graph:
                dependencies = children(node, filter)
                graph[node] = dependencies
                stack.extend(dependencies)

        return cls(reversed(graph.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def nodes(self):
        return self.keys()

    def invert(self) -> Graph:
        result = {node: [] for node in self.keys()}
        for node, dependencies in self.items():
            for dependency in dependencies:
                result[dependency].append(node)
        return self.__class__({k: tuple(v) for k, v in result.items()})

    def toposort(self) -> Graph:
        dependents = self.invert()
        in_degree = {k: len(v) for k, v in self.items()}

        queue = deque(node for node, count in in_degree.items() if not count)
        result = self.__class__()

        while queue:
            node = queue.popleft()
            result[node] = self[node]

            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if not in_degree[dependent]:
                    queue.append(dependent)

        if any(in_degree.values()):
            raise ValueError("cycle detected in the graph")

        return result


def bfs(node: Traversable) -> Graph:
    return Graph.from_bfs(node)


def dfs(node: Traversable) -> Graph:
    return Graph.from_dfs(node)


def toposort(node: Traversable) -> Graph:
    return Graph(node).toposort()


# these could be callables instead
proceed = True
halt = False


def traverse(
    fn: Callable[[Traversable], tuple[bool | Iterable, Any]],
    node: Iterable[Traversable],
    dedup: bool = True,
    filter=Traversable,
) -> Iterator[Any]:
    """Utility for generic expression tree traversal.

    Parameters
    ----------
    fn
        A function applied on each expression. The first element of the tuple
        controls the traversal, and the second is the result if its not `None`.
    expr
        The traversable expression or a list of expressions.
    dedup
        Whether to allow expression traversal more than once
    """
    args = reversed(node) if isinstance(node, Iterable) else [node]
    todo = deque(arg for arg in args if isinstance(arg, filter))
    seen = set()

    while todo:
        node = todo.pop()

        if dedup:
            if node in seen:
                continue
            else:
                seen.add(node)

        control, result = fn(node)
        if result is not None:
            yield result

        if control is not halt:
            if control is proceed:
                args = children(node, filter)
            elif isinstance(control, Iterable):
                args = control
            else:
                raise TypeError(
                    'First item of the returned tuple must be '
                    'an instance of boolean or iterable'
                )

            todo.extend(reversed(args))
