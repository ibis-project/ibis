"""Various traversal utilities for the expression graph."""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Any, Callable, Dict, Sequence

from ibis.util import recursive_get


class Node(Hashable):
    __slots__ = ()

    @property
    @abstractmethod
    def __args__(self) -> Sequence:
        ...

    @property
    @abstractmethod
    def __argnames__(self) -> Sequence:
        ...

    def __children__(self, filter=None):
        return tuple(_flatten_collections(self.__args__, filter or Node))

    def __rich_repr__(self):
        return zip(self.__argnames__, self.__args__)

    def map(self, fn, filter=None):
        results = {}
        for node in Graph.from_bfs(self, filter=filter).toposort():
            kwargs = dict(zip(node.__argnames__, node.__args__))
            kwargs = recursive_get(kwargs, results)
            results[node] = fn(node, results, **kwargs)

        return results

    def find(self, type, filter=None):
        def fn(node, _, **kwargs):
            if isinstance(node, type):
                return node
            return None

        result = self.map(fn, filter=filter)

        return {node for node in result.values() if node is not None}

    def substitute(self, fn, filter=None):
        return self.map(fn, filter=filter)[self]

    def replace(self, subs, filter=None):
        def fn(node, _, **kwargs):
            try:
                return subs[node]
            except KeyError:
                return node.__class__(**kwargs)

        return self.substitute(fn, filter=filter)


def _flatten_collections(node, filter=Node):
    """Flatten collections of nodes into a single iterator.

    We treat common collection types inherently Node (e.g. list, tuple, dict)
    but as undesired in a graph representation, so we traverse them implicitly.

    Parameters
    ----------
    node : Any
        Flattaneble object unless it's an instance of the types passed as filter.
    filter : type, default Node
        Type to filter out for the traversal, e.g. Node.

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


class Graph(Dict[Node, Sequence[Node]]):
    def __init__(self, mapping=(), /, **kwargs):
        if isinstance(mapping, Node):
            mapping = self.from_bfs(mapping)
        super().__init__(mapping, **kwargs)

    @classmethod
    def from_bfs(cls, root: Node, filter=Node) -> Graph:
        if not isinstance(root, Node):
            raise TypeError('node must be an instance of ibis.common.graph.Node')

        queue = deque([root])
        graph = cls()

        while queue:
            if (node := queue.popleft()) not in graph:
                graph[node] = deps = node.__children__(filter)
                queue.extend(deps)

        return graph

    @classmethod
    def from_dfs(cls, root: Node, filter=Node) -> Graph:
        if not isinstance(root, Node):
            raise TypeError('node must be an instance of ibis.common.graph.Node')

        stack = deque([root])
        graph = dict()

        while stack:
            if (node := stack.pop()) not in graph:
                graph[node] = deps = node.__children__(filter)
                stack.extend(deps)

        return cls(reversed(graph.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def nodes(self):
        return self.keys()

    def invert(self) -> Graph:
        result = {node: [] for node in self}
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


def bfs(node: Node) -> Graph:
    return Graph.from_bfs(node)


def dfs(node: Node) -> Graph:
    return Graph.from_dfs(node)


def toposort(node: Node) -> Graph:
    return Graph(node).toposort()


# these could be callables instead
proceed = True
halt = False


def traverse(
    fn: Callable[[Node], tuple[bool | Iterable, Any]], node: Iterable[Node], filter=Node
) -> Iterator[Any]:
    """Utility for generic expression tree traversal.

    Parameters
    ----------
    fn
        A function applied on each expression. The first element of the tuple
        controls the traversal, and the second is the result if its not `None`.
    node
        The Node expression or a list of expressions.
    filter
        Restrict initial traversal to this kind of node
    """
    args = reversed(node) if isinstance(node, Iterable) else [node]
    todo = deque(arg for arg in args if isinstance(arg, filter))
    seen = set()

    while todo:
        node = todo.pop()

        if node in seen:
            continue
        else:
            seen.add(node)

        control, result = fn(node)
        if result is not None:
            yield result

        if control is not halt:
            if control is proceed:
                args = node.__children__(filter)
            elif isinstance(control, Iterable):
                args = control
            else:
                raise TypeError(
                    'First item of the returned tuple must be '
                    'an instance of boolean or iterable'
                )

            todo.extend(reversed(args))
