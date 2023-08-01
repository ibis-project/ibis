"""Various traversal utilities for the expression graph."""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

from ibis.common.patterns import NoMatch, pattern
from ibis.util import experimental

if TYPE_CHECKING:
    from typing_extensions import Self


def _flatten_collections(node: Node, filter: type) -> Iterator[Node]:
    """Flatten collections of nodes into a single iterator.

    We treat common collection types inherently traversable (e.g. list, tuple, dict)
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


def _recursive_get(obj: Any, dct: dict[Node, Any]) -> Any:
    """Recursively replace objects in a nested structure with values from a dict.

    Since we treat collection types inherently traversable (e.g. list, tuple, dict) we
    need to traverse them implicitly and replace the values given a result mapping.

    Parameters
    ----------
    obj : Any
        Object to replace.
    dct : dict[Node, Any]
        Mapping of objects to replace with their values.

    Returns
    -------
    Object with replaced values.
    """
    if isinstance(obj, tuple):
        return tuple(_recursive_get(o, dct) for o in obj)
    elif isinstance(obj, dict):
        return {k: _recursive_get(v, dct) for k, v in obj.items()}
    else:
        return dct.get(obj, obj)


class Node(Hashable):
    __slots__ = ()

    @property
    @abstractmethod
    def __args__(self) -> tuple[Any, ...]:
        """Sequence of arguments to traverse."""

    @property
    @abstractmethod
    def __argnames__(self) -> tuple[str, ...]:
        """Sequence of argument names."""

    def __children__(self, filter: Optional[type] = None) -> tuple[Node, ...]:
        """Return the children of this node.

        This method is used to traverse the Node so it returns the children of the node
        in the order they should be traversed. We treat common collection types
        inherently traversable (e.g. list, tuple, dict), so this method flattens and
        optionally filters the arguments of the node.

        Parameters
        ----------
        filter : type, default Node
            Type to filter out for the traversal, Node is used by default.

        Returns
        -------
        Child nodes of this node.
        """
        return tuple(_flatten_collections(self.__args__, filter or Node))

    def __rich_repr__(self):
        """Support for rich reprerentation of the node."""
        return zip(self.__argnames__, self.__args__)

    @experimental
    def branches(self) -> Iterator[Sequence[Node]]:
        """Yield all branches of the graph.

        A branch is a path from the root to a leaf node. This method is primarily
        used to implement the `path` method supporting `XPath`-like queries.

        Yields
        ------
        A sequence of nodes representing a branch.
        """
        stack = [(self, [])]

        while stack:
            node, path = stack.pop()

            if children := node.__children__():
                for child in reversed(children):
                    stack.append((child, path + [node]))
            else:
                yield path + [node]

    @experimental
    def path(
        self, *pats: Any, context: Optional[dict] = None
    ) -> Iterator[Sequence[Node]]:
        """Return the first tree branch matching a given sequence pattern.

        This method provides a way to query the graph using `XPath`-like expressions.
        The following XPath expression "//Alias//Value[dtype==int64]" would roughly
        translate to the following Python code:

            node.path(..., Alias, ..., Object(Value, dtype=dt.Int64), ...)

        Parameters
        ----------
        pats
            Sequence which is coerced to a sequence pattern. See `ibis.common.patterns`
            for more details.
        context
            Optional context to use for the pattern matching.
        """
        pat = pattern(list(pats))
        for branch in self.branches():
            result = pat.match(branch, context)
            if result is not NoMatch:
                return result
        return NoMatch

    def map(self, fn: Callable, filter: Optional[type] = None) -> dict[Node, Any]:
        """Apply a function to all nodes in the graph.

        The traversal is done in a topological order, so the function receives the
        results of its immediate children as keyword arguments.

        Parameters
        ----------
        fn : Callable
            Function to apply to each node. It receives the node as the first argument,
            the results as the second and the results of the children as keyword
            arguments.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.

        Returns
        -------
        A mapping of nodes to their results.
        """
        results = {}
        for node in Graph.from_bfs(self, filter=filter).toposort():
            kwargs = dict(zip(node.__argnames__, node.__args__))
            kwargs = _recursive_get(kwargs, results)
            results[node] = fn(node, results, **kwargs)
        return results

    def find(
        self, type: type | tuple[type], filter: Optional[type] = None
    ) -> set[Node]:
        """Find all nodes of a given type in the graph.

        Parameters
        ----------
        type : type | tuple[type]
            Type or tuple of types to find.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.

        Returns
        -------
        The set of nodes matching the given type.
        """
        nodes = Graph.from_bfs(self, filter=filter).nodes()
        return {node for node in nodes if isinstance(node, type)}

    @experimental
    def match(
        self, pat: Any, filter: Optional[type] = None, context: Optional[dict] = None
    ) -> set[Node]:
        """Find all nodes matching a given pattern in the graph.

        A more advanced version of find, this method allows to match nodes based on
        the more flexible pattern matching system implemented in the pattern module.

        Parameters
        ----------
        pat : Any
            Pattern to match. `ibis.common.pattern()` function is used to coerce the
            input value into a pattern. See the pattern module for more details.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.
        context : Optional[dict], default None
            Optional context to use for the pattern matching.

        Returns
        -------
        The set of nodes matching the given pattern.
        """
        pat = pattern(pat)
        ctx = context or {}
        nodes = Graph.from_bfs(self, filter=filter).nodes()
        return {node for node in nodes if pat.is_match(node, ctx)}

    @experimental
    def replace(
        self, pat: Any, filter: Optional[type] = None, context: Optional[dict] = None
    ) -> Any:
        """Match and replace nodes in the graph according to a given pattern.

        The pattern matching system is used to match nodes in the graph and replace them
        with the results of the pattern.

        Parameters
        ----------
        pat : Any
            Pattern to match. `ibis.common.pattern()` function is used to coerce the
            input value into a pattern. See the pattern module for more details.
            Actual replacement is done by the `ibis.common.pattern.Replace` pattern.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.
        context : Optional[dict], default None
            Optional context to use for the pattern matching.

        Returns
        -------
        The root node of the graph with the replaced nodes.
        """
        pat = pattern(pat)
        ctx = context or {}

        def fn(node, _, **kwargs):
            # TODO(kszucs): pass the reconstructed node from the results provided by the
            # kwargs to the pattern rather than the original one node object, this way
            # we can match on already replaced nodes
            if (result := pat.match(node, ctx)) is NoMatch:
                return node.__class__(**kwargs)
            else:
                return result

        return self.map(fn, filter=filter)[self]


class Graph(Dict[Node, Sequence[Node]]):
    """A mapping-like graph data structure for easier graph traversal and manipulation.

    The data structure is a mapping of nodes to their children. The children are
    represented as a sequence of nodes. The graph can be constructed from a root node
    using the `from_bfs` or `from_dfs` class methods.

    Parameters
    ----------
    mapping : Node or Mapping[Node, Sequence[Node]], default ()
        Either a root node or a mapping of nodes to their children.
    """

    def __init__(self, mapping=(), /, **kwargs):
        if isinstance(mapping, Node):
            mapping = self.from_bfs(mapping)
        super().__init__(mapping, **kwargs)

    @classmethod
    def from_bfs(cls, root: Node, filter=Node) -> Self:
        """Construct a graph from a root node using a breadth-first search.

        The traversal is implemented in an iterative fashion using a queue.

        Parameters
        ----------
        root : Node
            Root node of the graph.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.

        Returns
        -------
        A graph constructed from the root node.
        """
        if not isinstance(root, Node):
            raise TypeError("node must be an instance of ibis.common.graph.Node")

        queue = deque([root])
        graph = cls()

        while queue:
            if (node := queue.popleft()) not in graph:
                graph[node] = deps = node.__children__(filter)
                queue.extend(deps)

        return graph

    @classmethod
    def from_dfs(cls, root: Node, filter=Node) -> Self:
        """Construct a graph from a root node using a depth-first search.

        The traversal is implemented in an iterative fashion using a stack.

        Parameters
        ----------
        root : Node
            Root node of the graph.
        filter : Optional[type], default None
            Type to filter out for the traversal, Node is filtered out by default.

        Returns
        -------
        A graph constructed from the root node.
        """
        if not isinstance(root, Node):
            raise TypeError("node must be an instance of ibis.common.graph.Node")

        stack = deque([root])
        graph = dict()

        while stack:
            if (node := stack.pop()) not in graph:
                graph[node] = deps = node.__children__(filter)
                stack.extend(deps)

        return cls(reversed(graph.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def nodes(self) -> set[Node]:
        """Return all unique nodes in the graph."""
        return self.keys()

    def invert(self) -> Self:
        """Invert the data structure.

        The graph originally maps nodes to their children, this method inverts the
        mapping to map nodes to their parents.

        Returns
        -------
        The inverted graph.
        """
        result = {node: [] for node in self}
        for node, dependencies in self.items():
            for dependency in dependencies:
                result[dependency].append(node)
        return self.__class__({k: tuple(v) for k, v in result.items()})

    def toposort(self) -> Self:
        """Topologically sort the graph using Kahn's algorithm.

        The graph is sorted in a way that all the dependencies of a node are placed
        before the node itself. The graph must not contain any cycles. Especially useful
        for mutating the graph in a way that the dependencies of a node are mutated
        before the node itself.

        Returns
        -------
        The topologically sorted graph.
        """
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
    """Construct a graph from a root node using a breadth-first search.

    Parameters
    ----------
    node : Node
        Root node of the graph.

    Returns
    -------
    A graph constructed from the root node.
    """
    return Graph.from_bfs(node)


def dfs(node: Node) -> Graph:
    """Construct a graph from a root node using a depth-first search.

    Parameters
    ----------
    node : Node
        Root node of the graph.

    Returns
    -------
    A graph constructed from the root node.
    """
    return Graph.from_dfs(node)


def toposort(node: Node) -> Graph:
    """Construct a graph from a root node then topologically sort it.

    Parameters
    ----------
    node : Node
        Root node of the graph.

    Returns
    -------
    A topologically sorted graph constructed from the root node.
    """
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
                    "First item of the returned tuple must be "
                    "an instance of boolean or iterable"
                )

            todo.extend(reversed(args))
