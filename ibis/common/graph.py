"""Various traversal utilities for the expression graph."""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator, KeysView, Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from ibis.common.bases import Hashable
from ibis.common.collections import frozendict
from ibis.common.patterns import NoMatch, Pattern, pattern
from ibis.util import experimental

if TYPE_CHECKING:
    from typing_extensions import Self

    N = TypeVar("N")


def _flatten_collections(node: Any) -> Iterator[N]:
    """Flatten collections of nodes into a single iterator.

    We treat common collection types inherently traversable (e.g. list, tuple, dict)
    but as undesired in a graph representation, so we traverse them implicitly.

    Parameters
    ----------
    node
        Flattaneble object.

    Returns
    -------
    A flat generator of the filtered nodes.

    Examples
    --------
    >>> from ibis.common.grounds import Concrete
    >>> from ibis.common.graph import Node
    >>>
    >>> class MyNode(Concrete, Node):
    ...     number: int
    ...     string: str
    ...     children: tuple[Node, ...]
    ...
    >>> a = MyNode(4, "a", ())
    >>>
    >>> b = MyNode(3, "b", ())
    >>> c = MyNode(2, "c", (a, b))
    >>> d = MyNode(1, "d", (c,))
    >>>
    >>> assert list(_flatten_collections(a)) == [a]
    >>> assert list(_flatten_collections((c,))) == [c]
    >>> assert list(_flatten_collections([a, b, (c, a)])) == [a, b, c, a]
    """
    if isinstance(node, Node):
        yield node
    elif isinstance(node, (tuple, list)):
        for item in node:
            yield from _flatten_collections(item)
    elif isinstance(node, (dict, frozendict)):
        for value in node.values():
            yield from _flatten_collections(value)


def _recursive_lookup(obj: Any, dct: dict) -> Any:
    """Recursively replace objects in a nested structure with values from a dict.

    Since we treat common collection types inherently traversable, so we need to
    traverse them implicitly and replace the values given a result mapping.

    Parameters
    ----------
    obj
        Object to replace.
    dct
        Mapping of objects to replace with their values.

    Returns
    -------
    Object with replaced values.

    Examples
    --------
    >>> from ibis.common.grounds import Concrete
    >>> from ibis.common.graph import Node
    >>>
    >>> class MyNode(Concrete, Node):
    ...     number: int
    ...     string: str
    ...     children: tuple[Node, ...]
    ...
    >>> a = MyNode(4, "a", ())
    >>>
    >>> b = MyNode(3, "b", ())
    >>> c = MyNode(2, "c", (a, b))
    >>> d = MyNode(1, "d", (c,))
    >>>
    >>> dct = {a: "A", b: "B"}
    >>> _recursive_lookup(a, dct)
    'A'
    >>> _recursive_lookup((a, b), dct)
    ('A', 'B')
    >>> _recursive_lookup({1: a, 2: b}, dct)
    {1: 'A', 2: 'B'}
    >>> _recursive_lookup((a, frozendict({1: c})), dct)
    ('A', {1: MyNode(number=2, ...)})
    """
    if isinstance(obj, Node):
        return dct.get(obj, obj)
    elif isinstance(obj, (tuple, list)):
        return tuple(_recursive_lookup(o, dct) for o in obj)
    elif isinstance(obj, (dict, frozendict)):
        return {k: _recursive_lookup(v, dct) for k, v in obj.items()}
    else:
        return obj


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

    def __rich_repr__(self):
        """Support for rich reprerentation of the node."""
        return zip(self.__argnames__, self.__args__)

    def map(self, fn: Callable, filter: Optional[Any] = None) -> dict[Node, Any]:
        """Apply a function to all nodes in the graph.

        The traversal is done in a topological order, so the function receives the
        results of its immediate children as keyword arguments.

        Parameters
        ----------
        fn
            Function to apply to each node. It receives the node as the first argument,
            the results as the second and the results of the children as keyword
            arguments.
        filter
            Pattern-like object to filter out nodes from the traversal. The traversal
            will only visit nodes that match the given pattern and stop otherwise.

        Returns
        -------
        A mapping of nodes to their results.
        """
        results: dict[Node, Any] = {}
        for node in Graph.from_bfs(self, filter=filter).toposort():
            # minor optimization to directly recurse into the children
            kwargs = {
                k: _recursive_lookup(v, results)
                for k, v in zip(node.__argnames__, node.__args__)
            }
            results[node] = fn(node, results, **kwargs)
        return results

    def find(self, type: type | tuple[type], filter: Optional[Any] = None) -> set[Node]:
        """Find all nodes of a given type in the graph.

        Parameters
        ----------
        type
            Type or tuple of types to find.
        filter
            Pattern-like object to filter out nodes from the traversal. The traversal
            will only visit nodes that match the given pattern and stop otherwise.

        Returns
        -------
        The set of nodes matching the given type.
        """
        nodes = Graph.from_bfs(self, filter=filter).nodes()
        return {node for node in nodes if isinstance(node, type)}

    @experimental
    def match(
        self, pat: Any, filter: Optional[Any] = None, context: Optional[dict] = None
    ) -> set[Node]:
        """Find all nodes matching a given pattern in the graph.

        A more advanced version of find, this method allows to match nodes based on
        the more flexible pattern matching system implemented in the pattern module.

        Parameters
        ----------
        pat
            Pattern to match. `ibis.common.pattern()` function is used to coerce the
            input value into a pattern. See the pattern module for more details.
        filter
            Pattern-like object to filter out nodes from the traversal. The traversal
            will only visit nodes that match the given pattern and stop otherwise.
        context
            Optional context to use for the pattern matching.

        Returns
        -------
        The set of nodes matching the given pattern.
        """
        pat = pattern(pat)
        ctx = context or {}
        nodes = Graph.from_bfs(self, filter=filter).nodes()
        return {node for node in nodes if pat.match(node, ctx) is not NoMatch}

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
            # need to first reconstruct the node from the possible rewritten
            # children, so we can match on the new node containing the rewritten
            # child arguments, this way we can propagate the rewritten nodes
            # upward in the hierarchy
            # TODO(kszucs): add a __recreate__() method to the Node interface
            # with a default implementation that uses the __class__ constructor
            # which is supposed to provide an implementation for quick object
            # reconstruction (the __recreate__ implementation in grounds.py
            # should be sped up as well by totally avoiding the validation)
            recreated = node.__class__(**kwargs)
            if (result := pat.match(recreated, ctx)) is NoMatch:
                return recreated
            else:
                return result

        results = self.map(fn, filter=filter)
        return results.get(self, self)


class Graph(dict[Node, Sequence[Node]]):
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
    def from_bfs(cls, root: Node, filter: Optional[Any] = None) -> Self:
        """Construct a graph from a root node using a breadth-first search.

        The traversal is implemented in an iterative fashion using a queue.

        Parameters
        ----------
        root
            Root node of the graph.
        filter
            Pattern-like object to filter out nodes from the traversal. The traversal
            will only visit nodes that match the given pattern and stop otherwise.

        Returns
        -------
        A graph constructed from the root node.
        """
        if not isinstance(root, Node):
            raise TypeError("node must be an instance of ibis.common.graph.Node")

        queue = deque()
        graph = cls()

        if filter is None:
            # fast path for the default no filter case, according to benchmarks
            # this is gives a 10% speedup compared to the filtered version
            queue.append(root)
            while queue:
                if (node := queue.popleft()) not in graph:
                    children = tuple(_flatten_collections(node.__args__))
                    graph[node] = children
                    queue.extend(children)
        else:
            filter = pattern(filter)
            if filter.match(root, {}) is not NoMatch:
                queue.append(root)
            while queue:
                if (node := queue.popleft()) not in graph:
                    children = tuple(
                        child
                        for child in _flatten_collections(node.__args__)
                        if filter.match(child, {}) is not NoMatch
                    )
                    graph[node] = children
                    queue.extend(children)

        return graph

    @classmethod
    def from_dfs(cls, root: Node, filter: Optional[Any] = None) -> Self:
        """Construct a graph from a root node using a depth-first search.

        The traversal is implemented in an iterative fashion using a stack.

        Parameters
        ----------
        root
            Root node of the graph.
        filter
            Pattern-like object to filter out nodes from the traversal. The traversal
            will only visit nodes that match the given pattern and stop otherwise.

        Returns
        -------
        A graph constructed from the root node.
        """
        if not isinstance(root, Node):
            raise TypeError("node must be an instance of ibis.common.graph.Node")

        stack = deque()
        graph = dict()

        if filter is None:
            # fast path for the default no filter case, according to benchmarks
            # this is gives a 10% speedup compared to the filtered version
            stack.append(root)
            while stack:
                if (node := stack.pop()) not in graph:
                    children = tuple(_flatten_collections(node.__args__))
                    graph[node] = children
                    stack.extend(children)
        else:
            filter = pattern(filter)
            if filter.match(root, {}) is not NoMatch:
                stack.append(root)
            while stack:
                if (node := stack.pop()) not in graph:
                    children = tuple(
                        child
                        for child in _flatten_collections(node.__args__)
                        if filter.match(child, {}) is not NoMatch
                    )
                    graph[node] = children
                    stack.extend(children)

        return cls(reversed(graph.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def nodes(self) -> KeysView[Node]:
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
        result: dict[Node, list[Node]] = {node: [] for node in self}
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


def bfs(node: Node, filter: Optional[Any] = None) -> Graph:
    """Construct a graph from a root node using a breadth-first search.

    Parameters
    ----------
    node
        Root node of the graph.
    filter
        Pattern-like object to filter out nodes from the traversal. The traversal
        will only visit nodes that match the given pattern and stop otherwise.

    Returns
    -------
    A graph constructed from the root node.
    """
    return Graph.from_bfs(node, filter=filter)


def dfs(node: Node, filter: Optional[Any] = None) -> Graph:
    """Construct a graph from a root node using a depth-first search.

    Parameters
    ----------
    node
        Root node of the graph.
    filter
        Pattern-like object to filter out nodes from the traversal. The traversal
        will only visit nodes that match the given pattern and stop otherwise.

    Returns
    -------
    A graph constructed from the root node.
    """
    return Graph.from_dfs(node, filter=filter)


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
    fn: Callable[[Node], tuple[bool | Iterable, Any]],
    node: Iterable[Node] | Node,
    filter: Optional[Any] = None,
) -> Iterator[Any]:
    """Utility for generic expression tree traversal.

    Parameters
    ----------
    fn
        A function applied on each expression. The first element of the tuple controls
        the traversal, and the second is the result if its not `None`.
    node
        The Node expression or a list of expressions.
    filter
        Pattern-like object to filter out nodes from the traversal. The traversal will
        only visit nodes that match the given pattern and stop otherwise.
    """

    args = reversed(node) if isinstance(node, Sequence) else [node]
    todo: deque[Node] = deque(args)
    seen: set[Node] = set()
    filter: Pattern = pattern(filter or ...)

    while todo:
        node = todo.pop()

        if node in seen:
            continue
        if filter.match(node, {}) is NoMatch:
            continue

        seen.add(node)

        control, result = fn(node)
        if result is not None:
            yield result

        if control is not halt:
            if control is proceed:
                children = tuple(_flatten_collections(node.__args__))
            elif isinstance(control, Iterable):
                children = control
            else:
                raise TypeError(
                    "First item of the returned tuple must be "
                    "an instance of boolean or iterable"
                )

            todo.extend(reversed(children))
