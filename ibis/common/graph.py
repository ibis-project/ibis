"""Various traversal utilities for the expression graph."""

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator, KeysView, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

from ibis.common.bases import Hashable
from ibis.common.collections import frozendict
from ibis.common.patterns import NoMatch, Pattern
from ibis.common.typing import _ClassInfo
from ibis.util import experimental

if TYPE_CHECKING:
    from typing_extensions import Self

    N = TypeVar("N")


Finder = Callable[["Node"], bool]
FinderLike = Union[Finder, Pattern, _ClassInfo]

Replacer = Callable[["Node", dict["Node", Any]], "Node"]
ReplacerLike = Union[Replacer, Pattern, Mapping]


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


def _coerce_finder(obj: FinderLike, context: Optional[dict] = None) -> Finder:
    """Coerce an object into a callable finder function.

    Parameters
    ----------
    obj
        A callable accepting the node, a pattern or a type to match on.
    context
        Optional context to use if the finder is a pattern.

    Returns
    -------
    A callable finder function which can be used to match nodes.

    """
    if isinstance(obj, Pattern):
        ctx = context or {}

        def fn(node):
            return obj.match(node, ctx) is not NoMatch
    elif isinstance(obj, (tuple, type)):

        def fn(node):
            return isinstance(node, obj)
    elif callable(obj):
        fn = obj
    else:
        raise TypeError("finder must be callable, type, tuple of types or a pattern")

    return fn


def _coerce_replacer(obj: ReplacerLike, context: Optional[dict] = None) -> Replacer:
    """Coerce an object into a callable replacer function.

    Parameters
    ----------
    obj
        A Pattern, a Mapping or a callable which can be fed to `node.map()`
        to replace nodes.
    context
        Optional context to use if the replacer is a pattern.

    Returns
    -------
    A callable replacer function which can be used to replace nodes.

    """
    if isinstance(obj, Pattern):
        ctx = context or {}

        def fn(node, _, **kwargs):
            # need to first reconstruct the node from the possible rewritten
            # children, so we can match on the new node containing the rewritten
            # child arguments, this way we can propagate the rewritten nodes
            # upward in the hierarchy, using a specialized __recreate__ method
            # improves the performance by 17% compared node.__class__(**kwargs)
            recreated = node.__recreate__(kwargs)
            if (result := obj.match(recreated, ctx)) is NoMatch:
                return recreated
            else:
                return result

    elif isinstance(obj, Mapping):

        def fn(node, _, **kwargs):
            try:
                return obj[node]
            except KeyError:
                return node.__class__(**kwargs)
    elif callable(obj):
        fn = obj
    else:
        raise TypeError("replacer must be callable, mapping or a pattern")

    return fn


class Node(Hashable):
    __slots__ = ()

    @classmethod
    def __recreate__(cls, kwargs: Any) -> Self:
        """Reconstruct the node from the given arguments."""
        return cls(**kwargs)

    @property
    @abstractmethod
    def __args__(self) -> tuple[Any, ...]:
        """Sequence of arguments to traverse."""

    @property
    @abstractmethod
    def __argnames__(self) -> tuple[str, ...]:
        """Sequence of argument names."""

    @property
    def __children__(self) -> tuple[Node, ...]:
        """Sequence of children nodes."""
        return tuple(_flatten_collections(self.__args__))

    def __rich_repr__(self):
        """Support for rich reprerentation of the node."""
        return zip(self.__argnames__, self.__args__)

    def map(self, fn: Callable, filter: Optional[Finder] = None) -> dict[Node, Any]:
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

        graph, _ = Graph.from_bfs(self, filter=filter).toposort()
        for node in graph:
            # minor optimization to directly recurse into the children
            kwargs = {
                k: _recursive_lookup(v, results)
                for k, v in zip(node.__argnames__, node.__args__)
            }
            results[node] = fn(node, results, **kwargs)

        return results

    @experimental
    def map_clear(
        self, fn: Callable, filter: Optional[Finder] = None
    ) -> dict[Node, Any]:
        """Apply a function to all nodes in the graph more memory efficiently.

        Alternative implementation of `map` to reduce memory usage. While `map` keeps
        all the results in memory until the end of the traversal, this method removes
        intermediate results as soon as they are not needed anymore.

        Prefer this method over `map` if the results consume significant amount of
        memory and if the intermediate results are not needed.

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
        In contrast to `map`, this method returns the result of the root node only since
        the rest of the results are already discarded.

        """
        results: dict[Node, Any] = {}

        graph, dependents = Graph.from_bfs(self, filter=filter).toposort()
        dependents = {k: set(v) for k, v in dependents.items()}

        for node, dependencies in graph.items():
            kwargs = {
                k: _recursive_lookup(v, results)
                for k, v in zip(node.__argnames__, node.__args__)
            }
            results[node] = fn(node, results, **kwargs)

            # remove the results belonging to the dependencies if they are not
            # needed by other nodes during the rest of the traversal
            for dependency in set(dependencies):
                dependents[dependency].remove(node)
                if not dependents[dependency]:
                    del results[dependency]

        return results[self]

    # TODO(kszucs): perhaps rename it to find_all() for better clarity
    def find(
        self,
        finder: FinderLike,
        filter: Optional[FinderLike] = None,
        context: Optional[dict] = None,
    ) -> list[Node]:
        """Find all nodes matching a given pattern or type in the graph.

        Allow to match nodes based on the flexible pattern matching system implemented
        in the pattern module, but also provide a fast path for matching based on the
        type of the node.

        Parameters
        ----------
        finder
            A type, tuple of types, a pattern or a callable to match upon.
        filter
            A type, tuple of types, a pattern or a callable to filter out nodes
            from the traversal. The traversal will only visit nodes that match
            the given filter and stop otherwise.
        context
            Optional context to use if `finder` or `filter` is a pattern.

        Returns
        -------
        The list of nodes matching the given pattern. The order of the nodes is
        determined by a breadth-first search.

        """
        nodes = Graph.from_bfs(self, filter=filter, context=context).nodes()
        finder = _coerce_finder(finder, context)
        return [node for node in nodes if finder(node)]

    @experimental
    def find_topmost(
        self, finder: FinderLike, context: Optional[dict] = None
    ) -> list[Node]:
        """Find all topmost nodes matching a given pattern in the graph.

        A more advanced version of find, this method stops the traversal at the first
        node that matches the given pattern and does not descend into its children.

        Parameters
        ----------
        finder
            A type, tuple of types, a pattern or a callable to match upon.
        context
            Optional context to use if `finder` is a pattern.

        Returns
        -------
        The list of topmost nodes matching the given pattern.

        """
        seen = set()
        queue = deque([self])
        result = []
        finder = _coerce_finder(finder, context)

        while queue:
            if (node := queue.popleft()) not in seen:
                if finder(node):
                    result.append(node)
                else:
                    queue.extend(node.__children__)
                seen.add(node)
        return result

    @experimental
    def replace(
        self,
        replacer: ReplacerLike,
        filter: Optional[FinderLike] = None,
        context: Optional[dict] = None,
    ) -> Any:
        """Match and replace nodes in the graph according to a given pattern.

        The pattern matching system is used to match nodes in the graph and replace them
        with the results of the pattern.

        Parameters
        ----------
        replacer
            A `Pattern`, a `Mapping` or a callable which can be fed to
            `node.map()` directly to replace nodes.
        filter
            A type, tuple of types, a pattern or a callable to filter out nodes
            from the traversal. The traversal will only visit nodes that match
            the given filter and stop otherwise.
        context
            Optional context to use for the pattern matching.

        Returns
        -------
        The root node of the graph with the replaced nodes.

        """
        replacer = _coerce_replacer(replacer, context)
        results = self.map(replacer, filter=filter)
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
    def from_bfs(
        cls,
        root: Node,
        filter: Optional[FinderLike] = None,
        context: Optional[dict] = None,
    ) -> Self:
        """Construct a graph from a root node using a breadth-first search.

        The traversal is implemented in an iterative fashion using a queue.

        Parameters
        ----------
        root
            Root node of the graph.
        filter
            A type, tuple of types, a pattern or a callable to filter out nodes
            from the traversal. The traversal will only visit nodes that match
            the given filter and stop otherwise.
        context
            Optional context to use for the pattern matching.

        Returns
        -------
        A graph constructed from the root node.

        """
        if filter is None:
            return bfs(root)
        else:
            filter = _coerce_finder(filter, context)
            return bfs_while(root, filter=filter)

    @classmethod
    def from_dfs(
        cls,
        root: Node,
        filter: Optional[FinderLike] = None,
        context: Optional[dict] = None,
    ) -> Self:
        """Construct a graph from a root node using a depth-first search.

        The traversal is implemented in an iterative fashion using a stack.

        Parameters
        ----------
        root
            Root node of the graph.
        filter
            A type, tuple of types, a pattern or a callable to filter out nodes
            from the traversal. The traversal will only visit nodes that match
            the given filter and stop otherwise.
        context
            Optional context to use for the pattern matching.

        Returns
        -------
        A graph constructed from the root node.

        """
        if filter is None:
            return dfs(root)
        else:
            filter = _coerce_finder(filter, None)
            return dfs_while(root, filter=filter)

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

        return result, dependents


# these could be callables instead
proceed = True
halt = False


def traverse(
    fn: Callable[[Node], tuple[bool | Iterable, Any]], node: Iterable[Node] | Node
) -> Iterator[Any]:
    """Utility for generic expression tree traversal.

    Parameters
    ----------
    fn
        A function applied on each expression. The first element of the tuple controls
        the traversal, and the second is the result if its not `None`.
    node
        The Node expression or a list of expressions.

    """

    args = reversed(node) if isinstance(node, Sequence) else [node]
    todo: deque[Node] = deque(args)
    seen: set[Node] = set()

    while todo:
        node = todo.pop()

        if node in seen:
            continue
        seen.add(node)

        control, result = fn(node)
        if result is not None:
            yield result

        if control is not halt:
            if control is proceed:
                children = node.__children__
            elif isinstance(control, Iterable):
                children = control
            else:
                raise TypeError(
                    "First item of the returned tuple must be "
                    "an instance of boolean or iterable"
                )

            todo.extend(reversed(children))


def bfs(root: Node) -> Graph:
    """Construct a graph from a root node using a breadth-first search.

    Parameters
    ----------
    root
        Root node of the graph.

    Returns
    -------
    A graph constructed from the root node.

    """
    # fast path for the default no filter case, according to benchmarks
    # this is gives a 10% speedup compared to the filtered version
    if not isinstance(root, Node):
        raise TypeError("node must be an instance of ibis.common.graph.Node")

    queue = deque([root])
    graph = Graph()

    while queue:
        if (node := queue.popleft()) not in graph:
            children = node.__children__
            graph[node] = children
            queue.extend(children)

    return graph


def bfs_while(root: Node, filter: Finder) -> Graph:
    """Construct a graph from a root node using a breadth-first search.

    Parameters
    ----------
    root
        Root node of the graph.
    filter
        A callable which returns a boolean given a node. The traversal will only
        visit nodes that match the given filter and stop otherwise.

    Returns
    -------
    A graph constructed from the root node.

    """
    if not isinstance(root, Node):
        raise TypeError("node must be an instance of ibis.common.graph.Node")

    queue = deque()
    graph = Graph()

    if filter(root):
        queue.append(root)

    while queue:
        if (node := queue.popleft()) not in graph:
            children = tuple(child for child in node.__children__ if filter(child))
            graph[node] = children
            queue.extend(children)

    return graph


def dfs(root: Node) -> Graph:
    """Construct a graph from a root node using a depth-first search.

    Parameters
    ----------
    root
        Root node of the graph.

    Returns
    -------
    A graph constructed from the root node.

    """
    # fast path for the default no filter case, according to benchmarks
    # this is gives a 10% speedup compared to the filtered version
    if not isinstance(root, Node):
        raise TypeError("node must be an instance of ibis.common.graph.Node")

    stack = deque([root])
    graph = {}

    while stack:
        if (node := stack.pop()) not in graph:
            children = node.__children__
            graph[node] = children
            stack.extend(children)

    return Graph(reversed(graph.items()))


def dfs_while(root: Node, filter: Finder) -> Graph:
    """Construct a graph from a root node using a depth-first search.

    Parameters
    ----------
    root
        Root node of the graph.
    filter
        A callable which returns a boolean given a node. The traversal will only
        visit nodes that match the given filter and stop otherwise.

    Returns
    -------
    A graph constructed from the root node.

    """
    if not isinstance(root, Node):
        raise TypeError("node must be an instance of ibis.common.graph.Node")

    stack = deque()
    graph = {}

    if filter(root):
        stack.append(root)

    while stack:
        if (node := stack.pop()) not in graph:
            children = tuple(child for child in node.__children__ if filter(child))
            graph[node] = children
            stack.extend(children)

    return Graph(reversed(graph.items()))
