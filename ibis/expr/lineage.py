from __future__ import annotations

import collections
from typing import Any, Callable, Iterable, Iterator

import ibis.expr.operations as ops


class Container:

    __slots__ = "data", "visitor"

    def __init__(self, data, visitor: Callable | None = None) -> None:
        self.visitor = (lambda val: val) if visitor is None else visitor
        self.data = collections.deque(self.visitor(data))

    def append(self, item):
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def get(self):
        raise NotImplementedError('Child classes must implement get')

    def extend(self, items):
        return self.data.extend(items)


class Stack(Container):
    """Wrapper around `collections.deque`.

    Implements the `Container` API for depth-first graph traversal.
    """

    __slots__ = ()

    def __init__(self, data):
        super().__init__(data, visitor=lambda data: reversed(list(data)))

    def get(self):
        return self.data.pop()


class Queue(Container):
    """Wrapper around `collections.deque`.

    Implements the `Container` API for breadth-first graph traversal.
    """

    __slots__ = ()

    def get(self):
        return self.data.popleft()


# these could be callables instead
proceed = True
halt = False


def traverse(
    fn: Callable[[ops.Node], tuple[bool | Iterable, Any]],
    node: ops.Node | Iterable[ops.Node],
    container: Container = Stack,
    dedup: bool = True,
) -> Iterator[Any]:
    """Utility for generic expression tree traversal

    Parameters
    ----------
    fn
        A function applied on each expression. The first element of the tuple
        controls the traversal, and the second is the result if its not `None`.
    expr
        The traversable expression or a list of expressions.
    container
        Defines the traversal order. Use `Stack` for depth-first order and
        `Queue` for breadth-first order.
    dedup
        Whether to allow expression traversal more than once
    """
    args = node if isinstance(node, collections.abc.Iterable) else [node]
    todo = container(arg for arg in args if isinstance(arg, ops.Node))
    seen = set()

    while todo:
        node = todo.get()
        assert isinstance(node, ops.Node), type(node)

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
                args = node.flat_args()
            elif isinstance(control, collections.abc.Iterable):
                args = control
            else:
                raise TypeError(
                    'First item of the returned tuple must be '
                    'an instance of boolean or iterable'
                )

            todo.extend(
                arg for arg in todo.visitor(args) if isinstance(arg, ops.Node)
            )
