from __future__ import annotations

import collections
from typing import Any, Callable, Iterable, Iterator

import ibis.expr.types as ir


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
    fn: Callable[[ir.Expr], tuple[bool | Iterable, Any]],
    expr: ir.Expr | Iterable[ir.Expr],
    type: type = ir.Expr,
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
    type
        Only the instances if this type are traversed.
    container
        Defines the traversal order. Use `Stack` for depth-first order and
        `Queue` for breadth-first order.
    dedup
        Whether to allow expression traversal more than once
    """
    args = expr if isinstance(expr, collections.abc.Iterable) else [expr]
    todo = container(arg for arg in args if isinstance(arg, type))
    seen = set()

    while todo:
        expr = todo.get()
        op = expr.op()

        if dedup:
            if op in seen:
                continue
            else:
                seen.add(op)

        control, result = fn(expr)
        if result is not None:
            yield result

        if control is not halt:
            if control is proceed:
                args = op.flat_args()
            elif isinstance(control, collections.abc.Iterable):
                args = control
            else:
                raise TypeError(
                    'First item of the returned tuple must be '
                    'an instance of boolean or iterable'
                )

            todo.extend(
                arg for arg in todo.visitor(args) if isinstance(arg, type)
            )
