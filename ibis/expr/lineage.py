try:
    import queue as q
except ImportError:
    import Queue as q  # noqa

from itertools import chain
from toolz import identity, compose
from collections import deque, Iterable

import ibis.expr.types as ir
import ibis.expr.operations as ops


def roots(expr, types=(ops.PhysicalTable,)):
    """Yield every node of a particular type on which an expression depends.

    Parameters
    ----------
    expr : Expr
        The expression to analyze
    types : tuple(type), optional, default
            (:mod:`ibis.expr.operations.PhysicalTable`,)
        The node types to traverse

    Yields
    ------
    table : Expr
        Unique node types on which an expression depends

    Notes
    -----
    If your question is: "What nodes of type T does `expr` depend on?", then
    you've come to the right place. By default, we yield the physical tables
    that an expression depends on.
    """
    seen = set()

    stack = [arg for arg in reversed(expr.op().root_tables())
             if isinstance(arg, types)]

    while stack:
        table = stack.pop()

        if table not in seen:
            seen.add(table)
            yield table

        # flatten and reverse so that we traverse in preorder
        stack.extend(reversed(list(chain.from_iterable(
            arg.op().root_tables() for arg in table.flat_args()
            if isinstance(arg, types)
        ))))


class Container(object):

    __slots__ = 'data',

    def __init__(self, data):
        self.data = deque(self.visitor(data))

    def append(self, item):
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def get(self):
        raise NotImplementedError('Child classes must implement get')

    @property
    def visitor(self):
        raise NotImplementedError('Child classes must implement visitor')

    def extend(self, items):
        return self.data.extend(items)


class Stack(Container):

    """Wrapper around a list to provide a common API for graph traversal
    """

    __slots__ = 'data',

    def get(self):
        return self.data.pop()

    @property
    def visitor(self):
        return compose(reversed, list)


class Queue(Container):

    """Wrapper around a queue.Queue to provide a common API for graph traversal
    """

    __slots__ = 'data',

    def get(self):
        return self.data.popleft()

    @property
    def visitor(self):
        return identity


def _get_args(op, name):
    """Hack to get relevant arguments for lineage computation.

    We need a better way to determine the relevant arguments of an expression.
    """
    # Could use multipledispatch here to avoid the pasta
    if isinstance(op, ops.Selection):
        assert name is not None, 'name is None'
        result = op.selections

        # if Selection.selections is always columnar, could use an
        # OrderedDict to prevent scanning the whole thing
        return [col for col in result if col._name == name]
    elif isinstance(op, ops.Aggregation):
        assert name is not None, 'name is None'
        return [col for col in chain(op.by, op.metrics) if col._name == name]
    else:
        return op.args


def lineage(expr, container=Stack):
    """Yield the path of the expression tree that comprises a column
    expression.

    Parameters
    ----------
    expr : Expr
        An ibis expression. It must be an instance of
        :class:`ibis.expr.types.ColumnExpr`.
    container : Container, {Stack, Queue}
        Stack for depth-first traversal, and Queue for breadth-first.
        Depth-first will reach root table nodes before continuing on to other
        columns in a column that is derived from multiple column. Breadth-
        first will traverse all columns at each level before reaching root
        tables.

    Yields
    ------
    node : Expr
        A column and its dependencies
    """
    if not isinstance(expr, ir.ColumnExpr):
        raise TypeError('Input expression must be an instance of ColumnExpr')

    c = container([(expr, expr._name)])

    seen = set()

    # while we haven't visited everything
    while c:
        node, name = c.get()

        if node not in seen:
            seen.add(node)
            yield node

        # add our dependencies to the container if they match our name
        # and are ibis expressions
        c.extend(
            (arg, getattr(arg, '_name', name))
            for arg in c.visitor(_get_args(node.op(), name))
            if isinstance(arg, ir.Expr)
        )


# these could be callables instead
proceed = True
halt = False


def traverse(fn, expr, type=ir.Expr, container=Stack):
    """Utility for generic expression tree traversal

    Parameters
    ----------
    fn : Callable[[ir.Expr], Tuple[Union[Boolean, Iterable], Any]]
        This function will be applied on each expressions, it must
        return a tuple. The first element of the tuple controls the
        traversal, and the second is the result if its not None.
    expr: ir.Expr
        The traversable expression or a list of expressions.
    type: Type
        Only the instances if this type are traversed.
    container: Union[Stack, Queue], default Stack
        Defines the traversing order.
    """
    args = expr if isinstance(expr, Iterable) else [expr]
    todo = container(arg for arg in args if isinstance(arg, type))
    seen = set()

    while todo:
        expr = todo.get()
        op = expr.op()
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
            elif isinstance(control, Iterable):
                args = control
            else:
                raise TypeError('First item of the returned tuple must be '
                                'an instance of boolean or iterable')

            todo.extend(arg for arg in todo.visitor(args)
                        if isinstance(arg, type))
