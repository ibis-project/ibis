import collections
import itertools

from toolz import compose, identity

import ibis.expr.operations as ops
import ibis.expr.types as ir


class Container:

    __slots__ = ('data',)

    def __init__(self, data):
        self.data = collections.deque(self.visitor(data))

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
    """Wrapper around `collections.deque`.

    Implements the `Container` API for depth-first graph traversal.
    """

    __slots__ = ('data',)

    def get(self):
        return self.data.pop()

    @property
    def visitor(self):
        return compose(reversed, list)


class Queue(Container):
    """Wrapper around `collections.deque`.

    Implements the `Container` API for breadth-first graph traversal.
    """

    __slots__ = ('data',)

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
        return [col for col in result if col.get_name() == name]
    elif isinstance(op, ops.Aggregation):
        assert name is not None, 'name is None'
        return [
            col
            for col in itertools.chain(op.by, op.metrics)
            if col.get_name() == name
        ]
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

    c = container([(expr, expr.get_name() if expr.has_name() else None)])

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
            (arg, arg.get_name() if arg.has_name() else name)
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
        Only the instances if this expression type gets traversed.
    container: Union[Stack, Queue], default Stack
        Defines the traversing order. Use Stack for depth-first and
        Queue for breadth-first search.
    """
    args = expr if isinstance(expr, collections.abc.Iterable) else [expr]
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
