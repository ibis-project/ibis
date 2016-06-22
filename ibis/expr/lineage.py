import queue as q
from itertools import chain
from toolz import identity

import ibis.expr.types as ir
import ibis.expr.operations as ops


def roots(expr, types=(ops.PhysicalTable,)):
    """Yield every node of a particular type on which an expression depends.

    Parameters
    ----------
    expr : Expr
        The expression to analyze
    types : tuple(type), optional, default (:mod:`ibis.expr.operations.PhysicalTable`,)
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

    stack = list(reversed(expr.op().root_tables()))

    while stack:
        table = stack.pop()

        if table not in seen:
            seen.add(table)
            yield table
        else:
            # flatten and reverse so that we traverse in preorder
            stack.extend(reversed(list(chain.from_iterable(
                arg.op().root_tables() for arg in table.flat_args()
                if isinstance(arg, types)
            ))))


class Stack(object):

    """Wrapper around a list to provide a common API for graph traversal
    """

    __slots__ = 'stack',

    def __init__(self, stack=None):
        self.stack = stack if stack is not None else []

    def put(self, item):
        self.stack.append(item)

    def get(self):
        return self.stack.pop()

    @property
    def empty(self):
        return not self.stack

    @property
    def visitor(self):
        return reversed


class Queue(object):

    """Wrapper around a queue.Queue to provide a common API for graph traversal
    """

    __slots__ = 'queue',

    def __init__(self, queue=None):
        self.queue = q.Queue()
        if queue is not None:
            for item in queue:
                self.queue.put(item)

    def put(self, item):
        self.queue.put(item)

    def get(self):
        return self.queue.get()

    @property
    def empty(self):
        return self.queue.empty()

    @property
    def visitor(self):
        return identity


def _get_args(op, name):
    """Hack to get relevant arguments for lineage computation.

    We need a better way to determine the relevant arguments of an expression.
    """
    # Could use multipledispatch here to avoid the pasta
    if isinstance(op, ops.Selection):
        result = op.selections

        if name is None:
            if isinstance(op.table, ops.Join):
                result.append(op.table)
            return result
        else:
            # if Selection.selections is always columnar, could use an
            # OrderedDict to prevent scanning the whole thing
            result = [col for col in result if col._name == name]
            if isinstance(op.table, ops.Join):
                result.append(op.table)
            return result
    elif isinstance(op, ops.Aggregation):
        return [
            col for col in chain(op.by, op.agg_exprs)
            if col._name == name
        ]
    elif isinstance(op, ops.Join):
        if name is None or (
            name not in op.left.columns and name not in op.right.columns
        ):
            return [op.left, op.right]
        else:
            return [op.left if name in op.left.columns else op.right]
    else:
        return op.args


def lineage(expr, container=Stack):
    """Show the expression tree that comprises a column expression

    Parameters
    ----------
    expr : Expr

    Notes
    -----
    The order of graph traversal is configurable through the `container`
    parameter.

    Yields
    ------
    node : Expr
    """
    # TODO: seems a bit brittle, is everything fair game here?
    types = (
        ir.ArrayExpr,
        ir.TableExpr,
        ir.TableColumn,
        ops.TableNode,  # includes Selection, Aggregation, Join
    )

    if not isinstance(expr, ir.ArrayExpr):
        raise TypeError('Input expression must be a column')

    current_name = expr._name
    c = container([(expr, current_name)])

    seen = set()
    visitor = c.visitor

    # while we haven't visited everything
    while not c.empty:
        node, proposed_name = c.get()

        # if a new column name to traverse has appeared, look for that
        # TODO: enforce that all columns are named?
        if proposed_name is not None:
            current_name = proposed_name

        if node not in seen:
            seen.add(node)
            yield node

        # add our dependencies to the stack if they match our name or
        # are a valid expression to traverse
        for arg in visitor(_get_args(node.op(), current_name)):
            if isinstance(arg, types):
                c.put((arg, getattr(arg, '_name', None)))
