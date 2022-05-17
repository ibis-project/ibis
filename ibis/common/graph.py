from collections import deque
from functools import cached_property

import ibis.util as util


# or rather call it Tree
class Graph:
    def __init__(self, root, node_types):
        self.root = root
        self.node_types = node_types

    @cached_property
    def dependencies(self):
        result = {}

        stack = [self.root]
        while stack:
            if (node := stack.pop()) not in result:
                if util.is_iterable(node):
                    children = tuple(
                        child
                        for child in node
                        if isinstance(child, self.node_types)
                    )
                else:
                    children = tuple()

                result[node] = children
                stack.extend(children)

        return result

    @cached_property
    def dependents(self):
        """Convert dependencies to dependents.

        Parameters
        ----------
        dependencies
            A mapping of [`ops.Node`][ibis.expr.operations.Node]s to a set of
            that node's `ops.Node` dependencies.

        Returns
        -------
        Graph
            A mapping of [`ops.Node`][ibis.expr.operations.Node]s to a set of
            that node's `ops.Node` dependents.
        """
        dependents = {src: [] for src in self.dependencies.keys()}
        for src, dests in self.dependencies.items():
            for dest in dests:
                dependents[dest].append(src)
        return dependents

    def toposort(self):
        """Topologically sort `graph` using Kahn's algorithm.

        Parameters
        ----------
        graph
            A DAG built from an ibis expression.

        Yields
        ------
        Node
            An operation node
        """
        dependencies, dependents = self.dependencies, self.dependents

        # TODO(kszucs): could do this with one iteration
        in_degree = {node: len(deps) for node, deps in dependencies.items()}
        queue = deque(node for node, count in in_degree.items() if not count)

        while queue:
            node = queue.popleft()
            yield node

            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if not in_degree[dependent]:
                    queue.append(dependent)

        if any(in_degree.values()):
            raise ValueError("cycle in expression graph")

    # add another function with weakreffed results mapping just returning the
    # final node
    def map(self, fn):
        results = {}
        for node in self.toposort():
            kwargs = {
                name: results[arg] if isinstance(arg, self.node_types) else arg
                for name, arg in zip(node.__argnames__, node.__args__)
            }
            results[node] = fn(node, **kwargs)
        return results
