from __future__ import annotations

import collections
import itertools
import math
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Any, TypeVar

from ibis.common.graph import Node
from ibis.util import promote_list

K = TypeVar("K", bound=Hashable)


class DisjointSet(Mapping[K, set[K]]):
    """Disjoint set data structure.

    Also known as union-find data structure. It is a data structure that keeps
    track of a set of elements partitioned into a number of disjoint (non-overlapping)
    subsets. It provides near-constant-time operations to add new sets, to merge
    existing sets, and to determine whether elements are in the same set.

    Parameters
    ----------
    data :
        Initial data to add to the disjoint set.

    Examples
    --------
    >>> ds = DisjointSet()
    >>> ds.add(1)
    1
    >>> ds.add(2)
    2
    >>> ds.add(3)
    3
    >>> ds.union(1, 2)
    True
    >>> ds.union(2, 3)
    True
    >>> ds.find(1)
    1
    >>> ds.find(2)
    1
    >>> ds.find(3)
    1
    >>> ds.union(1, 3)
    False
    """

    __slots__ = ("_parents", "_classes")
    _parents: dict
    _classes: dict

    def __init__(self, data: Iterable[K] | None = None):
        self._parents = {}
        self._classes = {}
        if data is not None:
            for id in data:
                self.add(id)

    def __contains__(self, id) -> bool:
        """Check if the given id is in the disjoint set.

        Parameters
        ----------
        id :
            The id to check.

        Returns
        -------
        ined:
            True if the id is in the disjoint set, False otherwise.
        """
        return id in self._parents

    def __getitem__(self, id) -> set[K]:
        """Get the set of ids that are in the same class as the given id.

        Parameters
        ----------
        id :
            The id to get the class for.

        Returns
        -------
        class:
            The set of ids that are in the same class as the given id, including
            the given id.
        """
        id = self._parents[id]
        return self._classes[id]

    def __iter__(self) -> Iterator[K]:
        """Iterate over the ids in the disjoint set."""
        return iter(self._parents)

    def __len__(self) -> int:
        """Get the number of ids in the disjoint set."""
        return len(self._parents)

    def __eq__(self, other: object) -> bool:
        """Check if the disjoint set is equal to another disjoint set.

        Parameters
        ----------
        other :
            The other disjoint set to compare to.

        Returns
        -------
        equal:
            True if the disjoint sets are equal, False otherwise.
        """
        if not isinstance(other, DisjointSet):
            return NotImplemented
        return self._parents == other._parents

    def add(self, id: K) -> K:
        """Add a new id to the disjoint set.

        If the id is not in the disjoint set, it will be added to the disjoint set
        along with a new class containing only the given id.

        Parameters
        ----------
        id :
            The id to add to the disjoint set.

        Returns
        -------
        id:
            The id that was added to the disjoint set.
        """
        if id in self._parents:
            return self._parents[id]
        self._parents[id] = id
        self._classes[id] = {id}
        return id

    def find(self, id: K) -> K:
        """Find the root of the class that the given id is in.

        Also called as the canonicalized id or the representative id.

        Parameters
        ----------
        id :
            The id to find the canonicalized id for.

        Returns
        -------
        id:
            The canonicalized id for the given id.
        """
        return self._parents[id]

    def union(self, id1, id2) -> bool:
        """Merge the classes that the given ids are in.

        If the ids are already in the same class, this will return False. Otherwise
        it will merge the classes and return True.

        Parameters
        ----------
        id1 :
            The first id to merge the classes for.
        id2 :
            The second id to merge the classes for.

        Returns
        -------
        merged:
            True if the classes were merged, False otherwise.
        """
        # Find the root of each class
        id1 = self._parents[id1]
        id2 = self._parents[id2]
        if id1 == id2:
            return False

        # Merge the smaller eclass into the larger one, aka. union-find by size
        class1 = self._classes[id1]
        class2 = self._classes[id2]
        if len(class1) >= len(class2):
            id1, id2 = id2, id1
            class1, class2 = class2, class1

        # Update the parent pointers, this is called path compression but done
        # during the union operation to keep the find operation minimal
        for id in class1:
            self._parents[id] = id2

        # Do the actual merging and clear the other eclass
        class2 |= class1
        class1.clear()

        return True

    def connected(self, id1, id2):
        """Check if the given ids are in the same class.

        True if both ids have the same canonicalized id, False otherwise.

        Parameters
        ----------
        id1 :
            The first id to check.
        id2 :
            The second id to check.

        Returns
        -------
        connected:
            True if the ids are connected, False otherwise.
        """
        return self._parents[id1] == self._parents[id2]

    def verify(self):
        """Verify that the disjoint set is not corrupted.

        Check that each id's canonicalized id's class. In general corruption
        should not happen if the public API is used, but this is a sanity check
        to make sure that the internal data structures are not corrupted.

        Returns
        -------
        verified:
            True if the disjoint set is not corrupted, False otherwise.
        """
        for id in self._parents:
            if id not in self._classes[self._parents[id]]:
                raise RuntimeError(
                    f"DisjointSet is corrupted: {id} is not in its class"
                )


class Slotted:
    """A lightweight alternative to `ibis.common.grounds.Concrete`.

    This class is used to create immutable dataclasses with slots and a precomputed
    hash value for quicker dictionary lookups.
    """

    __slots__ = ("__precomputed_hash__",)
    __precomputed_hash__: int

    def __init__(self, *args):
        for name, value in itertools.zip_longest(self.__slots__, args):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "__precomputed_hash__", hash(args))

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        for name in self.__slots__:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self):
        return self.__precomputed_hash__

    def __setattr__(self, name, value):
        raise AttributeError("Can't set attributes on immutable ENode instance")


class Variable(Slotted):
    """A named capture in a pattern.

    Parameters
    ----------
    name : str
        The name of the variable.
    """

    __slots__ = ("name",)
    name: str

    def __init__(self, name: str):
        if name is None:
            raise ValueError("Variable name cannot be None")
        super().__init__(name)

    def __repr__(self):
        return f"${self.name}"

    def substitute(self, egraph, enode, subst):
        """Substitute the variable with the corresponding value in the substitution.

        Parameters
        ----------
        egraph : EGraph
            The egraph instance.
        enode : ENode
            The matched enode.
        subst : dict
            The substitution dictionary.

        Returns
        -------
        value : Any
            The substituted value.
        """
        return subst[self.name]


# Pattern corresponds to a selection which is flattened to a join of selections
class Pattern(Slotted):
    """A non-ground term, tree of enodes possibly containing variables.

    This class is used to represent a pattern in a query. The pattern is almost
    identical to an ENode, except that it can contain variables.

    Parameters
    ----------
    head : type
        The head or python type of the ENode to match against.
    args : tuple
        The arguments of the pattern. The arguments can be enodes, patterns,
        variables or leaf values.
    name : str, optional
        The name of the pattern which is used to refer to it in a rewrite rule.
    """

    __slots__ = ("head", "args", "name")
    head: type
    args: tuple
    name: str | None

    # TODO(kszucs): consider to raise if the pattern matches none
    def __init__(self, head, args, name=None, conditions=None):
        # TODO(kszucs): ensure that args are either patterns, variables or leaf values
        assert all(not isinstance(arg, (ENode, Node)) for arg in args)
        super().__init__(head, tuple(args), name)

    def matches_none(self):
        """Evaluate whether the pattern is guaranteed to match nothing.

        This can be evaluated before the matching loop starts, so eventually can
        be eliminated from the flattened query.
        """
        return len(self.head.__argnames__) != len(self.args)

    def matches_all(self):
        """Evaluate whether the pattern is guaranteed to match everything.

        This can be evaluated before the matching loop starts, so eventually can
        be eliminated from the flattened query.
        """
        return not self.matches_none() and all(
            isinstance(arg, Variable) for arg in self.args
        )

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"P{self.head.__name__}({argstring})"

    def __rshift__(self, rhs):
        """Syntax sugar to create a rewrite rule."""
        return Rewrite(self, rhs)

    def __rmatmul__(self, name):
        """Syntax sugar to create a named pattern."""
        return self.__class__(self.head, self.args, name)

    def flatten(self, var=None, counter=None):
        """Recursively flatten the pattern to a join of selections.

        `Pattern(Add, (Pattern(Mul, ($x, 1)), $y))` is turned into a join of
        selections by introducing auxiliary variables where each selection gets
        executed as a dictionary lookup.

        In SQL terms this is equivalent to the following query:
        SELECT m.0 AS $x, a.1 AS $y FROM Add a JOIN Mul m ON a.0 = m.id WHERE m.1 = 1

        Parameters
        ----------
        var : Variable
            The variable to assign to the flattened pattern.
        counter : Iterator[int]
            The counter to generate unique variable names for auxiliary variables
            connecting the selections.

        Yields
        ------
        (var, pattern) : tuple[Variable, Pattern]
            The variable and the flattened pattern where the flattened pattern
            cannot contain any patterns just variables.
        """
        # TODO(kszucs): convert a pattern to a query object instead by flattening it
        counter = counter or itertools.count()

        if var is None:
            if self.name is None:
                var = Variable(next(counter))
            else:
                var = Variable(self.name)

        args = []
        for arg in self.args:
            if isinstance(arg, Pattern):
                if arg.name is None:
                    aux = Variable(next(counter))
                else:
                    aux = Variable(arg.name)
                yield from arg.flatten(aux, counter)
                args.append(aux)
            else:
                args.append(arg)

        yield (var, Pattern(self.head, args))

    def substitute(self, egraph, enode, subst):
        """Substitute the variables in the pattern with the corresponding values.

        Parameters
        ----------
        egraph : EGraph
            The egraph instance.
        enode : ENode
            The matched enode.
        subst : dict
            The substitution dictionary.

        Returns
        -------
        enode : ENode
            The substituted pattern which is a ground term aka. an ENode.
        """
        args = []
        for arg in self.args:
            if isinstance(arg, (Variable, Pattern)):
                arg = arg.substitute(egraph, enode, subst)
            args.append(arg)
        return ENode(self.head, tuple(args))


class DynamicApplier(Slotted):
    """A dynamic applier which calls a function to compute the result."""

    __slots__ = ("func",)
    func: Callable

    def substitute(self, egraph, enode, subst):
        kwargs = {k: v for k, v in subst.items() if isinstance(k, str)}
        result = self.func(egraph, enode, **kwargs)
        if not isinstance(result, ENode):
            raise TypeError(f"applier must return an ENode, got {type(result)}")
        return result


class Rewrite(Slotted):
    """A rewrite rule which matches a pattern and applies a pattern or a function."""

    __slots__ = ("matcher", "applier")
    matcher: Pattern
    applier: Callable | Pattern | Variable

    def __init__(self, matcher, applier):
        if callable(applier):
            applier = DynamicApplier(applier)
        elif not isinstance(applier, (Pattern, Variable)):
            raise TypeError(
                "applier must be a Pattern or a Variable returning an ENode"
            )
        super().__init__(matcher, applier)

    def __repr__(self):
        return f"{self.lhs} >> {self.rhs}"


class ENode(Slotted, Node):
    """A ground term which is a node in the EGraph, called ENode.

    Parameters
    ----------
    head : type
        The type of the Node the ENode represents.
    args : tuple
        The arguments of the ENode which are either ENodes or leaf values.
    """

    __slots__ = ("head", "args")
    head: type
    args: tuple

    def __init__(self, head, args):
        # TODO(kszucs): ensure that it is a ground term, this check should be removed
        assert all(not isinstance(arg, (Pattern, Variable)) for arg in args)
        super().__init__(head, tuple(args))

    @property
    def __argnames__(self):
        """Implementation for the `ibis.common.graph.Node` protocol."""
        return self.head.__argnames__

    @property
    def __args__(self):
        """Implementation for the `ibis.common.graph.Node` protocol."""
        return self.args

    def __repr__(self):
        argstring = ", ".join(map(repr, self.args))
        return f"E{self.head.__name__}({argstring})"

    def __lt__(self, other):
        return False

    @classmethod
    def from_node(cls, node: Any):
        """Convert an `ibis.common.graph.Node` to an `ENode`."""

        def mapper(node, _, **kwargs):
            return cls(node.__class__, kwargs.values())

        return node.map(mapper)[node]

    def to_node(self):
        """Convert the ENode back to an `ibis.common.graph.Node`."""

        def mapper(node, _, **kwargs):
            return node.head(**kwargs)

        return self.map(mapper)[self]


# TODO: move every E* into the Egraph so its API only uses Nodes
# TODO: track whether the egraph is saturated or not
# TODO: support parent classes in etables (Join <= InnerJoin)


class EGraph:
    __slots__ = ("_nodes", "_etables", "_eclasses")
    _nodes: dict
    _etables: collections.defaultdict
    _eclasses: DisjointSet

    def __init__(self):
        # store the nodes before converting them to enodes, so we can spare the initial
        # node traversal and omit the creation of enodes
        self._nodes = {}
        # map enode heads to their eclass ids and their arguments, this is required for
        # the relational e-matching (Node => dict[type, tuple[Union[ENode, Any], ...]])
        self._etables = collections.defaultdict(dict)
        # map enodes to their eclass, this is the heart of the egraph
        self._eclasses = DisjointSet()

    def __repr__(self):
        return f"EGraph({self._eclasses})"

    def _as_enode(self, node: Node) -> ENode:
        """Convert a node to an enode."""
        # order is important here since ENode is a subclass of Node
        if isinstance(node, ENode):
            return node
        elif isinstance(node, Node):
            return self._nodes.get(node) or ENode.from_node(node)
        else:
            raise TypeError(node)

    def add(self, node: Node) -> ENode:
        """Add a node to the egraph.

        The node is converted to an enode and added to the egraph. If the enode is
        already present in the egraph, then the canonical enode is returned.

        Parameters
        ----------
        node :
            The node to add to the egraph.

        Returns
        -------
        enode :
            The canonical enode.
        """
        enode = self._as_enode(node)
        if enode in self._eclasses:
            return self._eclasses.find(enode)

        args = []
        for arg in enode.args:
            if isinstance(arg, ENode):
                args.append(self.add(arg))
            else:
                args.append(arg)

        enode = ENode(enode.head, args)
        self._eclasses.add(enode)
        self._etables[enode.head][enode] = tuple(args)

        return enode

    def union(self, node1: Node, node2: Node) -> ENode:
        """Union two nodes in the egraph.

        The nodes are converted to enodes which must be present in the egraph.
        The eclasses of the nodes are merged and the canonical enode is returned.

        Parameters
        ----------
        node1 :
            The first node to union.
        node2 :
            The second node to union.

        Returns
        -------
        enode :
            The canonical enode.
        """
        enode1 = self._as_enode(node1)
        enode2 = self._as_enode(node2)
        return self._eclasses.union(enode1, enode2)

    def _match_args(self, args, patargs):
        """Match the arguments of an enode against a pattern's arguments.

        An enode matches a pattern if each of the arguments are:
        - both leaf values and equal
        - both enodes and in the same eclass
        - an enode and a variable, in which case the variable gets bound to the enode

        Parameters
        ----------
        args : tuple
            The arguments of the enode. Since an enode is a ground term, the arguments
            are either enodes or leaf values.
        patargs : tuple
            The arguments of the pattern. Since a pattern is a flat term (flattened
            using auxiliary variables), the arguments are either variables or leaf
            values.

        Returns
        -------
        dict[str, Any] :
            The mapping of variable names to enodes or leaf values.
        """
        subst = {}
        for arg, patarg in zip(args, patargs):
            if isinstance(patarg, Variable):
                if isinstance(arg, ENode):
                    subst[patarg.name] = self._eclasses.find(arg)
                else:
                    subst[patarg.name] = arg
            # TODO(kszucs): this is not needed since patarg is either a variable or a
            # leaf value due to the pattern flattening, though we may choose to
            # support this in the future
            # elif isinstance(arg, ENode):
            #     if self._eclasses.find(arg) != self._eclasses.find(arg):
            #         return None
            elif patarg != arg:
                return None
        return subst

    def match(self, pattern: Pattern) -> dict[ENode, dict[str, Any]]:
        """Match a pattern in the egraph.

        The pattern is converted to a conjunctive query (list of flat patterns) and
        matched against the relations represented by the egraph. This is called the
        relational e-matching.

        Parameters
        ----------
        pattern :
            The pattern to match in the egraph.

        Returns
        -------
        matches :
            A dictionary mapping the matched enodes to their substitutions.
        """
        # patterns could be reordered to match on the most selective one first
        patterns = dict(reversed(list(pattern.flatten())))
        if any(pat.matches_none() for pat in patterns.values()):
            return {}

        # extract the first pattern
        (auxvar, pattern), *rest = patterns.items()
        matches = {}

        # match the first pattern and create the initial substitutions
        rel = self._etables[pattern.head]
        for enode, args in rel.items():
            if (subst := self._match_args(args, pattern.args)) is not None:
                subst[auxvar.name] = enode
                matches[enode] = subst

        # match the rest of the patterns and extend the substitutions
        for auxvar, pattern in rest:
            rel = self._etables[pattern.head]
            tmp = {}
            for enode, subst in matches.items():
                if args := rel.get(subst[auxvar.name]):
                    if (newsubst := self._match_args(args, pattern.args)) is not None:
                        tmp[enode] = {**subst, **newsubst}
            matches = tmp

        return matches

    def apply(self, rewrites: list[Rewrite]) -> int:
        """Apply the given rewrites to the egraph.

        Iteratively match the patterns and apply the rewrites to the graph. The returned
        number of changes is the number of eclasses that were merged. This is the
        number of changes made to the egraph. The egraph is saturated if the number of
        changes is zero.

        Parameters
        ----------
        rewrites :
            A list of rewrites to apply.

        Returns
        -------
        n_changes
            The number of changes made to the egraph.
        """
        n_changes = 0
        for rewrite in promote_list(rewrites):
            for match, subst in self.match(rewrite.matcher).items():
                enode = rewrite.applier.substitute(self, match, subst)
                enode = self.add(enode)
                n_changes += self._eclasses.union(match, enode)
        return n_changes

    def run(self, rewrites: list[Rewrite], n: int = 10) -> bool:
        """Run the match-apply cycles for the given number of iterations.

        Parameters
        ----------
        rewrites :
            A list of rewrites to apply.
        n :
            The number of iterations to run.

        Returns
        -------
        saturated :
            True if the egraph is saturated, False otherwise.
        """
        return any(not self.apply(rewrites) for _i in range(n))

    # TODO(kszucs): investigate whether the costs and best enodes could be maintained
    # during the union operations after each match-apply cycle
    def extract(self, node: Node) -> Node:
        """Extract a node from the egraph.

        The node is converted to an enode which recursively gets converted to an
        enode having the lowest cost according to equivalence classes. Currently
        the cost function is hardcoded as the depth of the enode.

        Parameters
        ----------
        node :
            The node to extract from the egraph.

        Returns
        -------
        node :
            The extracted node.
        """
        enode = self._as_enode(node)
        enode = self._eclasses.find(enode)
        costs = {en: (math.inf, None) for en in self._eclasses.keys()}

        def enode_cost(enode):
            cost = 1
            for arg in enode.args:
                if isinstance(arg, ENode):
                    cost += costs[arg][0]
                else:
                    cost += 1
            return cost

        changed = True
        while changed:
            changed = False
            for en, enodes in self._eclasses.items():
                new_cost = min((enode_cost(en), en) for en in enodes)
                if costs[en][0] != new_cost[0]:
                    changed = True
                costs[en] = new_cost

        def extract(en):
            if not isinstance(en, ENode):
                return en
            best = costs[en][1]
            args = tuple(extract(a) for a in best.args)
            return best.head(*args)

        return extract(enode)

    def equivalent(self, node1: Node, node2: Node) -> bool:
        """Check if two nodes are equivalent.

        The nodes are converted to enodes and checked for equivalence: they are
        equivalent if they are in the same equivalence class.

        Parameters
        ----------
        node1 :
            The first node.
        node2 :
            The second node.

        Returns
        -------
        equivalent :
            True if the nodes are equivalent, False otherwise.
        """
        enode1 = self._as_enode(node1)
        enode2 = self._as_enode(node2)
        enode1 = self._eclasses.find(enode1)
        enode2 = self._eclasses.find(enode2)
        return enode1 == enode2
