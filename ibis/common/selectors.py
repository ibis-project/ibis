from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from ibis.common.bases import Abstract
from ibis.common.grounds import Concrete
from ibis.common.typing import VarTuple  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir


class Expandable(Abstract):
    __slots__ = ()

    @abc.abstractmethod
    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        """Expand `table` into value expressions that match the selector.

        Parameters
        ----------
        table
            An ibis table expression

        Returns
        -------
        Sequence[Value]
            A sequence of value expressions that match the selector

        """


class Selector(Concrete, Expandable):
    """A column selector."""

    @abc.abstractmethod
    def expand_names(self, table: ir.Table) -> frozenset[str]:
        """Compute the set of column names that match the selector."""

    def expand(self, table: ir.Table) -> Sequence[ir.Value]:
        names = self.expand_names(table)
        return list(map(table.__getitem__, filter(names.__contains__, table.columns)))

    def __and__(self, other: Selector) -> Selector:
        """Compute the logical conjunction of two `Selector`s.

        Parameters
        ----------
        other
            Another selector
        """
        if not isinstance(other, Selector):
            return NotImplemented
        return And(self, other)

    def __or__(self, other: Selector) -> Selector:
        """Compute the logical disjunction of two `Selector`s.

        Parameters
        ----------
        other
            Another selector
        """
        if not isinstance(other, Selector):
            return NotImplemented
        return Or(self, other)

    def __invert__(self) -> Selector:
        """Compute the logical negation of a `Selector`."""
        return Not(self)


class Or(Selector):
    left: Selector
    right: Selector

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return self.left.expand_names(table) | self.right.expand_names(table)


class And(Selector):
    left: Selector
    right: Selector

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        return self.left.expand_names(table) & self.right.expand_names(table)


class Any(Selector):
    selectors: VarTuple[Selector]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        names = (selector.expand_names(table) for selector in self.selectors)
        return frozenset.union(*names)


class All(Selector):
    selectors: VarTuple[Selector]

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        names = (selector.expand_names(table) for selector in self.selectors)
        return frozenset.intersection(*names)


class Not(Selector):
    selector: Selector

    def expand_names(self, table: ir.Table) -> frozenset[str]:
        names = self.selector.expand_names(table)
        return frozenset(col for col in table.columns if col not in names)
