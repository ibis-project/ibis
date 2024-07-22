from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from ibis.common.grounds import Concrete

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ibis.expr.types as ir


class Selector(Concrete):
    """A column selector."""

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
