"""Compatibility layer for SQLGlot expressions used by the backends."""

from __future__ import annotations

import sqlglot.expressions as sge

# sqlglot >= 30.18 replaced `Drop`'s single `this` argument with a list of `tables`
# the old spelling is silently ignored rather than raising, so adapt it here and let
# the call sites keep passing a single table
if "tables" in sge.Drop.arg_types:

    def Drop(*, this, **kwargs):
        return sge.Drop(tables=[this], **kwargs)

else:
    Drop = sge.Drop
