from __future__ import annotations

from typing import Any

import sqlglot.expressions as sge


class TimeTravelTable(sge.Table):
    arg_types = {
        "this": True,
        "alias": False,
        "db": False,
        "catalog": False,
        "laterals": False,
        "joins": False,
        "pivots": False,
        "hints": False,
        "system_time": False,
        "version": False,
        "format": False,
        "pattern": False,
        "ordinality": False,
        "when": False,
        "timestamp": True,  # Added only this, rest copied from sge.Table.
    }

    @property
    def timestamp(self) -> Any:
        """Retrieves the argument with key 'timestamp'."""

        return self.args.get("timestamp")
