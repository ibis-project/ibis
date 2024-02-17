from __future__ import annotations

from typing import Any

import sqlglot.expressions as sge


class TemporalJoin(sge.Join):
    arg_types = {
        "this": True,
        "on": False,
        "side": False,
        "kind": False,
        "using": False,
        "method": False,
        "global": False,
        "hint": False,
        "at_time": True,  # Added only this, rest copied from sge.Join.
    }

    @property
    def at_time(self) -> Any:
        """Retrieves the argument with key "at_time"."""

        return self.args.get("at_time")
