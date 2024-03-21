from __future__ import annotations

import sqlglot.expressions as sge


class WindowJoin(sge.Join):
    arg_types = {
        "this": True,
        "where": True,
    }


class AntiWindowJoin(WindowJoin):
    pass


class SemiWindowJoin(WindowJoin):
    pass
