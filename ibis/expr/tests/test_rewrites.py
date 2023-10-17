from __future__ import annotations

from ibis.expr.datatypes import String
from ibis.expr.operations import *

e = Selection(
    table=Selection(
        table=UnboundTable(schema=dict(color="string"), name="t"),
        selections=(UnboundTable(schema=dict(color="string"), name="t"),),
        predicates=(
            StringSQLLike(
                arg=Lowercase(
                    arg=TableColumn(
                        table=UnboundTable(schema=dict(color="string"), name="t"),
                        name="color",
                    )
                ),
                pattern=Literal(value="%de%", dtype=String(nullable=True)),
                escape=None,
            ),
            StringContains(
                haystack=Lowercase(
                    arg=TableColumn(
                        table=UnboundTable(schema=dict(color="string"), name="t"),
                        name="color",
                    )
                ),
                needle=Literal(value="de", dtype=String(nullable=True)),
            ),
        ),
        sort_keys=(),
    ),
    selections=(
        Selection(
            table=UnboundTable(schema=dict(color="string"), name="t"),
            selections=(UnboundTable(schema=dict(color="string"), name="t"),),
            predicates=(
                StringSQLLike(
                    arg=Lowercase(
                        arg=TableColumn(
                            table=UnboundTable(schema=dict(color="string"), name="t"),
                            name="color",
                        )
                    ),
                    pattern=Literal(value="%de%", dtype=String(nullable=True)),
                    escape=None,
                ),
            ),
            sort_keys=(),
        ),
    ),
    predicates=(),
    sort_keys=(),
)
