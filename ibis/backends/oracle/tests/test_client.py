from __future__ import annotations

import ibis


def test_ibis_is_not_defeated_by_statement_cache(con):
    con.execute(ibis.timestamp("2419-10-11 10:10:25").name("tmp"))
    con.execute(ibis.literal(0).name("tmp"))
