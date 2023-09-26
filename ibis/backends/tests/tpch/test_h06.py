from __future__ import annotations

import ibis

from .conftest import add_date, tpch_test


@tpch_test
def test_tpc_h06(lineitem):
    "Forecasting Revenue Change Query (Q6)"
    DATE = "1994-01-01"
    DISCOUNT = 0.06
    QUANTITY = 24

    q = lineitem
    discount_min = round(DISCOUNT - 0.01, 2)
    discount_max = round(DISCOUNT + 0.01, 2)
    q = q.filter(
        [
            q.l_shipdate >= ibis.date(DATE),
            q.l_shipdate < add_date(DATE, dy=1),
            q.l_discount.between(discount_min, discount_max),
            q.l_quantity < QUANTITY,
        ]
    )
    q = q.aggregate(revenue=(q.l_extendedprice * q.l_discount).sum())
    return q
