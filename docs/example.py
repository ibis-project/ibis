from __future__ import annotations

import ibis

con = ibis.sqlite.connect("crunchbase.db")

c = con.table("companies")
i = con.table("investments")

expr = (
    c.left_join(i, c.permalink == i.company_permalink)
    .group_by(investor_name=ibis.coalesce(i.investor_name, "NO INVESTOR"))
    .aggregate(
        num_investments=c.permalink.nunique(),
        acq_ipos=(
            c.status.isin(("ipo", "acquired")).ifelse(c.permalink, ibis.NA).nunique()
        ),
    )
    .mutate(acq_rate=lambda t: t.acq_ipos / t.num_investments)
    .order_by(ibis.desc(2))
)
