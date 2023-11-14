from __future__ import annotations

import sqlalchemy as sa

c = sa.table("companies")
i = sa.table("investments")

a = (
    sa.select(
        [
            sa.case(
                [(i.c.investor_name.is_(None), "NO INVESTOR")],
                else_=i.c.investor_name,
            ).label("investor_name"),
            sa.func.count(c.c.permalink.distinct()).label("num_investments"),
            sa.func.count(
                sa.case([(c.status.in_(("ipo", "acquired")), c.c.permalink)]).distinct()
            ).label("acq_ipos"),
        ]
    )
    .select_from(
        c.join(i, onclause=c.c.permalink == i.c.company_permalink, isouter=True)
    )
    .group_by(1)
    .order_by(sa.desc(2))
)
expr = sa.select([(a.c.acq_ipos / a.c.num_investments).label("acq_rate")])
