import ibis

con = ibis.sqlite.connect("crunchbase.db")

c = con.table("companies")
i = con.table("investors")

expr = (
    c.left_join(i, c.permalink == i.company_permalink)
    .group_by(investor_name=lambda t: t.investor_name.ifnull("NO INVESTOR"))
    .aggregate(
        num_investments=lambda t: t.permalink.nunique(),
        acq_ipos=lambda t: t.status.isin(("ipo", "acquired"))
        .ifelse(t.permalink, ibis.NA)
        .nunique(),
    )
    .mutate(acq_rate=lambda t: t.acq_ipos / t.num_investments)
    .sort_by(ibis.desc(2))
)
