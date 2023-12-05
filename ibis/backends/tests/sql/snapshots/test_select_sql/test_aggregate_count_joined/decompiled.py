import ibis


tpch_nation = ibis.table(
    name="tpch_nation",
    schema={
        "n_nationkey": "int16",
        "n_name": "string",
        "n_regionkey": "int16",
        "n_comment": "string",
    },
)
tpch_region = ibis.table(
    name="tpch_region",
    schema={"r_regionkey": "int16", "r_name": "string", "r_comment": "string"},
)

result = (
    tpch_region.inner_join(
        tpch_nation, tpch_region.r_regionkey == tpch_nation.n_regionkey
    )
    .select([tpch_nation, tpch_region.r_name.name("region")])
    .count()
)
