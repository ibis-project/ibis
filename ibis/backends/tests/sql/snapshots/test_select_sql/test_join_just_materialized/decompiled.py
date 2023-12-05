import ibis


tpch_customer = ibis.table(
    name="tpch_customer",
    schema={
        "c_custkey": "int64",
        "c_name": "string",
        "c_address": "string",
        "c_nationkey": "int16",
        "c_phone": "string",
        "c_acctbal": "decimal",
        "c_mktsegment": "string",
        "c_comment": "string",
    },
)
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

result = tpch_nation.inner_join(
    tpch_region, tpch_nation.n_regionkey == tpch_region.r_regionkey
).inner_join(tpch_customer, tpch_nation.n_nationkey == tpch_customer.c_nationkey)
