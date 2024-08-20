import ibis


lineitem = ibis.table(
    name="lineitem",
    schema={
        "l_orderkey": "int32",
        "l_partkey": "int32",
        "l_suppkey": "int32",
        "l_linenumber": "int32",
        "l_quantity": "decimal(15, 2)",
        "l_extendedprice": "decimal(15, 2)",
        "l_discount": "decimal(15, 2)",
        "l_tax": "decimal(15, 2)",
        "l_returnflag": "string",
        "l_linestatus": "string",
        "l_shipdate": "date",
        "l_commitdate": "date",
        "l_receiptdate": "date",
        "l_shipinstruct": "string",
        "l_shipmode": "string",
        "l_comment": "string",
    },
)
lit = ibis.literal(1)
f = lineitem.filter((lineitem.l_shipdate <= ibis.literal("1998-09-02").cast("date")))
multiply = f.l_extendedprice * ((lit - f.l_discount))
agg = f.aggregate(
    [
        f.l_quantity.sum().name("sum_qty"),
        f.l_extendedprice.sum().name("sum_base_price"),
        multiply.sum().name("sum_disc_price"),
        ((multiply) * ((lit + f.l_tax))).sum().name("sum_charge"),
        f.l_quantity.mean().name("avg_qty"),
        f.l_extendedprice.mean().name("avg_price"),
        f.l_discount.mean().name("avg_disc"),
        f.count().name("count_order"),
    ],
    by=[f.l_returnflag, f.l_linestatus],
)

result = agg.order_by(
    agg.l_returnflag.asc(nulls_first=True), agg.l_linestatus.asc(nulls_first=True)
)
