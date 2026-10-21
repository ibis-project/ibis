import ibis

customer = ibis.table(
    name="customer",
    schema={
        "c_custkey": "int64",
        "c_name": "string",
        "c_address": "string",
        "c_nationkey": "int32",
        "c_phone": "string",
        "c_acctbal": "decimal(15, 2)",
        "c_mktsegment": "string",
        "c_comment": "string",
    },
)
orders = ibis.table(
    name="orders",
    schema={
        "o_orderkey": "int64",
        "o_custkey": "int64",
        "o_orderstatus": "string",
        "o_totalprice": "decimal(15, 2)",
        "o_orderdate": "date",
        "o_orderpriority": "string",
        "o_clerk": "string",
        "o_shippriority": "int32",
        "o_comment": "string",
    },
)
lineitem = ibis.table(
    name="lineitem",
    schema={
        "l_orderkey": "int64",
        "l_partkey": "int64",
        "l_suppkey": "int64",
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
cast = ibis.literal("1995-03-15").cast("date")
joinchain = (
    customer.inner_join(orders, (customer.c_custkey == orders.o_custkey))
    .inner_join(lineitem, (lineitem.l_orderkey == orders.o_orderkey))
    .select(
        customer.c_custkey,
        customer.c_name,
        customer.c_address,
        customer.c_nationkey,
        customer.c_phone,
        customer.c_acctbal,
        customer.c_mktsegment,
        customer.c_comment,
        orders.o_orderkey,
        orders.o_custkey,
        orders.o_orderstatus,
        orders.o_totalprice,
        orders.o_orderdate,
        orders.o_orderpriority,
        orders.o_clerk,
        orders.o_shippriority,
        orders.o_comment,
        lineitem.l_orderkey,
        lineitem.l_partkey,
        lineitem.l_suppkey,
        lineitem.l_linenumber,
        lineitem.l_quantity,
        lineitem.l_extendedprice,
        lineitem.l_discount,
        lineitem.l_tax,
        lineitem.l_returnflag,
        lineitem.l_linestatus,
        lineitem.l_shipdate,
        lineitem.l_commitdate,
        lineitem.l_receiptdate,
        lineitem.l_shipinstruct,
        lineitem.l_shipmode,
        lineitem.l_comment,
    )
)
f = joinchain.filter(
    (joinchain.c_mktsegment == "BUILDING"),
    (joinchain.o_orderdate < cast),
    (joinchain.l_shipdate > cast),
)
agg = f.aggregate(
    [(f.l_extendedprice * ((ibis.literal(1) - f.l_discount))).sum().name("revenue")],
    by=[f.l_orderkey, f.o_orderdate, f.o_shippriority],
)
p = agg.select(agg.l_orderkey, agg.revenue, agg.o_orderdate, agg.o_shippriority)

result = p.order_by(p.revenue.desc(), p.o_orderdate.asc()).limit(10)
