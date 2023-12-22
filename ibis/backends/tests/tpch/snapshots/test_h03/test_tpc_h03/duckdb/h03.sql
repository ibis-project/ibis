SELECT
  t10.l_orderkey,
  t10.revenue,
  t10.o_orderdate,
  t10.o_shippriority
FROM (
  SELECT
    t9.l_orderkey,
    t9.o_orderdate,
    t9.o_shippriority,
    SUM(t9.l_extendedprice * (
      CAST(1 AS TINYINT) - t9.l_discount
    )) AS revenue
  FROM (
    SELECT
      t8.c_custkey,
      t8.c_name,
      t8.c_address,
      t8.c_nationkey,
      t8.c_phone,
      t8.c_acctbal,
      t8.c_mktsegment,
      t8.c_comment,
      t8.o_orderkey,
      t8.o_custkey,
      t8.o_orderstatus,
      t8.o_totalprice,
      t8.o_orderdate,
      t8.o_orderpriority,
      t8.o_clerk,
      t8.o_shippriority,
      t8.o_comment,
      t8.l_orderkey,
      t8.l_partkey,
      t8.l_suppkey,
      t8.l_linenumber,
      t8.l_quantity,
      t8.l_extendedprice,
      t8.l_discount,
      t8.l_tax,
      t8.l_returnflag,
      t8.l_linestatus,
      t8.l_shipdate,
      t8.l_commitdate,
      t8.l_receiptdate,
      t8.l_shipinstruct,
      t8.l_shipmode,
      t8.l_comment
    FROM (
      SELECT
        t3.c_custkey,
        t3.c_name,
        t3.c_address,
        t3.c_nationkey,
        t3.c_phone,
        t3.c_acctbal,
        t3.c_mktsegment,
        t3.c_comment,
        t4.o_orderkey,
        t4.o_custkey,
        t4.o_orderstatus,
        t4.o_totalprice,
        t4.o_orderdate,
        t4.o_orderpriority,
        t4.o_clerk,
        t4.o_shippriority,
        t4.o_comment,
        t5.l_orderkey,
        t5.l_partkey,
        t5.l_suppkey,
        t5.l_linenumber,
        t5.l_quantity,
        t5.l_extendedprice,
        t5.l_discount,
        t5.l_tax,
        t5.l_returnflag,
        t5.l_linestatus,
        t5.l_shipdate,
        t5.l_commitdate,
        t5.l_receiptdate,
        t5.l_shipinstruct,
        t5.l_shipmode,
        t5.l_comment
      FROM customer AS t3
      INNER JOIN orders AS t4
        ON t3.c_custkey = t4.o_custkey
      INNER JOIN lineitem AS t5
        ON t5.l_orderkey = t4.o_orderkey
    ) AS t8
    WHERE
      t8.c_mktsegment = 'BUILDING'
      AND t8.o_orderdate < MAKE_DATE(1995, 3, 15)
      AND t8.l_shipdate > MAKE_DATE(1995, 3, 15)
  ) AS t9
  GROUP BY
    1,
    2,
    3
) AS t10
ORDER BY
  t10.revenue DESC,
  t10.o_orderdate ASC
LIMIT 10