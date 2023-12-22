SELECT
  t13.c_custkey,
  t13.c_name,
  t13.revenue,
  t13.c_acctbal,
  t13.n_name,
  t13.c_address,
  t13.c_phone,
  t13.c_comment
FROM (
  SELECT
    t12.c_custkey,
    t12.c_name,
    t12.c_acctbal,
    t12.n_name,
    t12.c_address,
    t12.c_phone,
    t12.c_comment,
    SUM(t12.l_extendedprice * (
      CAST(1 AS TINYINT) - t12.l_discount
    )) AS revenue
  FROM (
    SELECT
      t11.c_custkey,
      t11.c_name,
      t11.c_address,
      t11.c_nationkey,
      t11.c_phone,
      t11.c_acctbal,
      t11.c_mktsegment,
      t11.c_comment,
      t11.o_orderkey,
      t11.o_custkey,
      t11.o_orderstatus,
      t11.o_totalprice,
      t11.o_orderdate,
      t11.o_orderpriority,
      t11.o_clerk,
      t11.o_shippriority,
      t11.o_comment,
      t11.l_orderkey,
      t11.l_partkey,
      t11.l_suppkey,
      t11.l_linenumber,
      t11.l_quantity,
      t11.l_extendedprice,
      t11.l_discount,
      t11.l_tax,
      t11.l_returnflag,
      t11.l_linestatus,
      t11.l_shipdate,
      t11.l_commitdate,
      t11.l_receiptdate,
      t11.l_shipinstruct,
      t11.l_shipmode,
      t11.l_comment,
      t11.n_nationkey,
      t11.n_name,
      t11.n_regionkey,
      t11.n_comment
    FROM (
      SELECT
        t4.c_custkey,
        t4.c_name,
        t4.c_address,
        t4.c_nationkey,
        t4.c_phone,
        t4.c_acctbal,
        t4.c_mktsegment,
        t4.c_comment,
        t5.o_orderkey,
        t5.o_custkey,
        t5.o_orderstatus,
        t5.o_totalprice,
        t5.o_orderdate,
        t5.o_orderpriority,
        t5.o_clerk,
        t5.o_shippriority,
        t5.o_comment,
        t6.l_orderkey,
        t6.l_partkey,
        t6.l_suppkey,
        t6.l_linenumber,
        t6.l_quantity,
        t6.l_extendedprice,
        t6.l_discount,
        t6.l_tax,
        t6.l_returnflag,
        t6.l_linestatus,
        t6.l_shipdate,
        t6.l_commitdate,
        t6.l_receiptdate,
        t6.l_shipinstruct,
        t6.l_shipmode,
        t6.l_comment,
        t7.n_nationkey,
        t7.n_name,
        t7.n_regionkey,
        t7.n_comment
      FROM customer AS t4
      INNER JOIN orders AS t5
        ON t4.c_custkey = t5.o_custkey
      INNER JOIN lineitem AS t6
        ON t6.l_orderkey = t5.o_orderkey
      INNER JOIN nation AS t7
        ON t4.c_nationkey = t7.n_nationkey
    ) AS t11
    WHERE
      t11.o_orderdate >= MAKE_DATE(1993, 10, 1)
      AND t11.o_orderdate < MAKE_DATE(1994, 1, 1)
      AND t11.l_returnflag = 'R'
  ) AS t12
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS t13
ORDER BY
  t13.revenue DESC
LIMIT 20