SELECT
  t19.n_name,
  t19.revenue
FROM (
  SELECT
    t18.n_name,
    SUM(t18.l_extendedprice * (
      CAST(1 AS TINYINT) - t18.l_discount
    )) AS revenue
  FROM (
    SELECT
      t17.c_custkey,
      t17.c_name,
      t17.c_address,
      t17.c_nationkey,
      t17.c_phone,
      t17.c_acctbal,
      t17.c_mktsegment,
      t17.c_comment,
      t17.o_orderkey,
      t17.o_custkey,
      t17.o_orderstatus,
      t17.o_totalprice,
      t17.o_orderdate,
      t17.o_orderpriority,
      t17.o_clerk,
      t17.o_shippriority,
      t17.o_comment,
      t17.l_orderkey,
      t17.l_partkey,
      t17.l_suppkey,
      t17.l_linenumber,
      t17.l_quantity,
      t17.l_extendedprice,
      t17.l_discount,
      t17.l_tax,
      t17.l_returnflag,
      t17.l_linestatus,
      t17.l_shipdate,
      t17.l_commitdate,
      t17.l_receiptdate,
      t17.l_shipinstruct,
      t17.l_shipmode,
      t17.l_comment,
      t17.s_suppkey,
      t17.s_name,
      t17.s_address,
      t17.s_nationkey,
      t17.s_phone,
      t17.s_acctbal,
      t17.s_comment,
      t17.n_nationkey,
      t17.n_name,
      t17.n_regionkey,
      t17.n_comment,
      t17.r_regionkey,
      t17.r_name,
      t17.r_comment
    FROM (
      SELECT
        t6.c_custkey,
        t6.c_name,
        t6.c_address,
        t6.c_nationkey,
        t6.c_phone,
        t6.c_acctbal,
        t6.c_mktsegment,
        t6.c_comment,
        t7.o_orderkey,
        t7.o_custkey,
        t7.o_orderstatus,
        t7.o_totalprice,
        t7.o_orderdate,
        t7.o_orderpriority,
        t7.o_clerk,
        t7.o_shippriority,
        t7.o_comment,
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
        t8.l_comment,
        t9.s_suppkey,
        t9.s_name,
        t9.s_address,
        t9.s_nationkey,
        t9.s_phone,
        t9.s_acctbal,
        t9.s_comment,
        t10.n_nationkey,
        t10.n_name,
        t10.n_regionkey,
        t10.n_comment,
        t11.r_regionkey,
        t11.r_name,
        t11.r_comment
      FROM customer AS t6
      INNER JOIN orders AS t7
        ON t6.c_custkey = t7.o_custkey
      INNER JOIN lineitem AS t8
        ON t8.l_orderkey = t7.o_orderkey
      INNER JOIN supplier AS t9
        ON t8.l_suppkey = t9.s_suppkey
      INNER JOIN nation AS t10
        ON t6.c_nationkey = t9.s_nationkey AND t9.s_nationkey = t10.n_nationkey
      INNER JOIN region AS t11
        ON t10.n_regionkey = t11.r_regionkey
    ) AS t17
    WHERE
      t17.r_name = 'ASIA'
      AND t17.o_orderdate >= MAKE_DATE(1994, 1, 1)
      AND t17.o_orderdate < MAKE_DATE(1995, 1, 1)
  ) AS t18
  GROUP BY
    1
) AS t19
ORDER BY
  t19.revenue DESC