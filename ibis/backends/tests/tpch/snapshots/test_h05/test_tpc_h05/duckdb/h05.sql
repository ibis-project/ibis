SELECT
  t18.n_name AS n_name,
  t18.revenue AS revenue
FROM (
  SELECT
    t17.n_name AS n_name,
    SUM(t17.l_extendedprice * (
      CAST(1 AS TINYINT) - t17.l_discount
    )) AS revenue
  FROM (
    SELECT
      t16.c_custkey AS c_custkey,
      t16.c_name AS c_name,
      t16.c_address AS c_address,
      t16.c_nationkey AS c_nationkey,
      t16.c_phone AS c_phone,
      t16.c_acctbal AS c_acctbal,
      t16.c_mktsegment AS c_mktsegment,
      t16.c_comment AS c_comment,
      t16.o_orderkey AS o_orderkey,
      t16.o_custkey AS o_custkey,
      t16.o_orderstatus AS o_orderstatus,
      t16.o_totalprice AS o_totalprice,
      t16.o_orderdate AS o_orderdate,
      t16.o_orderpriority AS o_orderpriority,
      t16.o_clerk AS o_clerk,
      t16.o_shippriority AS o_shippriority,
      t16.o_comment AS o_comment,
      t16.l_orderkey AS l_orderkey,
      t16.l_partkey AS l_partkey,
      t16.l_suppkey AS l_suppkey,
      t16.l_linenumber AS l_linenumber,
      t16.l_quantity AS l_quantity,
      t16.l_extendedprice AS l_extendedprice,
      t16.l_discount AS l_discount,
      t16.l_tax AS l_tax,
      t16.l_returnflag AS l_returnflag,
      t16.l_linestatus AS l_linestatus,
      t16.l_shipdate AS l_shipdate,
      t16.l_commitdate AS l_commitdate,
      t16.l_receiptdate AS l_receiptdate,
      t16.l_shipinstruct AS l_shipinstruct,
      t16.l_shipmode AS l_shipmode,
      t16.l_comment AS l_comment,
      t16.s_suppkey AS s_suppkey,
      t16.s_name AS s_name,
      t16.s_address AS s_address,
      t16.s_nationkey AS s_nationkey,
      t16.s_phone AS s_phone,
      t16.s_acctbal AS s_acctbal,
      t16.s_comment AS s_comment,
      t16.n_nationkey AS n_nationkey,
      t16.n_name AS n_name,
      t16.n_regionkey AS n_regionkey,
      t16.n_comment AS n_comment,
      t16.r_regionkey AS r_regionkey,
      t16.r_name AS r_name,
      t16.r_comment AS r_comment
    FROM (
      SELECT
        t0.c_custkey AS c_custkey,
        t0.c_name AS c_name,
        t0.c_address AS c_address,
        t0.c_nationkey AS c_nationkey,
        t0.c_phone AS c_phone,
        t0.c_acctbal AS c_acctbal,
        t0.c_mktsegment AS c_mktsegment,
        t0.c_comment AS c_comment,
        t6.o_orderkey AS o_orderkey,
        t6.o_custkey AS o_custkey,
        t6.o_orderstatus AS o_orderstatus,
        t6.o_totalprice AS o_totalprice,
        t6.o_orderdate AS o_orderdate,
        t6.o_orderpriority AS o_orderpriority,
        t6.o_clerk AS o_clerk,
        t6.o_shippriority AS o_shippriority,
        t6.o_comment AS o_comment,
        t7.l_orderkey AS l_orderkey,
        t7.l_partkey AS l_partkey,
        t7.l_suppkey AS l_suppkey,
        t7.l_linenumber AS l_linenumber,
        t7.l_quantity AS l_quantity,
        t7.l_extendedprice AS l_extendedprice,
        t7.l_discount AS l_discount,
        t7.l_tax AS l_tax,
        t7.l_returnflag AS l_returnflag,
        t7.l_linestatus AS l_linestatus,
        t7.l_shipdate AS l_shipdate,
        t7.l_commitdate AS l_commitdate,
        t7.l_receiptdate AS l_receiptdate,
        t7.l_shipinstruct AS l_shipinstruct,
        t7.l_shipmode AS l_shipmode,
        t7.l_comment AS l_comment,
        t8.s_suppkey AS s_suppkey,
        t8.s_name AS s_name,
        t8.s_address AS s_address,
        t8.s_nationkey AS s_nationkey,
        t8.s_phone AS s_phone,
        t8.s_acctbal AS s_acctbal,
        t8.s_comment AS s_comment,
        t9.n_nationkey AS n_nationkey,
        t9.n_name AS n_name,
        t9.n_regionkey AS n_regionkey,
        t9.n_comment AS n_comment,
        t10.r_regionkey AS r_regionkey,
        t10.r_name AS r_name,
        t10.r_comment AS r_comment
      FROM customer AS t0
      INNER JOIN orders AS t6
        ON t0.c_custkey = t6.o_custkey
      INNER JOIN lineitem AS t7
        ON t7.l_orderkey = t6.o_orderkey
      INNER JOIN supplier AS t8
        ON t7.l_suppkey = t8.s_suppkey
      INNER JOIN nation AS t9
        ON t0.c_nationkey = t8.s_nationkey AND t8.s_nationkey = t9.n_nationkey
      INNER JOIN region AS t10
        ON t9.n_regionkey = t10.r_regionkey
    ) AS t16
    WHERE
      t16.r_name = 'ASIA'
      AND t16.o_orderdate >= MAKE_DATE(1994, 1, 1)
      AND t16.o_orderdate < MAKE_DATE(1995, 1, 1)
  ) AS t17
  GROUP BY
    1
) AS t18
ORDER BY
  t18.revenue DESC