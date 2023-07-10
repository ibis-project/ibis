SELECT
  t9.l_orderkey AS l_orderkey,
  t9.revenue AS revenue,
  t9.o_orderdate AS o_orderdate,
  t9.o_shippriority AS o_shippriority
FROM (
  SELECT
    t8.l_orderkey AS l_orderkey,
    t8.o_orderdate AS o_orderdate,
    t8.o_shippriority AS o_shippriority,
    SUM(t8.l_extendedprice * (
      CAST(1 AS TINYINT) - t8.l_discount
    )) AS revenue
  FROM (
    SELECT
      t7.c_custkey AS c_custkey,
      t7.c_name AS c_name,
      t7.c_address AS c_address,
      t7.c_nationkey AS c_nationkey,
      t7.c_phone AS c_phone,
      t7.c_acctbal AS c_acctbal,
      t7.c_mktsegment AS c_mktsegment,
      t7.c_comment AS c_comment,
      t7.o_orderkey AS o_orderkey,
      t7.o_custkey AS o_custkey,
      t7.o_orderstatus AS o_orderstatus,
      t7.o_totalprice AS o_totalprice,
      t7.o_orderdate AS o_orderdate,
      t7.o_orderpriority AS o_orderpriority,
      t7.o_clerk AS o_clerk,
      t7.o_shippriority AS o_shippriority,
      t7.o_comment AS o_comment,
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
      t7.l_comment AS l_comment
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
        t3.o_orderkey AS o_orderkey,
        t3.o_custkey AS o_custkey,
        t3.o_orderstatus AS o_orderstatus,
        t3.o_totalprice AS o_totalprice,
        t3.o_orderdate AS o_orderdate,
        t3.o_orderpriority AS o_orderpriority,
        t3.o_clerk AS o_clerk,
        t3.o_shippriority AS o_shippriority,
        t3.o_comment AS o_comment,
        t4.l_orderkey AS l_orderkey,
        t4.l_partkey AS l_partkey,
        t4.l_suppkey AS l_suppkey,
        t4.l_linenumber AS l_linenumber,
        t4.l_quantity AS l_quantity,
        t4.l_extendedprice AS l_extendedprice,
        t4.l_discount AS l_discount,
        t4.l_tax AS l_tax,
        t4.l_returnflag AS l_returnflag,
        t4.l_linestatus AS l_linestatus,
        t4.l_shipdate AS l_shipdate,
        t4.l_commitdate AS l_commitdate,
        t4.l_receiptdate AS l_receiptdate,
        t4.l_shipinstruct AS l_shipinstruct,
        t4.l_shipmode AS l_shipmode,
        t4.l_comment AS l_comment
      FROM customer AS t0
      INNER JOIN orders AS t3
        ON t0.c_custkey = t3.o_custkey
      INNER JOIN lineitem AS t4
        ON t4.l_orderkey = t3.o_orderkey
    ) AS t7
    WHERE
      t7.c_mktsegment = 'BUILDING'
      AND t7.o_orderdate < MAKE_DATE(1995, 3, 15)
      AND t7.l_shipdate > MAKE_DATE(1995, 3, 15)
  ) AS t8
  GROUP BY
    1,
    2,
    3
) AS t9
ORDER BY
  t9.revenue DESC,
  t9.o_orderdate ASC
LIMIT 10