SELECT
  t12.c_custkey AS c_custkey,
  t12.c_name AS c_name,
  t12.revenue AS revenue,
  t12.c_acctbal AS c_acctbal,
  t12.n_name AS n_name,
  t12.c_address AS c_address,
  t12.c_phone AS c_phone,
  t12.c_comment AS c_comment
FROM (
  SELECT
    t11.c_custkey AS c_custkey,
    t11.c_name AS c_name,
    t11.c_acctbal AS c_acctbal,
    t11.n_name AS n_name,
    t11.c_address AS c_address,
    t11.c_phone AS c_phone,
    t11.c_comment AS c_comment,
    SUM(t11.l_extendedprice * (
      CAST(1 AS TINYINT) - t11.l_discount
    )) AS revenue
  FROM (
    SELECT
      t10.c_custkey AS c_custkey,
      t10.c_name AS c_name,
      t10.c_address AS c_address,
      t10.c_nationkey AS c_nationkey,
      t10.c_phone AS c_phone,
      t10.c_acctbal AS c_acctbal,
      t10.c_mktsegment AS c_mktsegment,
      t10.c_comment AS c_comment,
      t10.o_orderkey AS o_orderkey,
      t10.o_custkey AS o_custkey,
      t10.o_orderstatus AS o_orderstatus,
      t10.o_totalprice AS o_totalprice,
      t10.o_orderdate AS o_orderdate,
      t10.o_orderpriority AS o_orderpriority,
      t10.o_clerk AS o_clerk,
      t10.o_shippriority AS o_shippriority,
      t10.o_comment AS o_comment,
      t10.l_orderkey AS l_orderkey,
      t10.l_partkey AS l_partkey,
      t10.l_suppkey AS l_suppkey,
      t10.l_linenumber AS l_linenumber,
      t10.l_quantity AS l_quantity,
      t10.l_extendedprice AS l_extendedprice,
      t10.l_discount AS l_discount,
      t10.l_tax AS l_tax,
      t10.l_returnflag AS l_returnflag,
      t10.l_linestatus AS l_linestatus,
      t10.l_shipdate AS l_shipdate,
      t10.l_commitdate AS l_commitdate,
      t10.l_receiptdate AS l_receiptdate,
      t10.l_shipinstruct AS l_shipinstruct,
      t10.l_shipmode AS l_shipmode,
      t10.l_comment AS l_comment,
      t10.n_nationkey AS n_nationkey,
      t10.n_name AS n_name,
      t10.n_regionkey AS n_regionkey,
      t10.n_comment AS n_comment
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
        t4.o_orderkey AS o_orderkey,
        t4.o_custkey AS o_custkey,
        t4.o_orderstatus AS o_orderstatus,
        t4.o_totalprice AS o_totalprice,
        t4.o_orderdate AS o_orderdate,
        t4.o_orderpriority AS o_orderpriority,
        t4.o_clerk AS o_clerk,
        t4.o_shippriority AS o_shippriority,
        t4.o_comment AS o_comment,
        t5.l_orderkey AS l_orderkey,
        t5.l_partkey AS l_partkey,
        t5.l_suppkey AS l_suppkey,
        t5.l_linenumber AS l_linenumber,
        t5.l_quantity AS l_quantity,
        t5.l_extendedprice AS l_extendedprice,
        t5.l_discount AS l_discount,
        t5.l_tax AS l_tax,
        t5.l_returnflag AS l_returnflag,
        t5.l_linestatus AS l_linestatus,
        t5.l_shipdate AS l_shipdate,
        t5.l_commitdate AS l_commitdate,
        t5.l_receiptdate AS l_receiptdate,
        t5.l_shipinstruct AS l_shipinstruct,
        t5.l_shipmode AS l_shipmode,
        t5.l_comment AS l_comment,
        t6.n_nationkey AS n_nationkey,
        t6.n_name AS n_name,
        t6.n_regionkey AS n_regionkey,
        t6.n_comment AS n_comment
      FROM customer AS t0
      INNER JOIN orders AS t4
        ON t0.c_custkey = t4.o_custkey
      INNER JOIN lineitem AS t5
        ON t5.l_orderkey = t4.o_orderkey
      INNER JOIN nation AS t6
        ON t0.c_nationkey = t6.n_nationkey
    ) AS t10
    WHERE
      t10.o_orderdate >= MAKE_DATE(1993, 10, 1)
      AND t10.o_orderdate < MAKE_DATE(1994, 1, 1)
      AND t10.l_returnflag = 'R'
  ) AS t11
  GROUP BY
    1,
    2,
    3,
    4,
    5,
    6,
    7
) AS t12
ORDER BY
  t12.revenue DESC
LIMIT 20