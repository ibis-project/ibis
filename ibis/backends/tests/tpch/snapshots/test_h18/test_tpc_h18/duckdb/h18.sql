SELECT
  t11.c_name AS c_name,
  t11.c_custkey AS c_custkey,
  t11.o_orderkey AS o_orderkey,
  t11.o_orderdate AS o_orderdate,
  t11.o_totalprice AS o_totalprice,
  t11.sum_qty AS sum_qty
FROM (
  SELECT
    t10.c_name AS c_name,
    t10.c_custkey AS c_custkey,
    t10.o_orderkey AS o_orderkey,
    t10.o_orderdate AS o_orderdate,
    t10.o_totalprice AS o_totalprice,
    SUM(t10.l_quantity) AS sum_qty
  FROM (
    SELECT
      t8.c_custkey AS c_custkey,
      t8.c_name AS c_name,
      t8.c_address AS c_address,
      t8.c_nationkey AS c_nationkey,
      t8.c_phone AS c_phone,
      t8.c_acctbal AS c_acctbal,
      t8.c_mktsegment AS c_mktsegment,
      t8.c_comment AS c_comment,
      t8.o_orderkey AS o_orderkey,
      t8.o_custkey AS o_custkey,
      t8.o_orderstatus AS o_orderstatus,
      t8.o_totalprice AS o_totalprice,
      t8.o_orderdate AS o_orderdate,
      t8.o_orderpriority AS o_orderpriority,
      t8.o_clerk AS o_clerk,
      t8.o_shippriority AS o_shippriority,
      t8.o_comment AS o_comment,
      t8.l_orderkey AS l_orderkey,
      t8.l_partkey AS l_partkey,
      t8.l_suppkey AS l_suppkey,
      t8.l_linenumber AS l_linenumber,
      t8.l_quantity AS l_quantity,
      t8.l_extendedprice AS l_extendedprice,
      t8.l_discount AS l_discount,
      t8.l_tax AS l_tax,
      t8.l_returnflag AS l_returnflag,
      t8.l_linestatus AS l_linestatus,
      t8.l_shipdate AS l_shipdate,
      t8.l_commitdate AS l_commitdate,
      t8.l_receiptdate AS l_receiptdate,
      t8.l_shipinstruct AS l_shipinstruct,
      t8.l_shipmode AS l_shipmode,
      t8.l_comment AS l_comment
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
        ON t3.o_orderkey = t4.l_orderkey
    ) AS t8
    WHERE
      t8.o_orderkey IN ((
        SELECT
          t5.l_orderkey AS l_orderkey
        FROM (
          SELECT
            t2.l_orderkey AS l_orderkey,
            SUM(t2.l_quantity) AS qty_sum
          FROM lineitem AS t2
          GROUP BY
            1
        ) AS t5
        WHERE
          t5.qty_sum > CAST(300 AS SMALLINT)
      ))
  ) AS t10
  GROUP BY
    1,
    2,
    3,
    4,
    5
) AS t11
ORDER BY
  t11.o_totalprice DESC,
  t11.o_orderdate ASC
LIMIT 100