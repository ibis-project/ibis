SELECT
  t6.l_shipmode AS l_shipmode,
  t6.high_line_count AS high_line_count,
  t6.low_line_count AS low_line_count
FROM (
  SELECT
    t5.l_shipmode AS l_shipmode,
    SUM(
      CASE t5.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(1 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(1 AS TINYINT)
        ELSE CAST(0 AS TINYINT)
      END
    ) AS high_line_count,
    SUM(
      CASE t5.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(0 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(0 AS TINYINT)
        ELSE CAST(1 AS TINYINT)
      END
    ) AS low_line_count
  FROM (
    SELECT
      t4.o_orderkey AS o_orderkey,
      t4.o_custkey AS o_custkey,
      t4.o_orderstatus AS o_orderstatus,
      t4.o_totalprice AS o_totalprice,
      t4.o_orderdate AS o_orderdate,
      t4.o_orderpriority AS o_orderpriority,
      t4.o_clerk AS o_clerk,
      t4.o_shippriority AS o_shippriority,
      t4.o_comment AS o_comment,
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
    FROM (
      SELECT
        t0.o_orderkey AS o_orderkey,
        t0.o_custkey AS o_custkey,
        t0.o_orderstatus AS o_orderstatus,
        t0.o_totalprice AS o_totalprice,
        t0.o_orderdate AS o_orderdate,
        t0.o_orderpriority AS o_orderpriority,
        t0.o_clerk AS o_clerk,
        t0.o_shippriority AS o_shippriority,
        t0.o_comment AS o_comment,
        t2.l_orderkey AS l_orderkey,
        t2.l_partkey AS l_partkey,
        t2.l_suppkey AS l_suppkey,
        t2.l_linenumber AS l_linenumber,
        t2.l_quantity AS l_quantity,
        t2.l_extendedprice AS l_extendedprice,
        t2.l_discount AS l_discount,
        t2.l_tax AS l_tax,
        t2.l_returnflag AS l_returnflag,
        t2.l_linestatus AS l_linestatus,
        t2.l_shipdate AS l_shipdate,
        t2.l_commitdate AS l_commitdate,
        t2.l_receiptdate AS l_receiptdate,
        t2.l_shipinstruct AS l_shipinstruct,
        t2.l_shipmode AS l_shipmode,
        t2.l_comment AS l_comment
      FROM orders AS t0
      INNER JOIN lineitem AS t2
        ON t0.o_orderkey = t2.l_orderkey
    ) AS t4
    WHERE
      t4.l_shipmode IN ('MAIL', 'SHIP')
      AND t4.l_commitdate < t4.l_receiptdate
      AND t4.l_shipdate < t4.l_commitdate
      AND t4.l_receiptdate >= MAKE_DATE(1994, 1, 1)
      AND t4.l_receiptdate < MAKE_DATE(1995, 1, 1)
  ) AS t5
  GROUP BY
    1
) AS t6
ORDER BY
  t6.l_shipmode ASC