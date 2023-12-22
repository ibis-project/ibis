SELECT
  t7.l_shipmode,
  t7.high_line_count,
  t7.low_line_count
FROM (
  SELECT
    t6.l_shipmode,
    SUM(
      CASE t6.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(1 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(1 AS TINYINT)
        ELSE CAST(0 AS TINYINT)
      END
    ) AS high_line_count,
    SUM(
      CASE t6.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(0 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(0 AS TINYINT)
        ELSE CAST(1 AS TINYINT)
      END
    ) AS low_line_count
  FROM (
    SELECT
      t5.o_orderkey,
      t5.o_custkey,
      t5.o_orderstatus,
      t5.o_totalprice,
      t5.o_orderdate,
      t5.o_orderpriority,
      t5.o_clerk,
      t5.o_shippriority,
      t5.o_comment,
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
    FROM (
      SELECT
        t2.o_orderkey,
        t2.o_custkey,
        t2.o_orderstatus,
        t2.o_totalprice,
        t2.o_orderdate,
        t2.o_orderpriority,
        t2.o_clerk,
        t2.o_shippriority,
        t2.o_comment,
        t3.l_orderkey,
        t3.l_partkey,
        t3.l_suppkey,
        t3.l_linenumber,
        t3.l_quantity,
        t3.l_extendedprice,
        t3.l_discount,
        t3.l_tax,
        t3.l_returnflag,
        t3.l_linestatus,
        t3.l_shipdate,
        t3.l_commitdate,
        t3.l_receiptdate,
        t3.l_shipinstruct,
        t3.l_shipmode,
        t3.l_comment
      FROM orders AS t2
      INNER JOIN lineitem AS t3
        ON t2.o_orderkey = t3.l_orderkey
    ) AS t5
    WHERE
      t5.l_shipmode IN ('MAIL', 'SHIP')
      AND t5.l_commitdate < t5.l_receiptdate
      AND t5.l_shipdate < t5.l_commitdate
      AND t5.l_receiptdate >= MAKE_DATE(1994, 1, 1)
      AND t5.l_receiptdate < MAKE_DATE(1995, 1, 1)
  ) AS t6
  GROUP BY
    1
) AS t7
ORDER BY
  t7.l_shipmode ASC