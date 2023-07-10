SELECT
  t4.o_orderpriority AS o_orderpriority,
  t4.order_count AS order_count
FROM (
  SELECT
    t3.o_orderpriority AS o_orderpriority,
    COUNT(*) AS order_count
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
      t0.o_comment AS o_comment
    FROM orders AS t0
    WHERE
      EXISTS(
        (
          SELECT
            CAST(1 AS TINYINT) AS "1"
          FROM lineitem AS t1
          WHERE
            (
              t1.l_orderkey = t0.o_orderkey
            ) AND (
              t1.l_commitdate < t1.l_receiptdate
            )
        )
      )
      AND t0.o_orderdate >= MAKE_DATE(1993, 7, 1)
      AND t0.o_orderdate < MAKE_DATE(1993, 10, 1)
  ) AS t3
  GROUP BY
    1
) AS t4
ORDER BY
  t4.o_orderpriority ASC