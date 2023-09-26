SELECT
  t0.l_shipmode,
  t0.high_line_count,
  t0.low_line_count
FROM (
  SELECT
    t2.l_shipmode AS l_shipmode,
    SUM(
      CASE t1.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(1 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(1 AS TINYINT)
        ELSE CAST(0 AS TINYINT)
      END
    ) AS high_line_count,
    SUM(
      CASE t1.o_orderpriority
        WHEN '1-URGENT'
        THEN CAST(0 AS TINYINT)
        WHEN '2-HIGH'
        THEN CAST(0 AS TINYINT)
        ELSE CAST(1 AS TINYINT)
      END
    ) AS low_line_count
  FROM main.orders AS t1
  JOIN main.lineitem AS t2
    ON t1.o_orderkey = t2.l_orderkey
  WHERE
    t2.l_shipmode IN ('MAIL', 'SHIP')
    AND t2.l_commitdate < t2.l_receiptdate
    AND t2.l_shipdate < t2.l_commitdate
    AND t2.l_receiptdate >= CAST('1994-01-01' AS DATE)
    AND t2.l_receiptdate < CAST('1995-01-01' AS DATE)
  GROUP BY
    1
) AS t0
ORDER BY
  t0.l_shipmode ASC