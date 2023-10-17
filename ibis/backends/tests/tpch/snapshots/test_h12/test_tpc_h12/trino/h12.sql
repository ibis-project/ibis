SELECT
  t0.l_shipmode,
  t0.high_line_count,
  t0.low_line_count
FROM (
  SELECT
    t2.l_shipmode AS l_shipmode,
    SUM(CASE t1.o_orderpriority WHEN '1-URGENT' THEN 1 WHEN '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
    SUM(CASE t1.o_orderpriority WHEN '1-URGENT' THEN 0 WHEN '2-HIGH' THEN 0 ELSE 1 END) AS low_line_count
  FROM hive.ibis_sf1.orders AS t1
  JOIN hive.ibis_sf1.lineitem AS t2
    ON t1.o_orderkey = t2.l_orderkey
  WHERE
    t2.l_shipmode IN ('MAIL', 'SHIP')
    AND t2.l_commitdate < t2.l_receiptdate
    AND t2.l_shipdate < t2.l_commitdate
    AND t2.l_receiptdate >= FROM_ISO8601_DATE('1994-01-01')
    AND t2.l_receiptdate < FROM_ISO8601_DATE('1995-01-01')
  GROUP BY
    1
) AS t0
ORDER BY
  t0.l_shipmode ASC