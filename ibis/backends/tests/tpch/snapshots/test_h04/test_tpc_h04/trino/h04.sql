SELECT
  t0.o_orderpriority,
  COUNT(*) AS order_count
FROM hive.ibis_sf1.orders AS t0
WHERE
  (
    EXISTS(
      SELECT
        1 AS anon_1
      FROM hive.ibis_sf1.lineitem AS t1
      WHERE
        t1.l_orderkey = t0.o_orderkey AND t1.l_commitdate < t1.l_receiptdate
    )
  )
  AND t0.o_orderdate >= FROM_ISO8601_DATE('1993-07-01')
  AND t0.o_orderdate < FROM_ISO8601_DATE('1993-10-01')
GROUP BY
  1
ORDER BY
  t0.o_orderpriority ASC