SELECT
  SUM(t0.l_extendedprice * t0.l_discount) AS revenue
FROM hive.ibis_sf1.lineitem AS t0
WHERE
  t0.l_shipdate >= FROM_ISO8601_DATE('1994-01-01')
  AND t0.l_shipdate < FROM_ISO8601_DATE('1995-01-01')
  AND t0.l_discount BETWEEN 0.05 AND 0.07
  AND t0.l_quantity < 24