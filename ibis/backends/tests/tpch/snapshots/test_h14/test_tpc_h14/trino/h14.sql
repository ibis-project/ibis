SELECT
  (
    SUM(IF(t1.p_type LIKE 'PROMO%', t0.l_extendedprice * (
      1 - t0.l_discount
    ), 0)) * 100
  ) / SUM(t0.l_extendedprice * (
    1 - t0.l_discount
  )) AS promo_revenue
FROM hive.ibis_sf1.lineitem AS t0
JOIN hive.ibis_sf1.part AS t1
  ON t0.l_partkey = t1.p_partkey
WHERE
  t0.l_shipdate >= FROM_ISO8601_DATE('1995-09-01')
  AND t0.l_shipdate < FROM_ISO8601_DATE('1995-10-01')