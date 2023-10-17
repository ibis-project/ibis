SELECT
  SUM(t0.l_extendedprice) / 7.0 AS avg_yearly
FROM hive.ibis_sf1.lineitem AS t0
JOIN hive.ibis_sf1.part AS t1
  ON t1.p_partkey = t0.l_partkey
WHERE
  t1.p_brand = 'Brand#23'
  AND t1.p_container = 'MED BOX'
  AND t0.l_quantity < (
    SELECT
      AVG(t0.l_quantity) AS "Mean(l_quantity)"
    FROM hive.ibis_sf1.lineitem AS t0
    WHERE
      t0.l_partkey = t1.p_partkey
  ) * 0.2