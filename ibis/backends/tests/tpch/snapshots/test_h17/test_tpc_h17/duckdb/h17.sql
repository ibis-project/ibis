SELECT
  SUM(t0.l_extendedprice) / CAST(7.0 AS REAL(53)) AS avg_yearly
FROM main.lineitem AS t0
JOIN main.part AS t1
  ON t1.p_partkey = t0.l_partkey
WHERE
  t1.p_brand = 'Brand#23'
  AND t1.p_container = 'MED BOX'
  AND t0.l_quantity < (
    SELECT
      AVG(t0.l_quantity) AS "Mean(l_quantity)"
    FROM main.lineitem AS t0
    WHERE
      t0.l_partkey = t1.p_partkey
  ) * CAST(0.2 AS REAL(53))