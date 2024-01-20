SELECT
  SUM(t0.l_extendedprice * t0.l_discount) AS revenue
FROM main.lineitem AS t0
WHERE
  t0.l_shipdate >= MAKE_DATE(1994, 1, 1)
  AND t0.l_shipdate < MAKE_DATE(1995, 1, 1)
  AND t0.l_discount BETWEEN CAST(0.05 AS DOUBLE) AND CAST(0.07 AS DOUBLE)
  AND t0.l_quantity < CAST(24 AS TINYINT)