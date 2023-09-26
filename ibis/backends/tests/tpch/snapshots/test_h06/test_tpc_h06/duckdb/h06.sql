SELECT
  SUM(t0.l_extendedprice * t0.l_discount) AS revenue
FROM main.lineitem AS t0
WHERE
  t0.l_shipdate >= CAST('1994-01-01' AS DATE)
  AND t0.l_shipdate < CAST('1995-01-01' AS DATE)
  AND t0.l_discount BETWEEN CAST(0.05 AS REAL(53)) AND CAST(0.07 AS REAL(53))
  AND t0.l_quantity < CAST(24 AS TINYINT)