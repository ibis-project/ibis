SELECT
  (
    SUM(
      CASE
        WHEN (
          t1.p_type LIKE 'PROMO%'
        )
        THEN t0.l_extendedprice * (
          CAST(1 AS TINYINT) - t0.l_discount
        )
        ELSE CAST(0 AS TINYINT)
      END
    ) * CAST(100 AS TINYINT)
  ) / SUM(t0.l_extendedprice * (
    CAST(1 AS TINYINT) - t0.l_discount
  )) AS promo_revenue
FROM main.lineitem AS t0
JOIN main.part AS t1
  ON t0.l_partkey = t1.p_partkey
WHERE
  t0.l_shipdate >= CAST('1995-09-01' AS DATE)
  AND t0.l_shipdate < CAST('1995-10-01' AS DATE)