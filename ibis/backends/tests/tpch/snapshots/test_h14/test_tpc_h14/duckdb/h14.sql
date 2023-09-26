SELECT
  (
    (
      SUM(
        CASE
          WHEN t2.p_type LIKE 'PROMO%'
          THEN (
            t2.l_extendedprice * (
              CAST(1 AS TINYINT) - t2.l_discount
            )
          )
          ELSE CAST(0 AS TINYINT)
        END
      ) * CAST(100 AS TINYINT)
    ) / SUM((
      t2.l_extendedprice * (
        CAST(1 AS TINYINT) - t2.l_discount
      )
    ))
  ) AS promo_revenue
FROM (
  SELECT
    t0.*,
    t1.*
  FROM "lineitem" AS t0
  INNER JOIN "part" AS t1
    ON (
      t0.l_partkey = t1.p_partkey
    )
) AS t2
WHERE
  (
    t2.l_shipdate >= MAKE_DATE(1995, 9, 1)
  )
  AND (
    t2.l_shipdate < MAKE_DATE(1995, 10, 1)
  )