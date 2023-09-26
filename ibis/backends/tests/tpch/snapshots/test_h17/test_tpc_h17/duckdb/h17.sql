SELECT
  (
    SUM(t2.l_extendedprice) / CAST(7.0 AS DOUBLE)
  ) AS avg_yearly
FROM (
  SELECT
    t0.*,
    t1.*
  FROM "lineitem" AS t0
  INNER JOIN "part" AS t1
    ON (
      t1.p_partkey = t0.l_partkey
    )
) AS t2
WHERE
  (
    t2.p_brand = 'Brand#23'
  )
  AND (
    t2.p_container = 'MED BOX'
  )
  AND (
    t2.l_quantity < (
      (
        SELECT
          AVG(t0.l_quantity) AS "Mean(l_quantity)"
        FROM "lineitem" AS t0
        WHERE
          (
            t0.l_partkey = t2.p_partkey
          )
      ) * CAST(0.2 AS DOUBLE)
    )
  )