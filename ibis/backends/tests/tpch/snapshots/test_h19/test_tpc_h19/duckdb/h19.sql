SELECT
  SUM((
    t2.l_extendedprice * (
      CAST(1 AS TINYINT) - t2.l_discount
    )
  )) AS revenue
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
    (
      (
        (
          (
            (
              (
                (
                  (
                    t2.p_brand = 'Brand#12'
                  )
                  AND t2.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                )
                AND (
                  t2.l_quantity >= CAST(1 AS TINYINT)
                )
              )
              AND (
                t2.l_quantity <= CAST(11 AS TINYINT)
              )
            )
            AND t2.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(5 AS TINYINT)
          )
          AND t2.l_shipmode IN ('AIR', 'AIR REG')
        )
        AND (
          t2.l_shipinstruct = 'DELIVER IN PERSON'
        )
      )
      OR (
        (
          (
            (
              (
                (
                  (
                    t2.p_brand = 'Brand#23'
                  )
                  AND t2.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                )
                AND (
                  t2.l_quantity >= CAST(10 AS TINYINT)
                )
              )
              AND (
                t2.l_quantity <= CAST(20 AS TINYINT)
              )
            )
            AND t2.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(10 AS TINYINT)
          )
          AND t2.l_shipmode IN ('AIR', 'AIR REG')
        )
        AND (
          t2.l_shipinstruct = 'DELIVER IN PERSON'
        )
      )
    )
    OR (
      (
        (
          (
            (
              (
                (
                  t2.p_brand = 'Brand#34'
                )
                AND t2.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
              )
              AND (
                t2.l_quantity >= CAST(20 AS TINYINT)
              )
            )
            AND (
              t2.l_quantity <= CAST(30 AS TINYINT)
            )
          )
          AND t2.p_size BETWEEN CAST(1 AS TINYINT) AND CAST(15 AS TINYINT)
        )
        AND t2.l_shipmode IN ('AIR', 'AIR REG')
      )
      AND (
        t2.l_shipinstruct = 'DELIVER IN PERSON'
      )
    )
  )