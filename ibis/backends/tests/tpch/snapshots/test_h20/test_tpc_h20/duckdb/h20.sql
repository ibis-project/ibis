SELECT
  *
FROM (
  SELECT
    t3.s_name,
    t3.s_address
  FROM (
    SELECT
      *
    FROM (
      SELECT
        *
      FROM "supplier" AS t0
      INNER JOIN "nation" AS t1
        ON (
          t0.s_nationkey
        ) = (
          t1.n_nationkey
        )
    ) AS t2
    WHERE
      (
        t2.n_name
      ) = (
        'CANADA'
      )
      AND t2.s_suppkey IN (
        SELECT
          t1.ps_suppkey
        FROM (
          SELECT
            *
          FROM "partsupp" AS t0
          WHERE
            t0.ps_partkey IN (
              SELECT
                t1.p_partkey
              FROM (
                SELECT
                  *
                FROM "part" AS t0
                WHERE
                  t0.p_name LIKE 'forest%'
              ) AS t1
            )
            AND (
              t0.ps_availqty
            ) > (
              (
                SELECT
                  SUM(t0.l_quantity) AS "Sum(l_quantity)"
                FROM "lineitem" AS t0
                WHERE
                  (
                    t0.l_partkey
                  ) = (
                    ps_partkey
                  )
                  AND (
                    t0.l_suppkey
                  ) = (
                    ps_suppkey
                  )
                  AND (
                    t0.l_shipdate
                  ) >= (
                    MAKE_DATE(1994, 1, 1)
                  )
                  AND (
                    t0.l_shipdate
                  ) < (
                    MAKE_DATE(1995, 1, 1)
                  )
              ) * (
                CAST(0.5 AS DOUBLE)
              )
            )
        ) AS t1
      )
  ) AS t3
) AS t4
ORDER BY
  t4.s_name ASC