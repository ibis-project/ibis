SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      t13.s_name AS s_name,
      COUNT(*) AS numwait
    FROM (
      SELECT
        *
      FROM (
        SELECT
          t1.l_orderkey AS l1_orderkey,
          t2.o_orderstatus AS o_orderstatus,
          t1.l_receiptdate AS l_receiptdate,
          t1.l_commitdate AS l_commitdate,
          t1.l_suppkey AS l1_suppkey,
          t0.s_name AS s_name,
          t3.n_name AS n_name
        FROM "supplier" AS t0
        INNER JOIN "lineitem" AS t1
          ON t0.s_suppkey = t1.l_suppkey
        INNER JOIN "orders" AS t2
          ON t2.o_orderkey = t1.l_orderkey
        INNER JOIN "nation" AS t3
          ON t0.s_nationkey = t3.n_nationkey
      ) AS t8
      WHERE
        (
          t8.o_orderstatus = 'F'
        )
        AND (
          t8.l_receiptdate > t8.l_commitdate
        )
        AND (
          t8.n_name = 'SAUDI ARABIA'
        )
        AND EXISTS(
          (
            SELECT
              CAST(1 AS TINYINT) AS "1"
            FROM (
              SELECT
                *
              FROM "lineitem" AS t4
              WHERE
                (
                  (
                    t4.l_orderkey = t8.l1_orderkey
                  ) AND (
                    t4.l_suppkey <> t8.l1_suppkey
                  )
                )
            ) AS t9
          )
        )
        AND NOT EXISTS(
          (
            SELECT
              CAST(1 AS TINYINT) AS "1"
            FROM (
              SELECT
                *
              FROM "lineitem" AS t4
              WHERE
                (
                  (
                    (
                      t4.l_orderkey = t8.l1_orderkey
                    ) AND (
                      t4.l_suppkey <> t8.l1_suppkey
                    )
                  )
                  AND (
                    t4.l_receiptdate > t4.l_commitdate
                  )
                )
            ) AS t11
          )
        )
    ) AS t13
    GROUP BY
      1
  ) AS t14
  ORDER BY
    t14.numwait DESC,
    t14.s_name ASC
) AS t15
LIMIT 100