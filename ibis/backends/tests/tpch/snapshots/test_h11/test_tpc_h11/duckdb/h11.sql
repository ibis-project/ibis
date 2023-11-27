SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      t6.ps_partkey AS ps_partkey,
      SUM(t6.ps_supplycost * t6.ps_availqty) AS value
    FROM (
      SELECT
        *
      FROM (
        SELECT
          t0.ps_partkey AS ps_partkey,
          t0.ps_suppkey AS ps_suppkey,
          t0.ps_availqty AS ps_availqty,
          t0.ps_supplycost AS ps_supplycost,
          t0.ps_comment AS ps_comment,
          t1.s_suppkey AS s_suppkey,
          t1.s_name AS s_name,
          t1.s_address AS s_address,
          t1.s_nationkey AS s_nationkey,
          t1.s_phone AS s_phone,
          t1.s_acctbal AS s_acctbal,
          t1.s_comment AS s_comment,
          t2.n_nationkey AS n_nationkey,
          t2.n_name AS n_name,
          t2.n_regionkey AS n_regionkey,
          t2.n_comment AS n_comment
        FROM "partsupp" AS t0
        INNER JOIN "supplier" AS t1
          ON t0.ps_suppkey = t1.s_suppkey
        INNER JOIN "nation" AS t2
          ON t2.n_nationkey = t1.s_nationkey
      ) AS t5
      WHERE
        (
          t5.n_name = 'GERMANY'
        )
    ) AS t6
    GROUP BY
      1
  ) AS t7
  WHERE
    (
      t7.value > (
        (
          SELECT
            SUM(t6.ps_supplycost * t6.ps_availqty) AS "Sum(Multiply(ps_supplycost, ps_availqty))"
          FROM (
            SELECT
              *
            FROM (
              SELECT
                t0.ps_partkey AS ps_partkey,
                t0.ps_suppkey AS ps_suppkey,
                t0.ps_availqty AS ps_availqty,
                t0.ps_supplycost AS ps_supplycost,
                t0.ps_comment AS ps_comment,
                t1.s_suppkey AS s_suppkey,
                t1.s_name AS s_name,
                t1.s_address AS s_address,
                t1.s_nationkey AS s_nationkey,
                t1.s_phone AS s_phone,
                t1.s_acctbal AS s_acctbal,
                t1.s_comment AS s_comment,
                t2.n_nationkey AS n_nationkey,
                t2.n_name AS n_name,
                t2.n_regionkey AS n_regionkey,
                t2.n_comment AS n_comment
              FROM "partsupp" AS t0
              INNER JOIN "supplier" AS t1
                ON t0.ps_suppkey = t1.s_suppkey
              INNER JOIN "nation" AS t2
                ON t2.n_nationkey = t1.s_nationkey
            ) AS t5
            WHERE
              (
                t5.n_name = 'GERMANY'
              )
          ) AS t6
        ) * CAST(0.0001 AS DOUBLE)
      )
    )
) AS t9
ORDER BY
  t9.value DESC