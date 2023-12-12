SELECT
  t16.s_name AS s_name,
  t16.numwait AS numwait
FROM (
  SELECT
    t15.s_name AS s_name,
    COUNT(*) AS numwait
  FROM (
    SELECT
      t12.l1_orderkey AS l1_orderkey,
      t12.o_orderstatus AS o_orderstatus,
      t12.l_receiptdate AS l_receiptdate,
      t12.l_commitdate AS l_commitdate,
      t12.l1_suppkey AS l1_suppkey,
      t12.s_name AS s_name,
      t12.n_name AS n_name
    FROM (
      SELECT
        t4.l_orderkey AS l1_orderkey,
        t7.o_orderstatus AS o_orderstatus,
        t4.l_receiptdate AS l_receiptdate,
        t4.l_commitdate AS l_commitdate,
        t4.l_suppkey AS l1_suppkey,
        t0.s_name AS s_name,
        t8.n_name AS n_name
      FROM supplier AS t0
      INNER JOIN lineitem AS t4
        ON t0.s_suppkey = t4.l_suppkey
      INNER JOIN orders AS t7
        ON t7.o_orderkey = t4.l_orderkey
      INNER JOIN nation AS t8
        ON t0.s_nationkey = t8.n_nationkey
    ) AS t12
    WHERE
      t12.o_orderstatus = 'F'
      AND t12.l_receiptdate > t12.l_commitdate
      AND t12.n_name = 'SAUDI ARABIA'
      AND EXISTS(
        (
          SELECT
            CAST(1 AS TINYINT) AS "1"
          FROM lineitem AS t5
          WHERE
            (
              t5.l_orderkey = t12.l1_orderkey
            ) AND (
              t5.l_suppkey <> t12.l1_suppkey
            )
        )
      )
      AND NOT (
        EXISTS(
          (
            SELECT
              CAST(1 AS TINYINT) AS "1"
            FROM lineitem AS t6
            WHERE
              (
                (
                  t6.l_orderkey = t12.l1_orderkey
                ) AND (
                  t6.l_suppkey <> t12.l1_suppkey
                )
              )
              AND (
                t6.l_receiptdate > t6.l_commitdate
              )
          )
        )
      )
  ) AS t15
  GROUP BY
    1
) AS t16
ORDER BY
  t16.numwait DESC,
  t16.s_name ASC
LIMIT 100