SELECT
  t17.s_name,
  t17.numwait
FROM (
  SELECT
    t16.s_name,
    COUNT(*) AS numwait
  FROM (
    SELECT
      t13.l1_orderkey,
      t13.o_orderstatus,
      t13.l_receiptdate,
      t13.l_commitdate,
      t13.l1_suppkey,
      t13.s_name,
      t13.n_name
    FROM (
      SELECT
        t5.l_orderkey AS l1_orderkey,
        t8.o_orderstatus,
        t5.l_receiptdate,
        t5.l_commitdate,
        t5.l_suppkey AS l1_suppkey,
        t4.s_name,
        t9.n_name
      FROM supplier AS t4
      INNER JOIN lineitem AS t5
        ON t4.s_suppkey = t5.l_suppkey
      INNER JOIN orders AS t8
        ON t8.o_orderkey = t5.l_orderkey
      INNER JOIN nation AS t9
        ON t4.s_nationkey = t9.n_nationkey
    ) AS t13
    WHERE
      t13.o_orderstatus = 'F'
      AND t13.l_receiptdate > t13.l_commitdate
      AND t13.n_name = 'SAUDI ARABIA'
      AND EXISTS(
        SELECT
          CAST(1 AS TINYINT) AS "1"
        FROM lineitem AS t6
        WHERE
          (
            t6.l_orderkey = t13.l1_orderkey
          ) AND (
            t6.l_suppkey <> t13.l1_suppkey
          )
      )
      AND NOT (
        EXISTS(
          SELECT
            CAST(1 AS TINYINT) AS "1"
          FROM lineitem AS t7
          WHERE
            (
              (
                t7.l_orderkey = t13.l1_orderkey
              ) AND (
                t7.l_suppkey <> t13.l1_suppkey
              )
            )
            AND (
              t7.l_receiptdate > t7.l_commitdate
            )
        )
      )
  ) AS t16
  GROUP BY
    1
) AS t17
ORDER BY
  t17.numwait DESC,
  t17.s_name ASC
LIMIT 100