SELECT
  *
FROM foo_t AS t0
WHERE
  NOT (
    EXISTS(
      (
        SELECT
          CAST(1 AS TINYINT) AS "1"
        FROM (
          SELECT
            *
          FROM bar_t AS t1
          WHERE
            (
              t0.key1 = t1.key1
            )
        ) AS t2
      )
    )
  )