SELECT
  *
FROM functional_alltypes AS t0
WHERE
  NOT (
    EXISTS(
      (
        SELECT
          CAST(1 AS TINYINT) AS "1"
        FROM (
          SELECT
            *
          FROM functional_alltypes AS t1
          WHERE
            (
              t0.string_col = t1.string_col
            )
        ) AS t2
      )
    )
  )