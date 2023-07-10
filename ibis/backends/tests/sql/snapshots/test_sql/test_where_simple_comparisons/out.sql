SELECT
  *
FROM star1 AS t0
WHERE
  (
    t0.f > CAST(0 AS TINYINT)
  ) AND (
    t0.c < (
      t0.f * CAST(2 AS TINYINT)
    )
  )