SELECT
  *
FROM airlines AS t0
WHERE
  (
    (
      CAST(t0.dest AS BIGINT) = CAST(0 AS TINYINT)
    ) = TRUE
  )