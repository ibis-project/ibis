SELECT
  CASE
    WHEN (
      t0.f > 0
    )
    THEN t0.d * 2
    WHEN (
      t0.c < 0
    )
    THEN t0.a * 2
    ELSE CAST(NULL AS BIGINT)
  END AS tmp
FROM alltypes AS t0