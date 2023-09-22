SELECT
  CASE
    WHEN (
      0 <= t0.`value`
    ) AND (
      t0.`value` < 1
    )
    THEN 0
    WHEN (
      1 <= t0.`value`
    ) AND (
      t0.`value` <= 3
    )
    THEN 1
    ELSE CAST(NULL AS INT64)
  END AS `tmp`
FROM t AS t0