SELECT
  CASE
    WHEN (0 <= `value`) AND (`value` < 1) THEN 0
    WHEN (1 <= `value`) AND (`value` <= 3) THEN 1
    ELSE CAST(NULL AS INT64)
  END AS `tmp`
FROM t