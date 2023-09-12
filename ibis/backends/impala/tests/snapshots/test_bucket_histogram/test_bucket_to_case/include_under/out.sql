CASE
  WHEN `f` < 0 THEN 0
  WHEN (0 <= `f`) AND (`f` < 10) THEN 1
  WHEN (10 <= `f`) AND (`f` < 25) THEN 2
  WHEN (25 <= `f`) AND (`f` <= 50) THEN 3
  ELSE CAST(NULL AS tinyint)
END