CASE
  WHEN (0 <= `f`) AND (`f` <= 10) THEN 0
  WHEN (10 < `f`) AND (`f` <= 25) THEN 1
  WHEN (25 < `f`) AND (`f` <= 50) THEN 2
  ELSE CAST(NULL AS tinyint)
END