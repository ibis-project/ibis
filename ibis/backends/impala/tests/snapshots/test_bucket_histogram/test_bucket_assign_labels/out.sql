SELECT
  CASE `tier`
    WHEN 0 THEN 'Under 0'
    WHEN 1 THEN '0 to 10'
    WHEN 2 THEN '10 to 25'
    WHEN 3 THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`, `count`
FROM (
  SELECT
    CASE
      WHEN `f` < 0 THEN 0
      WHEN (0 <= `f`) AND (`f` < 10) THEN 1
      WHEN (10 <= `f`) AND (`f` < 25) THEN 2
      WHEN (25 <= `f`) AND (`f` <= 50) THEN 3
      ELSE CAST(NULL AS tinyint)
    END AS `tier`, count(1) AS `count`
  FROM alltypes
  GROUP BY 1
) t0