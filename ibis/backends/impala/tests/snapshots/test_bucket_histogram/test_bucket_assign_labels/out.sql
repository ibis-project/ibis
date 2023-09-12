SELECT
  CASE t0.`tier`
    WHEN 0 THEN 'Under 0'
    WHEN 1 THEN '0 to 10'
    WHEN 2 THEN '10 to 25'
    WHEN 3 THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`, t0.`CountStar(alltypes)`
FROM (
  SELECT
    CASE
      WHEN t1.`f` < 0 THEN 0
      WHEN (0 <= t1.`f`) AND (t1.`f` < 10) THEN 1
      WHEN (10 <= t1.`f`) AND (t1.`f` < 25) THEN 2
      WHEN (25 <= t1.`f`) AND (t1.`f` <= 50) THEN 3
      ELSE CAST(NULL AS tinyint)
    END AS `tier`, count(1) AS `CountStar(alltypes)`
  FROM `alltypes` t1
  GROUP BY 1
) t0