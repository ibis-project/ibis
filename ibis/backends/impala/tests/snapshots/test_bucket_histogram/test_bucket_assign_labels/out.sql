SELECT
  CASE `t1`.`tier`
    WHEN 0
    THEN 'Under 0'
    WHEN 1
    THEN '0 to 10'
    WHEN 2
    THEN '10 to 25'
    WHEN 3
    THEN '25 to 50'
    ELSE 'error'
  END AS `tier2`,
  `t1`.`CountStar(alltypes)`
FROM (
  SELECT
    CASE
      WHEN `t0`.`f` < 0
      THEN 0
      WHEN (
        0 <= `t0`.`f`
      ) AND (
        `t0`.`f` < 10
      )
      THEN 1
      WHEN (
        10 <= `t0`.`f`
      ) AND (
        `t0`.`f` < 25
      )
      THEN 2
      WHEN (
        25 <= `t0`.`f`
      ) AND (
        `t0`.`f` <= 50
      )
      THEN 3
      ELSE CAST(NULL AS TINYINT)
    END AS `tier`,
    COUNT(*) AS `CountStar(alltypes)`
  FROM `alltypes` AS `t0`
  GROUP BY
    1
) AS `t1`