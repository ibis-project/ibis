SELECT
  *
FROM (
  SELECT
    *
  FROM `functional_alltypes` AS `t0`
  UNION DISTINCT
  SELECT
    *
  FROM `functional_alltypes` AS `t0`
) AS `t1`