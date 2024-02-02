SELECT
  `t2`.`a`
FROM (
  SELECT
    *
  FROM `t0` AS `t0`
  INTERSECT DISTINCT
  SELECT
    *
  FROM `t1` AS `t1`
) AS `t2`