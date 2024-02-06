WITH `t1` AS (
  SELECT
    `t0`.`uuid`,
    COUNT(*) AS `CountStar(t)`
  FROM `t` AS `t0`
  GROUP BY
    1
)
SELECT
  `t7`.`uuid`,
  `t4`.`CountStar(t)`,
  `t5`.`last_visit`
FROM (
  SELECT
    `t2`.`uuid`,
    MAX(`t2`.`CountStar(t)`) AS `max_count`
  FROM `t1` AS `t2`
  GROUP BY
    1
) AS `t7`
LEFT OUTER JOIN `t1` AS `t4`
  ON `t7`.`uuid` = `t4`.`uuid` AND `t7`.`max_count` = `t4`.`CountStar(t)`
LEFT OUTER JOIN (
  SELECT
    `t0`.`uuid`,
    MAX(`t0`.`ts`) AS `last_visit`
  FROM `t` AS `t0`
  GROUP BY
    1
) AS `t5`
  ON `t7`.`uuid` = `t5`.`uuid`