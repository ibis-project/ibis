WITH `t1` AS (
  SELECT
    `t0`.`uuid`,
    COUNT(*) AS `CountStar(t)`
  FROM `t` AS `t0`
  GROUP BY
    1
)
SELECT
  `t5`.`uuid`,
  `t3`.`CountStar(t)`
FROM (
  SELECT
    `t2`.`uuid`,
    MAX(`t2`.`CountStar(t)`) AS `max_count`
  FROM `t1` AS `t2`
  GROUP BY
    1
) AS `t5`
LEFT OUTER JOIN `t1` AS `t3`
  ON `t5`.`uuid` = `t3`.`uuid`