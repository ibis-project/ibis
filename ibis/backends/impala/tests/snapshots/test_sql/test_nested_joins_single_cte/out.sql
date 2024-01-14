SELECT
  `t6`.`uuid`,
  `t2`.`CountStar()`,
  `t4`.`last_visit`
FROM (
  SELECT
    `t1`.`uuid`,
    MAX(`t1`.`CountStar()`) AS `max_count`
  FROM (
    SELECT
      `t0`.`uuid`,
      COUNT(*) AS `CountStar()`
    FROM `t` AS `t0`
    GROUP BY
      1
  ) AS `t1`
  GROUP BY
    1
) AS `t6`
LEFT OUTER JOIN (
  SELECT
    `t0`.`uuid`,
    COUNT(*) AS `CountStar()`
  FROM `t` AS `t0`
  GROUP BY
    1
) AS `t2`
  ON `t6`.`uuid` = `t2`.`uuid` AND `t6`.`max_count` = `t2`.`CountStar()`
LEFT OUTER JOIN (
  SELECT
    `t0`.`uuid`,
    MAX(`t0`.`ts`) AS `last_visit`
  FROM `t` AS `t0`
  GROUP BY
    1
) AS `t4`
  ON `t6`.`uuid` = `t4`.`uuid`