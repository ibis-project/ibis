SELECT
  `t4`.`uuid`,
  `t2`.`CountStar()`
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
) AS `t4`
LEFT OUTER JOIN (
  SELECT
    `t0`.`uuid`,
    COUNT(*) AS `CountStar()`
  FROM `t` AS `t0`
  GROUP BY
    1
) AS `t2`
  ON `t4`.`uuid` = `t2`.`uuid`