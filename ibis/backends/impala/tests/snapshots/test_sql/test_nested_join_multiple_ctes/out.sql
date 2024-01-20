WITH `t6` AS (
  SELECT
    `t4`.`userid`,
    `t4`.`movieid`,
    `t4`.`rating`,
    `t4`.`datetime`,
    `t2`.`title`
  FROM (
    SELECT
      `t0`.`userid`,
      `t0`.`movieid`,
      `t0`.`rating`,
      CAST(`t0`.`timestamp` AS TIMESTAMP) AS `datetime`
    FROM `ratings` AS `t0`
  ) AS `t4`
  INNER JOIN `movies` AS `t2`
    ON `t4`.`movieid` = `t2`.`movieid`
)
SELECT
  `t8`.`userid`,
  `t8`.`movieid`,
  `t8`.`rating`,
  `t8`.`datetime`,
  `t8`.`title`
FROM (
  SELECT
    `t7`.`userid`,
    `t7`.`movieid`,
    `t7`.`rating`,
    `t7`.`datetime`,
    `t7`.`title`
  FROM `t6` AS `t7`
  WHERE
    `t7`.`userid` = 118205 AND EXTRACT(year FROM `t7`.`datetime`) > 2001
) AS `t8`
WHERE
  `t8`.`movieid` IN (
    SELECT
      `t7`.`movieid`
    FROM `t6` AS `t7`
    WHERE
      `t7`.`userid` = 118205
      AND EXTRACT(year FROM `t7`.`datetime`) > 2001
      AND EXTRACT(year FROM `t7`.`datetime`) < 2009
  )