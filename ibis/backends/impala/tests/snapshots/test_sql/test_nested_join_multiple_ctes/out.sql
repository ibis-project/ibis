WITH `t5` AS (
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
  *
FROM (
  SELECT
    *
  FROM `t5` AS `t6`
  WHERE
    `t6`.`userid` = 118205 AND EXTRACT(year FROM `t6`.`datetime`) > 2001
) AS `t7`
WHERE
  `t7`.`movieid` IN (
    SELECT
      `t6`.`movieid`
    FROM `t5` AS `t6`
    WHERE
      `t6`.`userid` = 118205
      AND EXTRACT(year FROM `t6`.`datetime`) > 2001
      AND EXTRACT(year FROM `t6`.`datetime`) < 2009
  )