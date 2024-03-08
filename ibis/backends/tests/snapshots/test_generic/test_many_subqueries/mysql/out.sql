WITH `t1` AS (
  SELECT
    `t0`.`street`,
    ROW_NUMBER() OVER (ORDER BY CASE WHEN `t0`.`street` IS NULL THEN 1 ELSE 0 END, `t0`.`street` ASC) - 1 AS `key`
  FROM `data` AS `t0`
), `t7` AS (
  SELECT
    `t6`.`street`,
    ROW_NUMBER() OVER (ORDER BY CASE WHEN `t6`.`street` IS NULL THEN 1 ELSE 0 END, `t6`.`street` ASC) - 1 AS `key`
  FROM (
    SELECT
      `t3`.`street`,
      `t3`.`key`
    FROM `t1` AS `t3`
    INNER JOIN (
      SELECT
        `t2`.`key`
      FROM `t1` AS `t2`
    ) AS `t5`
      ON `t3`.`key` = `t5`.`key`
  ) AS `t6`
)
SELECT
  `t9`.`street`,
  `t9`.`key`
FROM `t7` AS `t9`
INNER JOIN (
  SELECT
    `t8`.`key`
  FROM `t7` AS `t8`
) AS `t11`
  ON `t9`.`key` = `t11`.`key`