SELECT
  `t7`.`street`,
  `t7`.`key`
FROM (
  SELECT
    `t5`.`street`,
    ROW_NUMBER() OVER (ORDER BY `t5`.`street` ASC) - 1 AS `key`
  FROM (
    SELECT
      `t2`.`street`,
      `t2`.`key`
    FROM (
      SELECT
        `t0`.`street`,
        ROW_NUMBER() OVER (ORDER BY `t0`.`street` ASC) - 1 AS `key`
      FROM `data` AS `t0`
    ) AS `t2`
    INNER JOIN (
      SELECT
        `t1`.`key`
      FROM (
        SELECT
          `t0`.`street`,
          ROW_NUMBER() OVER (ORDER BY `t0`.`street` ASC) - 1 AS `key`
        FROM `data` AS `t0`
      ) AS `t1`
    ) AS `t4`
      ON `t2`.`key` = `t4`.`key`
  ) AS `t5`
) AS `t7`
INNER JOIN (
  SELECT
    `t6`.`key`
  FROM (
    SELECT
      `t5`.`street`,
      ROW_NUMBER() OVER (ORDER BY `t5`.`street` ASC) - 1 AS `key`
    FROM (
      SELECT
        `t2`.`street`,
        `t2`.`key`
      FROM (
        SELECT
          `t0`.`street`,
          ROW_NUMBER() OVER (ORDER BY `t0`.`street` ASC) - 1 AS `key`
        FROM `data` AS `t0`
      ) AS `t2`
      INNER JOIN (
        SELECT
          `t1`.`key`
        FROM (
          SELECT
            `t0`.`street`,
            ROW_NUMBER() OVER (ORDER BY `t0`.`street` ASC) - 1 AS `key`
          FROM `data` AS `t0`
        ) AS `t1`
      ) AS `t4`
        ON `t2`.`key` = `t4`.`key`
    ) AS `t5`
  ) AS `t6`
) AS `t9`
  ON `t7`.`key` = `t9`.`key`