SELECT
  `t6`.`d`,
  `t6`.`b`,
  `t6`.`count`,
  `t6`.`unique`,
  `t13`.`total`
FROM (
  SELECT
    `t1`.`d`,
    `t1`.`b`,
    COUNT(*) AS `count`,
    COUNT(DISTINCT `t1`.`c`) AS `unique`
  FROM (
    SELECT
      `t0`.`a`,
      `t0`.`b`,
      `t0`.`c`,
      `t0`.`a` + 20 AS `d`
    FROM `test_table` AS `t0`
  ) AS `t1`
  GROUP BY
    1,
    2
) AS `t6`
INNER JOIN (
  SELECT
    `t11`.`d`,
    `t11`.`idx`,
    `t11`.`c`,
    `t11`.`row_count`,
    `t11`.`total`
  FROM (
    SELECT
      `t7`.`d`,
      `t7`.`idx`,
      `t7`.`c`,
      `t7`.`row_count`,
      `t9`.`total`
    FROM (
      SELECT
        `t2`.`d`,
        CAST(`t2`.`d` / 15 AS BIGINT) AS `idx`,
        `t2`.`c`,
        COUNT(*) AS `row_count`
      FROM (
        SELECT
          `t0`.`a` + 20 AS `d`,
          `t0`.`c`
        FROM `test_table` AS `t0`
      ) AS `t2`
      GROUP BY
        1,
        2,
        3
    ) AS `t7`
    INNER JOIN (
      SELECT
        `t5`.`d`,
        SUM(`t5`.`row_count`) AS `total`
      FROM (
        SELECT
          `t2`.`d`,
          CAST(`t2`.`d` / 15 AS BIGINT) AS `idx`,
          `t2`.`c`,
          COUNT(*) AS `row_count`
        FROM (
          SELECT
            `t0`.`a` + 20 AS `d`,
            `t0`.`c`
          FROM `test_table` AS `t0`
        ) AS `t2`
        GROUP BY
          1,
          2,
          3
      ) AS `t5`
      GROUP BY
        1
    ) AS `t9`
      ON `t7`.`d` = `t9`.`d`
  ) AS `t11`
  WHERE
    `t11`.`row_count` < (
      `t11`.`total` / 2
    )
) AS `t13`
  ON `t6`.`d` = `t13`.`d`