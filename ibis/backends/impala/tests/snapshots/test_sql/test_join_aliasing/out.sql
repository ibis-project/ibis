WITH `t5` AS (
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
)
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
      `t8`.`d`,
      `t8`.`idx`,
      `t8`.`c`,
      `t8`.`row_count`,
      `t10`.`total`
    FROM `t5` AS `t8`
    INNER JOIN (
      SELECT
        `t7`.`d`,
        SUM(`t7`.`row_count`) AS `total`
      FROM `t5` AS `t7`
      GROUP BY
        1
    ) AS `t10`
      ON `t8`.`d` = `t10`.`d`
  ) AS `t11`
  WHERE
    `t11`.`row_count` < (
      `t11`.`total` / 2
    )
) AS `t13`
  ON `t6`.`d` = `t13`.`d`