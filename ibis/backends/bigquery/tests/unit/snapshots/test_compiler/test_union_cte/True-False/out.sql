WITH `t1` AS (
  SELECT
    `t0`.`string_col`,
    SUM(`t0`.`double_col`) AS `metric`
  FROM `functional_alltypes` AS `t0`
  GROUP BY
    1
)
SELECT
  `t7`.`string_col`,
  `t7`.`metric`
FROM (
  SELECT
    `t5`.`string_col`,
    `t5`.`metric`
  FROM (
    SELECT
      *
    FROM `t1` AS `t2`
    UNION ALL
    SELECT
      *
    FROM `t1` AS `t4`
  ) AS `t5`
  UNION DISTINCT
  SELECT
    *
  FROM `t1` AS `t3`
) AS `t7`