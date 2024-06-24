SELECT
  COUNT(`t2`.`foo`) AS `count`
FROM (
  SELECT
    `t1`.`string_col`,
    SUM(`t1`.`float_col`) AS `foo`
  FROM (
    SELECT
      *
    FROM `alltypes` AS `t0`
    WHERE
      `t0`.`timestamp_col` < DATETIME('2014-01-01T00:00:00')
  ) AS `t1`
  GROUP BY
    1
) AS `t2`