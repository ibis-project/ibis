SELECT
  NOT (
    `t0`.`a` IN ('foo') AND NOT `t0`.`c` IS NULL
  ) AS `tmp`
FROM `t` AS `t0`