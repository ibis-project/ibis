SELECT
  `t8`.`num`,
  `t8`.`id`,
  `t8`.`window_start`,
  `t8`.`window_end`
FROM (
  SELECT
    `t6`.`num`,
    `t6`.`id`,
    `t6`.`window_start`,
    `t6`.`window_end`
  FROM (
    SELECT
      `t0`.*
    FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
  ) AS `t6`
  INNER JOIN (
    SELECT
      `t1`.*
    FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
  ) AS `t7`
    ON `t6`.`num` = `t7`.`num`
) AS `t8`
WHERE
  NOT CAST(EXISTS(
    SELECT
      1
    FROM (
      SELECT
        `t2`.*
      FROM TABLE(TUMBLE(TABLE `outer_tumble`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t2`
    ) AS `t5`
    WHERE
      `t8`.`num` = `t5`.`num`
  ) AS BOOLEAN)