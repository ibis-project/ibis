SELECT
  `t3`.`num`,
  `t3`.`id`,
  `t3`.`window_start`,
  `t3`.`window_end`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `outer_tumble`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t0`
) AS `t3`
WHERE
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        `t6`.`row_time`,
        `t6`.`num`,
        `t6`.`id`,
        `t6`.`window_start`,
        `t6`.`window_end`,
        `t6`.`window_time`,
        `t7`.`row_time` AS `row_time_right`,
        `t7`.`id` AS `id_right`,
        `t7`.`window_time` AS `window_time_right`
      FROM (
        SELECT
          `t1`.*
        FROM TABLE(TUMBLE(TABLE `left`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t1`
      ) AS `t6`
      INNER JOIN (
        SELECT
          `t2`.*
        FROM TABLE(TUMBLE(TABLE `right`, DESCRIPTOR(`row_time`), INTERVAL '5' MINUTE(2))) AS `t2`
      ) AS `t7`
        ON `t6`.`num` = `t7`.`num`
        AND `t6`.`window_start` = `t7`.`window_start`
        AND `t6`.`window_end` = `t7`.`window_end`
    ) AS `t8`
    WHERE
      `t3`.`num` = `t8`.`num`
      AND `t3`.`window_start` = `t8`.`window_start`
      AND `t3`.`window_end` = `t8`.`window_end`
  )