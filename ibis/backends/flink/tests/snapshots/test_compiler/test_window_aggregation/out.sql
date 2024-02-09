SELECT
  `t1`.`window_start`,
  `t1`.`window_end`,
  `t1`.`g`,
  AVG(`t1`.`d`) AS `mean`
FROM (
  SELECT
    `t0`.*
  FROM TABLE(TUMBLE(TABLE `table`, DESCRIPTOR(`i`), INTERVAL '15' MINUTE)) AS `t0`
) AS `t1`
GROUP BY
  `t1`.`window_start`,
  `t1`.`window_end`,
  `t1`.`g`