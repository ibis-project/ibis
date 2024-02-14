SELECT
  `t3`.`a`,
  `t3`.`b`,
  `t3`.`c`,
  `t3`.`d`,
  `t3`.`g`,
  `t3`.`window_start`,
  `t3`.`window_end`,
  `t3`.`rownum`
FROM (
  SELECT
    `t2`.`a`,
    `t2`.`b`,
    `t2`.`c`,
    `t2`.`d`,
    `t2`.`g`,
    `t2`.`window_start`,
    `t2`.`window_end`,
    ROW_NUMBER() OVER (PARTITION BY `t2`.`window_start`, `t2`.`window_end` ORDER BY `t2`.`g` DESC) - 1 AS `rownum`
  FROM (
    SELECT
      `t1`.`a`,
      `t1`.`b`,
      `t1`.`c`,
      `t1`.`d`,
      `t1`.`g`,
      `t1`.`window_start`,
      `t1`.`window_end`
    FROM (
      SELECT
        `t0`.*
      FROM TABLE(TUMBLE(TABLE `table`, DESCRIPTOR(`i`), INTERVAL '600' SECOND(3))) AS `t0`
    ) AS `t1`
  ) AS `t2`
) AS `t3`
WHERE
  `t3`.`rownum` <= 3