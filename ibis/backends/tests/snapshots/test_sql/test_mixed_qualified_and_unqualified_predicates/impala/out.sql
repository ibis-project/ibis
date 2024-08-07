SELECT
  `x`,
  `y`
FROM (
  SELECT
    `t1`.`x`,
    `t1`.`y`,
    AVG(`t1`.`x`) OVER (ORDER BY NULL ASC) AS _w
  FROM (
    SELECT
      `t0`.`x`,
      SUM(`t0`.`x`) OVER (ORDER BY NULL ASC) AS `y`
    FROM `t` AS `t0`
  ) AS `t1`
  WHERE
    `t1`.`y` <= 37
) AS _t
WHERE
  _w IS NOT NULL