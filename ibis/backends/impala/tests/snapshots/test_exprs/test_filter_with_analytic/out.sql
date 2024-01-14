SELECT
  `t2`.`col`,
  `t2`.`analytic`
FROM (
  SELECT
    `t1`.`col`,
    COUNT(*) OVER (ORDER BY NULL ASC NULLS LAST) AS `analytic`
  FROM (
    SELECT
      `t0`.`col`,
      NULL AS `filter`
    FROM `x` AS `t0`
    WHERE
      NULL IS NULL
  ) AS `t1`
) AS `t2`