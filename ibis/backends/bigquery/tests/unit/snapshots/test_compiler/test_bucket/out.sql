SELECT
  CASE
    WHEN (
      0 <= `t0`.`value`
    ) AND (
      `t0`.`value` < 1
    )
    THEN 0
    WHEN (
      1 <= `t0`.`value`
    ) AND (
      `t0`.`value` <= 3
    )
    THEN 1
    ELSE NULL
  END AS `tmp`
FROM `t` AS `t0`