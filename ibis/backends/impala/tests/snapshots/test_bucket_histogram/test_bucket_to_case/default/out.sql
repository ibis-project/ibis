SELECT
  CASE
    WHEN (
      0 <= `t0`.`f`
    ) AND (
      `t0`.`f` < 10
    )
    THEN 0
    WHEN (
      10 <= `t0`.`f`
    ) AND (
      `t0`.`f` < 25
    )
    THEN 1
    WHEN (
      25 <= `t0`.`f`
    ) AND (
      `t0`.`f` <= 50
    )
    THEN 2
    ELSE CAST(NULL AS TINYINT)
  END AS `Bucket(f)`
FROM `alltypes` AS `t0`