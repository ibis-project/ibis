SELECT
  CASE
    WHEN `t0`.`f` <= 10
    THEN 0
    WHEN 10 < `t0`.`f`
    THEN 1
    ELSE CAST(NULL AS TINYINT)
  END AS `Bucket(f)`
FROM `alltypes` AS `t0`