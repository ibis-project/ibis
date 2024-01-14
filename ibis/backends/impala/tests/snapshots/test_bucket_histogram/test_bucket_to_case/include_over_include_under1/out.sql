SELECT
  CAST(CASE
    WHEN `t0`.`f` < 10
    THEN 0
    WHEN 10 <= `t0`.`f`
    THEN 1
    ELSE CAST(NULL AS TINYINT)
  END AS INT) AS `Cast(Bucket(f), int32)`
FROM `alltypes` AS `t0`