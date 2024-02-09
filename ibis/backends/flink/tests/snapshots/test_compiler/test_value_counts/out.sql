SELECT
  `t1`.`ExtractYear(i)`,
  COUNT(*) AS `ExtractYear(i)_count`
FROM (
  SELECT
    EXTRACT(year FROM `t0`.`i`) AS `ExtractYear(i)`
  FROM `table` AS `t0`
) AS `t1`
GROUP BY
  `t1`.`ExtractYear(i)`