SELECT t0.`ExtractYear(i)`, COUNT(*) AS `ExtractYear(i)_count`
FROM (
  SELECT EXTRACT(year from t1.`i`) AS `ExtractYear(i)`
  FROM table t1
) t0
GROUP BY t0.`ExtractYear(i)`