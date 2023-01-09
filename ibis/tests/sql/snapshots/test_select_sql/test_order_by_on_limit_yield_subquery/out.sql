SELECT *
FROM (
  SELECT `string_col` AS `string_col`, count(1) AS `nrows`
  FROM functional_alltypes
  GROUP BY 1
  LIMIT 5
) t0
ORDER BY `string_col` ASC