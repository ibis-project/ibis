SELECT t0.*
FROM (
  SELECT t1.`string_col`, count(1) AS `nrows`
  FROM functional_alltypes t1
  GROUP BY 1
  LIMIT 5
) t0
ORDER BY t0.`string_col` ASC