SELECT
  ARRAY_CONCAT_AGG(IF(`t0`.`keep`, `t0`.`arr`, NULL) ORDER BY `t0`.`key` DESC
  LIMIT 2) AS `result`
FROM `t` AS `t0`