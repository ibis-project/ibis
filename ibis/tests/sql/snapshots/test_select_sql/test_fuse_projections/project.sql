SELECT t0.*, t0.`foo` * 2 AS `qux`
FROM (
  SELECT t1.*, t1.`foo` + t1.`bar` AS `baz`
  FROM tbl t1
) t0