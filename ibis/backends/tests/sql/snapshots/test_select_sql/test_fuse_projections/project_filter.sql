SELECT *, `foo` * 2 AS `qux`
FROM (
  SELECT t1.*, t1.`foo` + t1.`bar` AS `baz`
  FROM tbl t1
  WHERE t1.`value` > 0
) t0