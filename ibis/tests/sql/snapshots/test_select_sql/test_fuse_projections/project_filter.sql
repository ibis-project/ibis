SELECT *, `foo` * 2 AS `qux`
FROM (
  SELECT *, `foo` + `bar` AS `baz`
  FROM tbl
  WHERE `value` > 0
) t0