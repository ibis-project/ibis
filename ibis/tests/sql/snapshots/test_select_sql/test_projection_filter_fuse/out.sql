SELECT `a`, `b`, `c`
FROM (
  SELECT *
  FROM foo
  WHERE `a` > 0
) t0