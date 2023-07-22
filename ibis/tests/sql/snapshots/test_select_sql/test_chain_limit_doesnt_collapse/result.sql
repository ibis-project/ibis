WITH t0 AS (
  SELECT t2.`city`, count(t2.`city`) AS `Count(city)`
  FROM tbl t2
  GROUP BY 1
)
SELECT t1.*
FROM (
  SELECT t0.*
  FROM t0
  ORDER BY t0.`Count(city)` DESC
  LIMIT 10
) t1
LIMIT 5 OFFSET (SELECT count(1) + -5 FROM (
  SELECT t0.*
  FROM t0
  ORDER BY t0.`Count(city)` DESC
  LIMIT 10
) t1)