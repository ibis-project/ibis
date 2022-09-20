SELECT *
FROM (
  SELECT *
  FROM star1
  LIMIT 10
) t0
WHERE `f` > 0
