SELECT CAST(`g` AS double) AS `Cast(g, float64)`
FROM (
  SELECT `g`, count(1) AS `count`
  FROM alltypes
  GROUP BY 1
) t0