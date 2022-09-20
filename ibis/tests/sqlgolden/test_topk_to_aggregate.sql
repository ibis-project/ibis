SELECT *
FROM (
  SELECT `dest`, avg(`arrdelay`) AS `mean`
  FROM airlines
  GROUP BY 1
) t0
ORDER BY `mean` DESC
LIMIT 10
