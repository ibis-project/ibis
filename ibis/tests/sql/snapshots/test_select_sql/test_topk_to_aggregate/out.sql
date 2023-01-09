SELECT *
FROM (
  SELECT `dest` AS `dest`, avg(`arrdelay`) AS `Mean(arrdelay)`
  FROM airlines
  GROUP BY 1
) t0
ORDER BY `Mean(arrdelay)` DESC
LIMIT 10