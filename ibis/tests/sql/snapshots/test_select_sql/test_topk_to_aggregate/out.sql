SELECT t0.*
FROM (
  SELECT t1.`dest`, avg(t1.`arrdelay`) AS `Mean(arrdelay)`
  FROM airlines t1
  GROUP BY 1
) t0
ORDER BY t0.`Mean(arrdelay)` DESC
LIMIT 10