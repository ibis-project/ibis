SELECT *
FROM (
  SELECT `dest`, avg(`arrdelay`)
  FROM airlines
  GROUP BY 1
) t0
ORDER BY `Mean(arrdelay)` DESC
LIMIT 10