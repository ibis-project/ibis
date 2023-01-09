SELECT `origin` AS `origin`, count(1) AS `count`
FROM (
  SELECT *
  FROM airlines
  WHERE `dest` IN ('ORD', 'JFK', 'SFO')
) t0
  LEFT SEMI JOIN (
    SELECT *
    FROM (
      SELECT `dest` AS `dest`, avg(`arrdelay`) AS `Mean(arrdelay)`
      FROM airlines
      WHERE `dest` IN ('ORD', 'JFK', 'SFO')
      GROUP BY 1
    ) t2
    ORDER BY `Mean(arrdelay)` DESC
    LIMIT 10
  ) t1
    ON t0.`dest` = t1.`dest`
GROUP BY 1