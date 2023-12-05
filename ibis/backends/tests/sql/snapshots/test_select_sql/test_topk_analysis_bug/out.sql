WITH t0 AS (
  SELECT t3.`dest`, avg(t3.`arrdelay`) AS `Mean(arrdelay)`
  FROM airlines t3
  WHERE t3.`dest` IN ('ORD', 'JFK', 'SFO')
  GROUP BY 1
),
t1 AS (
  SELECT t0.*
  FROM t0
  ORDER BY t0.`Mean(arrdelay)` DESC
  LIMIT 10
),
t2 AS (
  SELECT t3.*
  FROM airlines t3
  WHERE t3.`dest` IN ('ORD', 'JFK', 'SFO')
)
SELECT `origin`, count(1) AS `CountStar()`
FROM t2
  LEFT SEMI JOIN t1
    ON t2.`dest` = t1.`dest`
GROUP BY 1