WITH t0 AS (
  SELECT t4.*, t4.`yearID` AS `year`
  FROM batting t4
),
t1 AS (
  SELECT t0.*
  FROM t0
  WHERE t0.`year` = 2015
),
t2 AS (
  SELECT t1.`year`, t1.`RBI`
  FROM t1
)
SELECT *
FROM t2
  INNER JOIN awards_players t3
    ON t2.`year` = t3.`yearID`
LIMIT 5