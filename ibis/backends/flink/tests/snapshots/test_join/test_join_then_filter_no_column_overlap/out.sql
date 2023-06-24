WITH t0 AS (
  SELECT t3.*
  FROM batting t3
  WHERE t3.`yearID` = 2015
),
t1 AS (
  SELECT t3.*
  FROM awards_players t3
  WHERE t3.`lgID` = 'NL'
),
t2 AS (
  SELECT t0.`yearID` AS `year`, t0.`RBI`
  FROM t0
)
SELECT *
FROM t2
  INNER JOIN t1
    ON t2.`year` = t1.`yearID`
WHERE `RBI` = 9