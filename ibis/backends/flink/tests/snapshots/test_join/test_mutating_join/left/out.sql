WITH t0 AS (
  SELECT t3.*
  FROM awards_players t3
  WHERE t3.`lgID` = 'NL'
),
t1 AS (
  SELECT t0.`playerID`, t0.`awardID`, t0.`tie`, t0.`notes`
  FROM t0
),
t2 AS (
  SELECT t3.*
  FROM batting t3
  WHERE t3.`yearID` = 2015
)
SELECT t2.`playerID`, t2.`yearID`, t2.`stint`, t2.`teamID`, t2.`lgID`, t2.`G`,
       t2.`AB`, t2.`R`, t2.`H`, t2.`X2B`, t2.`X3B`, t2.`HR`, t2.`RBI`,
       t2.`SB`, t2.`CS`, t2.`BB`, t2.`SO`, t2.`IBB`, t2.`HBP`, t2.`SH`,
       t2.`SF`, t2.`GIDP`, t1.`playerID` AS `playerID_right`,
       t1.`awardID`, t1.`tie`, t1.`notes`
FROM t2
  LEFT OUTER JOIN t1
    ON t2.`playerID` = t1.`playerID`