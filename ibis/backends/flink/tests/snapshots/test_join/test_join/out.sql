SELECT t0.`playerID`, t0.`yearID`, t0.`stint`, t0.`teamID`, t0.`lgID`, t0.`G`,
       t0.`AB`, t0.`R`, t0.`H`, t0.`X2B`, t0.`X3B`, t0.`HR`, t0.`RBI`,
       t0.`SB`, t0.`CS`, t0.`BB`, t0.`SO`, t0.`IBB`, t0.`HBP`, t0.`SH`,
       t0.`SF`, t0.`GIDP`, t1.`awardID`, t1.`yearID` AS `yearID_right`,
       t1.`lgID` AS `lgID_right`, t1.`tie`, t1.`notes`
FROM batting t0
  INNER JOIN awards_players t1
    ON t0.`playerID` = t1.`playerID`