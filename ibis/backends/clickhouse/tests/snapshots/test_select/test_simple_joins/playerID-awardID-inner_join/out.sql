SELECT
  "t2"."playerID" AS "playerID",
  "t2"."yearID" AS "yearID",
  "t2"."stint" AS "stint",
  "t2"."teamID" AS "teamID",
  "t2"."lgID" AS "lgID",
  "t2"."G" AS "G",
  "t2"."AB" AS "AB",
  "t2"."R" AS "R",
  "t2"."H" AS "H",
  "t2"."X2B" AS "X2B",
  "t2"."X3B" AS "X3B",
  "t2"."HR" AS "HR",
  "t2"."RBI" AS "RBI",
  "t2"."SB" AS "SB",
  "t2"."CS" AS "CS",
  "t2"."BB" AS "BB",
  "t2"."SO" AS "SO",
  "t2"."IBB" AS "IBB",
  "t2"."HBP" AS "HBP",
  "t2"."SH" AS "SH",
  "t2"."SF" AS "SF",
  "t2"."GIDP" AS "GIDP"
FROM "batting" AS "t2"
INNER JOIN "awards_players" AS "t3"
  ON "t2"."playerID" = "t3"."awardID"