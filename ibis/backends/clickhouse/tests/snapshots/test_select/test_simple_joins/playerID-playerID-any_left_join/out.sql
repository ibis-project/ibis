SELECT
  "t2"."playerID",
  "t2"."yearID",
  "t2"."stint",
  "t2"."teamID",
  "t2"."lgID",
  "t2"."G",
  "t2"."AB",
  "t2"."R",
  "t2"."H",
  "t2"."X2B",
  "t2"."X3B",
  "t2"."HR",
  "t2"."RBI",
  "t2"."SB",
  "t2"."CS",
  "t2"."BB",
  "t2"."SO",
  "t2"."IBB",
  "t2"."HBP",
  "t2"."SH",
  "t2"."SF",
  "t2"."GIDP"
FROM "batting" AS "t2"
LEFT ANY JOIN "awards_players" AS "t3"
  ON "t2"."playerID" = "t3"."playerID"