SELECT
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    t1."playerID" AS "playerID",
    t1."yearID" AS "yearID",
    t1.stint AS stint,
    t1."teamID" AS "teamID",
    t1."lgID" AS "lgID",
    t1."G" AS "G",
    t1."AB" AS "AB",
    t1."R" AS "R",
    t1."H" AS "H",
    t1."X2B" AS "X2B",
    t1."X3B" AS "X3B",
    t1."HR" AS "HR",
    t1."RBI" AS "RBI",
    t1."SB" AS "SB",
    t1."CS" AS "CS",
    t1."BB" AS "BB",
    t1."SO" AS "SO",
    t1."IBB" AS "IBB",
    t1."HBP" AS "HBP",
    t1."SH" AS "SH",
    t1."SF" AS "SF",
    t1."GIDP" AS "GIDP"
  FROM batting AS t1
  ORDER BY
    t1."playerID" DESC
) AS t0