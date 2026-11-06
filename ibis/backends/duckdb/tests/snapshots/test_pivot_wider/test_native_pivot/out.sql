SELECT
  *
FROM "temp"."main"."pivot_fish" AS "t0"
PIVOT(SUM("n") FOR "station" IN ('Lisbon' AS "Lisbon", 'Release' AS "Release")
GROUP BY
  "fish",
  "seen")