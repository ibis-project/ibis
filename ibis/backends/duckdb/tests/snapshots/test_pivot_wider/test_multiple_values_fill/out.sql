SELECT
  "fish",
  COALESCE("0_seen", 0) AS "seen_Lisbon",
  COALESCE("0_n", 0) AS "n_Lisbon",
  COALESCE("1_seen", 0) AS "seen_Release",
  COALESCE("1_n", 0) AS "n_Release"
FROM (
  SELECT
    *
  FROM "temp"."main"."pivot_fish" AS "t0"
  PIVOT(SUM("seen") AS "seen", SUM("n") AS "n" FOR "station" IN ('Lisbon' AS "0", 'Release' AS "1")
  GROUP BY
    "fish")
)