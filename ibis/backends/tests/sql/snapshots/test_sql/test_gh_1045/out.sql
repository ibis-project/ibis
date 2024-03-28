SELECT
  "t5"."t1_id1",
  "t5"."t1_val1",
  "t10"."id3",
  "t10"."val2",
  "t10"."dt",
  "t10"."t3_val2",
  "t10"."id2a",
  "t10"."id2b",
  "t10"."val2_right"
FROM (
  SELECT
    "t0"."id1" AS "t1_id1",
    "t0"."val1" AS "t1_val1"
  FROM "test1" AS "t0"
) AS "t5"
LEFT OUTER JOIN (
  SELECT
    "t8"."id3",
    "t8"."val2",
    "t8"."dt",
    "t8"."t3_val2",
    "t3"."id2a",
    "t3"."id2b",
    "t3"."val2" AS "val2_right"
  FROM (
    SELECT
      "t6"."id3",
      "t6"."val2",
      "t6"."dt",
      "t6"."id3" AS "t3_val2"
    FROM (
      SELECT
        CAST("t2"."id3" AS BIGINT) AS "id3",
        "t2"."val2",
        "t2"."dt"
      FROM "test3" AS "t2"
    ) AS "t6"
  ) AS "t8"
  INNER JOIN "test2" AS "t3"
    ON "t3"."id2b" = "t8"."id3"
) AS "t10"
  ON "t5"."t1_id1" = "t10"."id2a"