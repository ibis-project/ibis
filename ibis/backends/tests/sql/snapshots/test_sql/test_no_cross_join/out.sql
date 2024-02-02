SELECT
  "t3"."id",
  "t3"."personal",
  "t3"."family",
  "t4"."taken",
  "t4"."person",
  "t4"."quant",
  "t4"."reading",
  "t5"."id" AS "id_right",
  "t5"."site",
  "t5"."dated"
FROM "person" AS "t3"
INNER JOIN "survey" AS "t4"
  ON "t3"."id" = "t4"."person"
INNER JOIN "visited" AS "t5"
  ON "t5"."id" = "t4"."taken"