SELECT
  CAST("t0"."id" IN (
    SELECT
      arrayJoin("t1"."ids") AS "ids"
    FROM "way_view" AS "t1"
  ) AS Nullable(Bool)) AS "InSubquery(id)"
FROM "node_view" AS "t0"