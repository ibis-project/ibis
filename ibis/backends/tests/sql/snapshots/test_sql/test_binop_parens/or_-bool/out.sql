SELECT
  "t0"."a" OR "t0"."b" OR "t0"."c" AS "x"
FROM "t" AS "t0" --- op(op(a, b), c);
SELECT
  "t0"."a" OR (
    "t0"."b" OR "t0"."c"
  ) AS "x"
FROM "t" AS "t0" --- op(a, op(b, c));
