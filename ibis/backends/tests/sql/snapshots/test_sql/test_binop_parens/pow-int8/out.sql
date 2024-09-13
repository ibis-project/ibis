SELECT
  POWER((
    POWER("t0"."a", "t0"."b")
  ), "t0"."c") AS "x"
FROM "t" AS "t0" --- op(op(a, b), c);
SELECT
  POWER("t0"."a", (
    POWER("t0"."b", "t0"."c")
  )) AS "x"
FROM "t" AS "t0" --- op(a, op(b, c));
