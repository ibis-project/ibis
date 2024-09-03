SELECT
  XOR(XOR("t0"."a", "t0"."b"), "t0"."c") AS "x"
FROM "t" AS "t0" --- op(op(a, b), c);
SELECT
  XOR("t0"."a", (
    XOR("t0"."b", "t0"."c")
  )) AS "x"
FROM "t" AS "t0" --- op(a, op(b, c));
