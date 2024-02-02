SELECT
  "t0"."tinyint_col" + -(
    "t0"."int_col" + "t0"."double_col"
  ) AS "Add(tinyint_col, Negate(Add(int_col, double_col)))"
FROM "functional_alltypes" AS "t0"