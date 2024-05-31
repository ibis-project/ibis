SELECT
  reinterpretAsInt64(sipHash64("t0"."string_col")) AS "Hash(string_col)"
FROM "functional_alltypes" AS "t0"