SELECT
  t0.*
FROM UNNEST(ARRAY<STRUCT<`Column One` INT64>>[STRUCT(1 AS `Column One`), STRUCT(2 AS `Column One`), STRUCT(3 AS `Column One`)]) AS t0