SELECT
  IF(pos = pos_2, level_two, NULL) AS level_two
FROM (
  SELECT
    t1.`rowindex`,
    IF(pos = pos_2, level_one, NULL).`nested_struct_col` AS level_one
  FROM array_test AS t1, UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(t1.`repeated_struct_col`)) - 1)) AS pos
  CROSS JOIN UNNEST(t1.`repeated_struct_col`) AS level_one WITH OFFSET AS pos_2
  WHERE
    pos = pos_2
    OR (
      pos > (
        ARRAY_LENGTH(t1.`repeated_struct_col`) - 1
      )
      AND pos_2 = (
        ARRAY_LENGTH(t1.`repeated_struct_col`) - 1
      )
    )
) AS t0, UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(t0.`level_one`)) - 1)) AS pos
CROSS JOIN UNNEST(t0.`level_one`) AS level_two WITH OFFSET AS pos_2
WHERE
  pos = pos_2
  OR (
    pos > (
      ARRAY_LENGTH(t0.`level_one`) - 1
    )
    AND pos_2 = (
      ARRAY_LENGTH(t0.`level_one`) - 1
    )
  )