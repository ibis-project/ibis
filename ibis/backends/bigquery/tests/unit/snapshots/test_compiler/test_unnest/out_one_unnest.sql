SELECT
  t0.`rowindex`,
  IF(pos = pos_2, repeated_struct_col, NULL) AS repeated_struct_col
FROM array_test AS t0, UNNEST(GENERATE_ARRAY(0, GREATEST(ARRAY_LENGTH(t0.`repeated_struct_col`)) - 1)) AS pos
CROSS JOIN UNNEST(t0.`repeated_struct_col`) AS repeated_struct_col WITH OFFSET AS pos_2
WHERE
  pos = pos_2
  OR (
    pos > (
      ARRAY_LENGTH(t0.`repeated_struct_col`) - 1
    )
    AND pos_2 = (
      ARRAY_LENGTH(t0.`repeated_struct_col`) - 1
    )
  )