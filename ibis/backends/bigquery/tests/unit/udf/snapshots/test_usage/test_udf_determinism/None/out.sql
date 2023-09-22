CREATE TEMPORARY FUNCTION my_len_0(
    s STRING
)
RETURNS FLOAT64
LANGUAGE js AS
'\n\'use strict\';\nfunction my_len(s) {\n    return s.length;\n}\nreturn my_len(s);\n';

SELECT
  my_len_0('abcd') AS `my_len_0_'abcd'`