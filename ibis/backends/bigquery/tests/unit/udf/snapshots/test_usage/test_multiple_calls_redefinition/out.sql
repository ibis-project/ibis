CREATE TEMPORARY FUNCTION my_len_0(
    s STRING
)
RETURNS FLOAT64
LANGUAGE js AS
'\n\'use strict\';\nfunction my_len(s) {\n    return s.length;\n}\nreturn my_len(s);\n';

CREATE TEMPORARY FUNCTION my_len_1(
    s STRING
)
RETURNS FLOAT64
LANGUAGE js AS
'\n\'use strict\';\nfunction my_len(s) {\n    return (s.length + 1);\n}\nreturn my_len(s);\n';

SELECT
  (
    my_len_0('abcd') + my_len_0('abcd')
  ) + my_len_1('abcd') AS `Add_Add_my_len_0_'abcd'_ my_len_0_'abcd'_ my_len_1_'abcd'`