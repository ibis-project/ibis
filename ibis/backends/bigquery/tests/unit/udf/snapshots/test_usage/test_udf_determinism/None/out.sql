CREATE TEMPORARY FUNCTION my_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return s.length;
}
return my_len(s);
""";

SELECT my_len_0('abcd') AS `my_len_0_'abcd'`