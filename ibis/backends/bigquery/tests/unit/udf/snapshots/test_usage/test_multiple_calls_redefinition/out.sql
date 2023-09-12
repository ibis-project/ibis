CREATE TEMPORARY FUNCTION my_len_0(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return s.length;
}
return my_len(s);
""";

CREATE TEMPORARY FUNCTION my_len_1(s STRING)
RETURNS FLOAT64
LANGUAGE js AS """
'use strict';
function my_len(s) {
    return (s.length + 1);
}
return my_len(s);
""";

SELECT (my_len_0('abcd') + my_len_0('abcd')) + my_len_1('abcd') AS `Add_Add_my_len_0_'abcd'_ my_len_0_'abcd'_ my_len_1_'abcd'`