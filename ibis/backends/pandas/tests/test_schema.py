import pandas as pd
import pandas.testing as tm

import ibis


# TODO(kszucs): move it to ibis/formats
def test_apply_to_schema_with_timezone():
    data = {'time': pd.date_range('2018-01-01', '2018-01-02', freq='H')}
    df = expected = pd.DataFrame(data).assign(
        time=lambda df: df.time.dt.tz_localize("EST")
    )
    desired_schema = ibis.schema(dict(time='timestamp("EST")'))
    result = desired_schema.apply_to(df.copy())
    tm.assert_frame_equal(expected, result)
