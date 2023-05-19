import pandas as pd
import pytest

import ibis

dd = pytest.importorskip("dask.dataframe")

from dask.dataframe.utils import tm  # noqa: E402


# TODO(kszucs): move it to ibis/formats
def test_apply_to_schema_with_timezone(npartitions):
    data = {'time': pd.date_range('2018-01-01', '2018-01-02', freq='H')}
    df = dd.from_pandas(pd.DataFrame(data), npartitions=npartitions)
    expected = df.assign(time=df.time.dt.tz_localize("EST"))
    desired_schema = ibis.schema([('time', 'timestamp("EST")')])
    result = desired_schema.apply_to(df.copy())
    tm.assert_frame_equal(result.compute(), expected.compute())
