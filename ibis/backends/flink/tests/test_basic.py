import pandas as pd
import pandas.testing as tm


def test_string_split(con):
    ft = con.tables.functional_alltypes[:1]
    ft = ft.mutate(s="a,b,c")
    ft = ft.mutate(split_s=ft.s.split(","))[["split_s"]]
    result = ft.execute()
    expected = pd.DataFrame({
        "split_s": ["a", "b", "c"]
    })
    tm.assert_frame_equal(result, expected)
