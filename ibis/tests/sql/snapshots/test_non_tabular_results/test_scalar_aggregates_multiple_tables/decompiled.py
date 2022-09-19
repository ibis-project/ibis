import ibis


tbl = ibis.table(name="tbl", schema={"flag": "string", "value": "float64"})

result = (
    (
        tbl.filter(tbl.flag == "1").value.mean().name("mean")
        / tbl.filter(tbl.flag == "1").value.sum().name("sum")
    ).name("fv")
    - (
        tbl.filter(tbl.flag == "0").value.mean().name("mean")
        / tbl.filter(tbl.flag == "0").value.sum().name("sum")
    ).name("uv")
).name("tmp")
