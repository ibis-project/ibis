from __future__ import annotations

import calendar as cal
from operator import itemgetter

import pytest

from ibis import _, coalesce, cumulative_window, date, ifelse, null, rank, union
from ibis import literal as lit
from ibis import selectors as s
from ibis.backends.tests.errors import (
    ArrowNotImplementedError,
    ClickHouseDatabaseError,
    TrinoUserError,
)
from ibis.backends.tests.tpc.conftest import tpc_test


@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
@tpc_test("ds")
def test_01(store_returns, date_dim, store, customer):
    customer_total_return = (
        store_returns.join(
            date_dim.filter(_.d_year == 2000), [("sr_returned_date_sk", "d_date_sk")]
        )
        .group_by(ctr_customer_sk=_.sr_customer_sk, ctr_store_sk=_.sr_store_sk)
        .agg(ctr_total_return=_.sr_return_amt.sum())
    )
    ctr2 = customer_total_return.view()
    return (
        customer_total_return.join(
            store.filter(_.s_state == "TN"), [("ctr_store_sk", "s_store_sk")]
        )
        .join(customer, _.ctr_customer_sk == customer.c_customer_sk)
        .filter(
            lambda t: t.ctr_total_return
            > ctr2.filter(t.ctr_store_sk == ctr2.ctr_store_sk)
            .ctr_total_return.mean()
            .as_scalar()
            * 1.2
        )
        .select(_.c_customer_id)
        .order_by(_.c_customer_id)
        .limit(100)
    )


@tpc_test("ds")
def test_02(web_sales, catalog_sales, date_dim):
    wscs = web_sales.select(
        sold_date_sk=_.ws_sold_date_sk, sales_price=_.ws_ext_sales_price
    ).union(
        catalog_sales.select(
            sold_date_sk=_.cs_sold_date_sk, sales_price=_.cs_ext_sales_price
        )
    )

    # start on Sunday
    days = [(i, cal.day_abbr[i].lower(), cal.day_name[i]) for i in range(-1, 6)]

    wswscs = (
        wscs.join(date_dim, _.sold_date_sk == date_dim.d_date_sk)
        .group_by(_.d_week_seq)
        .agg(
            **{
                f"{day_abbr}_sales": _.sales_price.sum(where=_.d_day_name == day_name)
                for _i, day_abbr, day_name in days
            }
        )
    )

    y = wswscs.join(date_dim.filter(_.d_year == 2001), "d_week_seq").select(
        d_week_seq1=wswscs.d_week_seq,
        **{c + "1": c for c in wswscs.columns if c.endswith("_sales")},
    )
    z = wswscs.join(date_dim.filter(_.d_year == 2001 + 1), "d_week_seq").select(
        d_week_seq2=wswscs.d_week_seq,
        **{c + "2": c for c in wswscs.columns if c.endswith("_sales")},
    )
    return (
        y.join(z, y.d_week_seq1 == z.d_week_seq2 - 53)
        .select(
            _.d_week_seq1,
            **{
                f"r{i + 2}": (_[f"{day_abbr}_sales1"] / _[f"{day_abbr}_sales2"]).round(
                    2
                )
                for i, day_abbr, _n in days
            },
        )
        .order_by(_.d_week_seq1)
    )


@tpc_test("ds")
def test_03(date_dim, store_sales, item):
    return (
        date_dim.join(store_sales, date_dim.d_date_sk == store_sales.ss_sold_date_sk)
        .join(item, store_sales.ss_item_sk == item.i_item_sk)
        .filter(_.i_manufact_id == 128, _.d_moy == 11)
        .group_by(_.d_year, brand_id=_.i_brand_id, brand=_.i_brand)
        .agg(sum_agg=_.ss_ext_sales_price.sum())
        .order_by(_.d_year, _.sum_agg.desc(), _.brand_id)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notimpl(
    ["datafusion"], reason="Optimizer rule 'common_sub_expression_eliminate' failed"
)
def test_04(customer, store_sales, catalog_sales, web_sales, date_dim):
    def profile(sales, *, name):
        char = name[0]
        prefix = {"w": "ws_bill", "c": "cs_bill", "s": "ss"}[char]
        return (
            customer.join(
                sales, customer.c_customer_sk == sales[f"{prefix}_customer_sk"]
            )
            .join(date_dim, sales[f"{char}s_sold_date_sk"] == date_dim.d_date_sk)
            .group_by(
                customer_id=_.c_customer_id,
                customer_first_name=_.c_first_name,
                customer_last_name=_.c_last_name,
                customer_preferred_cust_flag=_.c_preferred_cust_flag,
                customer_birth_country=_.c_birth_country,
                customer_login=_.c_login,
                customer_email_address=_.c_email_address,
                dyear=_.d_year,
            )
            .agg(
                year_total=(
                    (
                        (
                            _[f"{char}s_ext_list_price"]
                            - _[f"{char}s_ext_wholesale_cost"]
                            - _[f"{char}s_ext_discount_amt"]
                        )
                        + _[f"{char}s_ext_sales_price"]
                    )
                    / 2
                ).sum(),
                sale_type=lit(char),
            )
        )

    year_total = (
        profile(store_sales, name="store_sales")
        .union(profile(catalog_sales, name="catalog_sales"))
        .union(profile(web_sales, name="web_sales"))
    )

    t_s_firstyear = year_total.filter(
        _.sale_type == "s", _.dyear == 2001, _.year_total > 0
    )
    t_s_secyear = year_total.view().filter(_.sale_type == "s", _.dyear == 2001 + 1)
    t_c_firstyear = year_total.view().filter(
        _.sale_type == "c", _.dyear == 2001, _.year_total > 0
    )
    t_c_secyear = year_total.view().filter(_.sale_type == "c", _.dyear == 2001 + 1)
    t_w_firstyear = year_total.view().filter(
        _.sale_type == "w", _.dyear == 2001, _.year_total > 0
    )
    t_w_secyear = year_total.view().filter(_.sale_type == "w", _.dyear == 2001 + 1)
    return (
        t_s_firstyear.join(t_s_secyear, "customer_id")
        .join(t_c_secyear, t_s_firstyear.customer_id == t_c_secyear.customer_id)
        .join(
            t_c_firstyear,
            [
                t_s_firstyear.customer_id == t_c_firstyear.customer_id,
                ifelse(
                    t_c_firstyear.year_total > 0,
                    t_c_secyear.year_total / t_c_firstyear.year_total,
                    null(),
                )
                > ifelse(
                    t_s_firstyear.year_total > 0,
                    t_s_secyear.year_total / t_s_firstyear.year_total,
                    null(),
                ),
            ],
        )
        .join(t_w_firstyear, t_s_firstyear.customer_id == t_w_firstyear.customer_id)
        .join(
            t_w_secyear,
            [
                t_s_firstyear.customer_id == t_w_secyear.customer_id,
                ifelse(
                    t_c_firstyear.year_total > 0,
                    t_c_secyear.year_total / t_c_firstyear.year_total,
                    null(),
                )
                > ifelse(
                    t_w_firstyear.year_total > 0,
                    t_w_secyear.year_total / t_w_firstyear.year_total,
                    null(),
                ),
            ],
        )
        .select(
            t_s_secyear.customer_id,
            t_s_secyear.customer_first_name,
            t_s_secyear.customer_last_name,
            t_s_secyear.customer_preferred_cust_flag,
        )
        .order_by(s.all())
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.xfail(raises=NotImplementedError, reason="requires rollup")
def test_05(
    store_sales,
    store_returns,
    store,
    catalog_sales,
    catalog_returns,
    catalog_page,
    web_sales,
    web_returns,
    web_site,
    date_dim,
):
    raise NotImplementedError()


@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
@tpc_test("ds")
def test_06(customer_address, customer, store_sales, date_dim, item):
    return (
        customer_address.join(
            customer, customer_address.ca_address_sk == customer.c_current_addr_sk
        )
        .join(store_sales, customer.c_customer_sk == store_sales.ss_customer_sk)
        .join(date_dim, store_sales.ss_sold_date_sk == date_dim.d_date_sk)
        .join(item, store_sales.ss_item_sk == item.i_item_sk)
        .filter(
            date_dim.d_month_seq
            == (
                date_dim.filter(_.d_year == 2001, _.d_moy == 1)
                .select(_.d_month_seq)
                .distinct()
                .as_scalar()
            ),
            lambda i: (
                i.i_current_price
                > 1.2
                * item.view()
                .filter(lambda j: j.i_category == i.i_category)
                .agg(lambda j: j.i_current_price.mean())
                .as_scalar()
            ),
        )
        .group_by(state=_.ca_state)
        .having(_.count() >= 10)
        .agg(cnt=_.count())
        .order_by(_.cnt.asc(nulls_first=True), _.state.asc(nulls_first=True))
        .limit(100)
    )


@tpc_test("ds")
def test_07(store_sales, customer_demographics, date_dim, item, promotion):
    return (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(customer_demographics, [("ss_cdemo_sk", "cd_demo_sk")])
        .join(promotion, [("ss_promo_sk", "p_promo_sk")])
        .filter(
            _.cd_gender == "M",
            _.cd_marital_status == "S",
            _.cd_education_status == "College",
            ((_.p_channel_email == "N") | (_.p_channel_event == "N")),
            _.d_year == 2000,
        )
        .group_by(_.i_item_id)
        .agg(
            agg1=_.ss_quantity.mean(),
            agg2=_.ss_list_price.mean(),
            agg3=_.ss_coupon_amt.mean(),
            agg4=_.ss_sales_price.mean(),
        )
        .order_by(_.i_item_id)
        .limit(100)
    )


@tpc_test("ds")
def test_08(store_sales, date_dim, store, customer_address, customer):
    zip_codes = (
        "24128",
        "76232",
        "65084",
        "87816",
        "83926",
        "77556",
        "20548",
        "26231",
        "43848",
        "15126",
        "91137",
        "61265",
        "98294",
        "25782",
        "17920",
        "18426",
        "98235",
        "40081",
        "84093",
        "28577",
        "55565",
        "17183",
        "54601",
        "67897",
        "22752",
        "86284",
        "18376",
        "38607",
        "45200",
        "21756",
        "29741",
        "96765",
        "23932",
        "89360",
        "29839",
        "25989",
        "28898",
        "91068",
        "72550",
        "10390",
        "18845",
        "47770",
        "82636",
        "41367",
        "76638",
        "86198",
        "81312",
        "37126",
        "39192",
        "88424",
        "72175",
        "81426",
        "53672",
        "10445",
        "42666",
        "66864",
        "66708",
        "41248",
        "48583",
        "82276",
        "18842",
        "78890",
        "49448",
        "14089",
        "38122",
        "34425",
        "79077",
        "19849",
        "43285",
        "39861",
        "66162",
        "77610",
        "13695",
        "99543",
        "83444",
        "83041",
        "12305",
        "57665",
        "68341",
        "25003",
        "57834",
        "62878",
        "49130",
        "81096",
        "18840",
        "27700",
        "23470",
        "50412",
        "21195",
        "16021",
        "76107",
        "71954",
        "68309",
        "18119",
        "98359",
        "64544",
        "10336",
        "86379",
        "27068",
        "39736",
        "98569",
        "28915",
        "24206",
        "56529",
        "57647",
        "54917",
        "42961",
        "91110",
        "63981",
        "14922",
        "36420",
        "23006",
        "67467",
        "32754",
        "30903",
        "20260",
        "31671",
        "51798",
        "72325",
        "85816",
        "68621",
        "13955",
        "36446",
        "41766",
        "68806",
        "16725",
        "15146",
        "22744",
        "35850",
        "88086",
        "51649",
        "18270",
        "52867",
        "39972",
        "96976",
        "63792",
        "11376",
        "94898",
        "13595",
        "10516",
        "90225",
        "58943",
        "39371",
        "94945",
        "28587",
        "96576",
        "57855",
        "28488",
        "26105",
        "83933",
        "25858",
        "34322",
        "44438",
        "73171",
        "30122",
        "34102",
        "22685",
        "71256",
        "78451",
        "54364",
        "13354",
        "45375",
        "40558",
        "56458",
        "28286",
        "45266",
        "47305",
        "69399",
        "83921",
        "26233",
        "11101",
        "15371",
        "69913",
        "35942",
        "15882",
        "25631",
        "24610",
        "44165",
        "99076",
        "33786",
        "70738",
        "26653",
        "14328",
        "72305",
        "62496",
        "22152",
        "10144",
        "64147",
        "48425",
        "14663",
        "21076",
        "18799",
        "30450",
        "63089",
        "81019",
        "68893",
        "24996",
        "51200",
        "51211",
        "45692",
        "92712",
        "70466",
        "79994",
        "22437",
        "25280",
        "38935",
        "71791",
        "73134",
        "56571",
        "14060",
        "19505",
        "72425",
        "56575",
        "74351",
        "68786",
        "51650",
        "20004",
        "18383",
        "76614",
        "11634",
        "18906",
        "15765",
        "41368",
        "73241",
        "76698",
        "78567",
        "97189",
        "28545",
        "76231",
        "75691",
        "22246",
        "51061",
        "90578",
        "56691",
        "68014",
        "51103",
        "94167",
        "57047",
        "14867",
        "73520",
        "15734",
        "63435",
        "25733",
        "35474",
        "24676",
        "94627",
        "53535",
        "17879",
        "15559",
        "53268",
        "59166",
        "11928",
        "59402",
        "33282",
        "45721",
        "43933",
        "68101",
        "33515",
        "36634",
        "71286",
        "19736",
        "58058",
        "55253",
        "67473",
        "41918",
        "19515",
        "36495",
        "19430",
        "22351",
        "77191",
        "91393",
        "49156",
        "50298",
        "87501",
        "18652",
        "53179",
        "18767",
        "63193",
        "23968",
        "65164",
        "68880",
        "21286",
        "72823",
        "58470",
        "67301",
        "13394",
        "31016",
        "70372",
        "67030",
        "40604",
        "24317",
        "45748",
        "39127",
        "26065",
        "77721",
        "31029",
        "31880",
        "60576",
        "24671",
        "45549",
        "13376",
        "50016",
        "33123",
        "19769",
        "22927",
        "97789",
        "46081",
        "72151",
        "15723",
        "46136",
        "51949",
        "68100",
        "96888",
        "64528",
        "14171",
        "79777",
        "28709",
        "11489",
        "25103",
        "32213",
        "78668",
        "22245",
        "15798",
        "27156",
        "37930",
        "62971",
        "21337",
        "51622",
        "67853",
        "10567",
        "38415",
        "15455",
        "58263",
        "42029",
        "60279",
        "37125",
        "56240",
        "88190",
        "50308",
        "26859",
        "64457",
        "89091",
        "82136",
        "62377",
        "36233",
        "63837",
        "58078",
        "17043",
        "30010",
        "60099",
        "28810",
        "98025",
        "29178",
        "87343",
        "73273",
        "30469",
        "64034",
        "39516",
        "86057",
        "21309",
        "90257",
        "67875",
        "40162",
        "11356",
        "73650",
        "61810",
        "72013",
        "30431",
        "22461",
        "19512",
        "13375",
        "55307",
        "30625",
        "83849",
        "68908",
        "26689",
        "96451",
        "38193",
        "46820",
        "88885",
        "84935",
        "69035",
        "83144",
        "47537",
        "56616",
        "94983",
        "48033",
        "69952",
        "25486",
        "61547",
        "27385",
        "61860",
        "58048",
        "56910",
        "16807",
        "17871",
        "35258",
        "31387",
        "35458",
        "35576",
    )

    v1 = (
        customer_address.select(ca_zip=_.ca_zip[:5])
        .filter(_.ca_zip.isin(zip_codes))
        .intersect(
            customer_address.join(
                customer.filter(_.c_preferred_cust_flag == "Y"),
                [("ca_address_sk", "c_current_addr_sk")],
            )
            .group_by(_.ca_zip)
            .having(_.count() > 10)
            .agg()
            .select(ca_zip=_.ca_zip[:5])
        )
    )
    return (
        store_sales.join(
            date_dim.filter(_.d_qoy == 2, _.d_year == 1998),
            [("ss_sold_date_sk", "d_date_sk")],
        )
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(v1, store.s_zip[:2] == v1.ca_zip[:2])
        .group_by(_.s_store_name)
        .agg(net_profit=_.ss_net_profit.sum())
        .order_by(_.s_store_name)
    )


@tpc_test("ds")
def test_09(store_sales, reason):
    return reason.filter(_.r_reason_sk == 1).select(
        **{
            f"bucket{b:d}": ifelse(
                store_sales.filter(_.ss_quantity.between(lower, upper)).count() > value,
                store_sales.filter(
                    _.ss_quantity.between(lower, upper)
                ).ss_ext_discount_amt.mean(),
                store_sales.filter(
                    _.ss_quantity.between(lower, upper)
                ).ss_net_paid.mean(),
            )
            for b, (lower, upper, value) in enumerate(
                (
                    (1, 20, 74129),
                    (21, 40, 122840),
                    (41, 60, 56580),
                    (61, 80, 10097),
                    (81, 100, 165306),
                ),
                start=1,
            )
        }
    )


@tpc_test("ds")
@pytest.mark.notimpl(
    ["datafusion"], reason="Exception: Optimizer rule 'scalar_subquery_to_join' failed"
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
def test_10(
    customer,
    customer_address,
    customer_demographics,
    store_sales,
    date_dim,
    web_sales,
    catalog_sales,
):
    return (
        customer.join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(customer_demographics, [("c_current_cdemo_sk", "cd_demo_sk")])
        .filter(
            _.ca_county.isin(
                [
                    "Rush County",
                    "Toole County",
                    "Jefferson County",
                    "Dona Ana County",
                    "La Porte County",
                ]
            ),
            lambda t: (
                store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
                .filter(
                    t.c_customer_sk == store_sales.ss_customer_sk,
                    _.d_year == 2002,
                    _.d_moy.between(1, 1 + 3),
                )
                .count()
                > 0
            ),
            lambda t: (
                web_sales.join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
                .filter(
                    t.c_customer_sk == web_sales.ws_bill_customer_sk,
                    _.d_year == 2002,
                    _.d_moy.between(1, 1 + 3),
                )
                .count()
                > 0
            )
            | (
                catalog_sales.join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
                .filter(
                    t.c_customer_sk == catalog_sales.cs_ship_customer_sk,
                    _.d_year == 2002,
                    _.d_moy.between(1, 1 + 3),
                )
                .count()
                > 0
            ),
        )
        .group_by(
            "cd_gender",
            "cd_marital_status",
            "cd_education_status",
            "cd_purchase_estimate",
            "cd_credit_rating",
            "cd_dep_count",
            "cd_dep_employed_count",
            "cd_dep_college_count",
        )
        .agg({f"cnt{i:d}": _.count() for i in range(1, 7)})
        .select(
            "cd_gender",
            "cd_marital_status",
            "cd_education_status",
            "cnt1",
            "cd_purchase_estimate",
            "cnt2",
            "cd_credit_rating",
            "cnt3",
            "cd_dep_count",
            "cnt4",
            "cd_dep_employed_count",
            "cnt5",
            "cd_dep_college_count",
            "cnt6",
        )
        .order_by(~s.startswith("cnt"))
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["datafusion"],
    reason="Exception: Optimizer rule 'common_sub_expression_eliminate' failed",
)
def test_11(customer, store_sales, web_sales, date_dim):
    def agg(*, sale_type: str, table, join_key):
        prefix = f"{sale_type}s"
        return (
            customer.join(table, [("c_customer_sk", join_key)])
            .join(date_dim, [(f"{prefix}_sold_date_sk", "d_date_sk")])
            .group_by(
                customer_id=_.c_customer_id,
                customer_first_name=_.c_first_name,
                customer_last_name=_.c_last_name,
                customer_preferred_cust_flag=_.c_preferred_cust_flag,
                customer_birth_country=_.c_birth_country,
                customer_login=_.c_login,
                customer_email_address=_.c_email_address,
                dyear=_.d_year,
            )
            .agg(
                year_total=(
                    _[f"{prefix}_ext_list_price"] - _[f"{prefix}_ext_discount_amt"]
                ).sum(),
                sale_type=lit(sale_type),
            )
        )

    year_total = agg(sale_type="s", table=store_sales, join_key="ss_customer_sk").union(
        agg(sale_type="w", table=web_sales, join_key="ws_bill_customer_sk")
    )

    t_s_firstyear = year_total.view()
    t_s_secyear = year_total.view()
    t_w_firstyear = year_total.view()
    t_w_secyear = year_total.view()
    return (
        t_s_firstyear.filter(_.sale_type == "s", _.dyear == 2001, _.year_total > 0)
        .join(
            t_s_secyear.filter(_.sale_type == "s", _.dyear == 2001 + 1), "customer_id"
        )
        .join(t_w_firstyear.filter(_.sale_type == "w", _.dyear == 2001), "customer_id")
        .join(
            t_w_secyear.filter(
                _.sale_type == "w", _.dyear == 2001 + 1, _.year_total > 0
            ),
            "customer_id",
        )
        .select(
            t_s_secyear.customer_id,
            t_s_secyear.customer_first_name,
            t_s_secyear.customer_last_name,
            t_s_secyear.customer_preferred_cust_flag,
            w_first_year_total=t_w_firstyear.year_total,
            w_sec_year_total=t_w_secyear.year_total,
            s_first_year_total=t_s_firstyear.year_total,
            s_sec_year_total=t_s_secyear.year_total,
        )
        .filter(
            ifelse(
                _.w_first_year_total > 0,
                (_.w_sec_year_total * 1.0000) / _.w_first_year_total,
                0.0,
            )
            > ifelse(
                _.s_first_year_total > 0,
                (_.s_sec_year_total * 1.0000) / _.s_first_year_total,
                0.0,
            )
        )
        .drop(s.endswith("_year_total"))
        .order_by(s.across(s.all(), _.asc(nulls_first=True)))
        .limit(100)
    )


@tpc_test("ds")
def test_12(web_sales, item, date_dim):
    return (
        web_sales.join(item, [("ws_item_sk", "i_item_sk")])
        .join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .filter(
            _.i_category.isin(("Sports", "Books", "Home")),
            _.d_date.between(date("1999-02-22"), date("1999-03-24")),
        )
        .group_by(
            _.i_item_id, _.i_item_desc, _.i_category, _.i_class, _.i_current_price
        )
        .agg(itemrevenue=_.ws_ext_sales_price.sum())
        .mutate(
            revenueratio=_.itemrevenue
            * 100.000
            # snowflake divide by zero is an error, all others return a value (null, nan, or inf)
            / _.itemrevenue.sum().over(group_by=_.i_class).nullif(0.0)
        )
        .order_by(_.i_category, _.i_class, _.i_item_id, _.i_item_desc, "revenueratio")
        .limit(100)
    )


@tpc_test("ds")
def test_13(
    store_sales,
    store,
    customer_demographics,
    household_demographics,
    customer_address,
    date_dim,
):
    return (
        store_sales.join(store, [("ss_store_sk", "s_store_sk")])
        .join(household_demographics, [("ss_hdemo_sk", "hd_demo_sk")])
        .join(customer_demographics, [("ss_cdemo_sk", "cd_demo_sk")])
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .filter(
            _.ca_country == "United States",
            _.d_year == 2001,
            (
                (_.cd_marital_status == "M")
                & (_.cd_education_status == "Advanced Degree")
                & _.ss_sales_price.between(100.00, 150.00)
                & (_.hd_dep_count == 3)
            )
            | (
                (_.cd_marital_status == "S")
                & (_.cd_education_status == "College")
                & _.ss_sales_price.between(50.00, 100.00)
                & (_.hd_dep_count == 1)
            )
            | (
                (_.cd_marital_status == "W")
                & (_.cd_education_status == "2 yr Degree")
                & _.ss_sales_price.between(150.00, 200.00)
                & (_.hd_dep_count == 1)
            ),
            (_.ca_state.isin(("TX", "OH", "TX")) & _.ss_net_profit.between(100, 200))
            | (_.ca_state.isin(("OR", "NM", "KY")) & _.ss_net_profit.between(150, 300))
            | (_.ca_state.isin(("VA", "TX", "MS")) & _.ss_net_profit.between(50, 250)),
        )
        .agg(
            avg1=_.ss_quantity.mean(),
            avg2=_.ss_ext_sales_price.mean(),
            avg3=_.ss_ext_wholesale_cost.mean(),
            sum1=_.ss_ext_wholesale_cost.sum(),
        )
    )


@tpc_test("ds")
@pytest.mark.xfail(raises=NotImplementedError, reason="requires rollup")
def test_14(item, store_sales, date_dim, catalog_sales, web_sales):
    raise NotImplementedError()


@tpc_test("ds")
def test_15(catalog_sales, customer, customer_address, date_dim):
    return (
        catalog_sales.join(customer, [("cs_bill_customer_sk", "c_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .filter(
            _.ca_zip[:5].isin(
                (
                    "85669",
                    "86197",
                    "88274",
                    "83405",
                    "86475",
                    "85392",
                    "85460",
                    "80348",
                    "81792",
                )
            )
            | _.ca_state.isin(("CA", "WA", "GA"))
            | (_.cs_sales_price > 500),
            _.d_qoy == 2,
            _.d_year == 2001,
        )
        .group_by(_.ca_zip)
        .agg(total_sales_price=_.cs_sales_price.sum())
        .order_by(_.ca_zip.asc(nulls_first=True))
        .limit(100)
    )


@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
@pytest.mark.notyet(
    ["datafusion"],
    reason="Error during planning: Correlated column is not allowed in predicate",
)
@tpc_test("ds")
def test_16(catalog_sales, date_dim, customer_address, call_center, catalog_returns):
    return (
        catalog_sales.join(date_dim, [("cs_ship_date_sk", "d_date_sk")])
        .join(customer_address, [("cs_ship_addr_sk", "ca_address_sk")])
        .join(call_center, [("cs_call_center_sk", "cc_call_center_sk")])
        .filter(
            _.d_date.between(date("2002-02-01"), date("2002-04-02")),
            _.ca_state == "GA",
            _.cc_county == "Williamson County",
            lambda t: catalog_sales.filter(
                t.cs_order_number == _.cs_order_number,
                t.cs_warehouse_sk != _.cs_warehouse_sk,
            ).count()
            > 0,
            lambda t: catalog_returns.filter(
                t.cs_order_number == _.cr_order_number
            ).count()
            == 0,
        )
        .agg(
            **{
                "order count": _.cs_order_number.nunique(),
                "total shipping cost": _.cs_ext_ship_cost.sum(),
                "total net profit": _.cs_net_profit.sum(),
            }
        )
        .order_by(_["order count"])
        .limit(100)
    )


@tpc_test("ds")
def test_17(store_sales, store_returns, catalog_sales, date_dim, store, item):
    d1 = date_dim
    d2 = date_dim.view()
    d3 = date_dim.view()
    return (
        store_sales.join(
            store_returns,
            [
                ("ss_customer_sk", "sr_customer_sk"),
                ("ss_item_sk", "sr_item_sk"),
                ("ss_ticket_number", "sr_ticket_number"),
            ],
        )
        .join(
            catalog_sales,
            [("sr_customer_sk", "cs_bill_customer_sk"), ("sr_item_sk", "cs_item_sk")],
        )
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(
            d1.filter(_.d_quarter_name == "2001Q1").select("d_date_sk"),
            [("ss_sold_date_sk", "d_date_sk")],
        )
        # ideally we wouldn't need this but an integrity conflict results otherwise
        .drop("d_date_sk")
        .join(
            d2.filter(_.d_quarter_name.isin(("2001Q1", "2001Q2", "2001Q3"))).select(
                "d_date_sk"
            ),
            [("sr_returned_date_sk", "d_date_sk")],
        )
        .join(
            d3.filter(_.d_quarter_name.isin(("2001Q1", "2001Q2", "2001Q3"))).select(
                "d_date_sk"
            ),
            [("cs_sold_date_sk", "d_date_sk")],
        )
        .group_by(_.i_item_id, _.i_item_desc, _.s_state)
        .agg(
            store_sales_quantitycount=_.ss_quantity.count(),
            store_sales_quantityave=_.ss_quantity.mean(),
            store_sales_quantitystdev=_.ss_quantity.std(),
            store_sales_quantitycov=_.ss_quantity.std() / _.ss_quantity.mean(),
            store_returns_quantitycount=_.sr_return_quantity.count(),
            store_returns_quantityave=_.sr_return_quantity.mean(),
            store_returns_quantitystdev=_.sr_return_quantity.std(),
            store_returns_quantitycov=(
                _.sr_return_quantity.std() / _.sr_return_quantity.mean()
            ),
            catalog_sales_quantitycount=_.cs_quantity.count(),
            catalog_sales_quantityave=_.cs_quantity.mean(),
            catalog_sales_quantitystdev=_.cs_quantity.std(),
            catalog_sales_quantitycov=_.cs_quantity.std() / _.cs_quantity.mean(),
        )
        .order_by(_.i_item_id, _.i_item_desc, _.s_state)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.xfail(raises=NotImplementedError, reason="requires rollup")
def test_18(
    catalog_sales, customer_demographics, customer, customer_address, date_dim, item
):
    raise NotImplementedError()


@tpc_test("ds")
def test_19(date_dim, store_sales, item, customer, customer_address, store):
    return (
        date_dim.join(store_sales, [("d_date_sk", "ss_sold_date_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(store, [("ss_store_sk", "s_store_sk"), _.ca_zip[:5] != store.s_zip[:5]])
        .filter(_.i_manager_id == 8, _.d_moy == 11, _.d_year == 1998)
        .group_by(
            brand_id=_.i_brand_id,
            brand=_.i_brand,
            i_manufact_id=_.i_manufact_id,
            i_manufact=_.i_manufact,
        )
        .agg(ext_price=_.ss_ext_sales_price.sum())
        .order_by(
            _.ext_price.desc(), _.brand, _.brand_id, _.i_manufact_id, _.i_manufact
        )
        .limit(100)
    )


@tpc_test("ds")
def test_20(catalog_sales, item, date_dim):
    return (
        catalog_sales.join(
            item.filter(_.i_category.isin(("Sports", "Books", "Home"))),
            [("cs_item_sk", "i_item_sk")],
        )
        .join(
            date_dim.filter(_.d_date.between(date("1999-02-22"), date("1999-03-24"))),
            [("cs_sold_date_sk", "d_date_sk")],
        )
        .group_by(
            _.i_item_id, _.i_item_desc, _.i_category, _.i_class, _.i_current_price
        )
        .agg(itemrevenue=_.cs_ext_sales_price.sum())
        .mutate(
            revenueratio=(
                _.itemrevenue * 100.0000 / _.itemrevenue.sum().over(group_by=_.i_class)
            )
        )
        .order_by(
            _.i_category.asc(nulls_first=True),
            _.i_class.asc(nulls_first=True),
            _.i_item_id.asc(nulls_first=True),
            _.i_item_desc.asc(nulls_first=True),
            _.revenueratio.asc(nulls_first=True),
        )
        .limit(100)
    )


@tpc_test("ds")
def test_21(inventory, warehouse, item, date_dim):
    return (
        inventory.join(warehouse, [("inv_warehouse_sk", "w_warehouse_sk")])
        .join(
            item.filter(_.i_current_price.between(0.99, 1.49)),
            [("inv_item_sk", "i_item_sk")],
        )
        .join(
            date_dim.filter(_.d_date.between(date("2000-02-10"), date("2000-04-10"))),
            [("inv_date_sk", "d_date_sk")],
        )
        .group_by(_.w_warehouse_name, _.i_item_id)
        .agg(
            inv_before=_.inv_quantity_on_hand.sum(where=_.d_date < date("2000-03-11")),
            inv_after=_.inv_quantity_on_hand.sum(where=_.d_date >= date("2000-03-11")),
        )
        .filter(
            ifelse(
                _.inv_before > 0, (_.inv_after * 1.000) / _.inv_before, null()
            ).between(2.000 / 3.000, 3.000 / 2.000)
        )
        .order_by(
            _.w_warehouse_name.asc(nulls_first=True), _.i_item_id.asc(nulls_first=True)
        )
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.xfail(raises=NotImplementedError, reason="requires rollup")
def test_22(inventory, date_dim, item):
    raise NotImplementedError()


@tpc_test("ds")
@pytest.mark.xfail(raises=NotImplementedError, reason="requires rollup")
def test_23(inventory, date_dim, item):
    raise NotImplementedError()


@tpc_test("ds", result_is_empty=True)
def test_24(store_sales, store_returns, store, item, customer, customer_address):
    ssales = (
        store_sales.join(
            store_returns,
            [("ss_ticket_number", "sr_ticket_number"), ("ss_item_sk", "sr_item_sk")],
        )
        .join(store.filter(_.s_market_id == 8), [("ss_store_sk", "s_store_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .join(
            customer_address,
            [
                ("c_current_addr_sk", "ca_address_sk"),
                _.c_birth_country != customer_address.ca_country.upper(),
                _.s_zip == customer_address.ca_zip,
            ],
        )
        .group_by(
            _.c_last_name,
            _.c_first_name,
            _.s_store_name,
            _.ca_state,
            _.s_state,
            _.i_color,
            _.i_current_price,
            _.i_manager_id,
            _.i_units,
            _.i_size,
        )
        .agg(netpaid=_.ss_net_paid.sum())
    )
    return (
        ssales.filter(_.i_color == "peach")
        .group_by(_.c_last_name, _.c_first_name, _.s_store_name)
        .having(_.netpaid.sum() > ssales.netpaid.mean().as_scalar() * 0.05)
        .agg(paid=_.netpaid.sum())
        .order_by(~s.c("paid"))
    )


@tpc_test("ds", result_is_empty=True)
def test_25(store_sales, store_returns, catalog_sales, date_dim, store, item):
    date_dim = date_dim.filter(_.d_year == 2001)
    return (
        store_sales.join(
            store_returns,
            [
                ("ss_customer_sk", "sr_customer_sk"),
                ("ss_ticket_number", "sr_ticket_number"),
                ("ss_item_sk", "sr_item_sk"),
            ],
        )
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(
            catalog_sales,
            [("sr_customer_sk", "cs_bill_customer_sk"), ("sr_item_sk", "cs_item_sk")],
        )
        .join(date_dim.filter(_.d_moy == 4), [("ss_sold_date_sk", "d_date_sk")])
        .drop(s.startswith("d_"))
        .join(
            date_dim.view().filter(_.d_moy.between(4, 10)),
            [("sr_returned_date_sk", "d_date_sk")],
        )
        .join(
            date_dim.view().filter(_.d_moy.between(4, 10)),
            [("cs_sold_date_sk", "d_date_sk")],
        )
        .group_by(_.i_item_id, _.i_item_desc, _.s_store_id, _.s_store_name)
        .agg(
            store_sales_profit=_.ss_net_profit.sum(),
            store_returns_loss=_.sr_net_loss.sum(),
            catalog_sales_profit=_.cs_net_profit.sum(),
        )
        .order_by(~s.endswith(("profit", "loss")))
        .limit(100)
    )


@tpc_test("ds")
def test_26(catalog_sales, customer_demographics, date_dim, item, promotion):
    return (
        catalog_sales.join(
            date_dim.filter(_.d_year == 2000), [("cs_sold_date_sk", "d_date_sk")]
        )
        .join(item, [("cs_item_sk", "i_item_sk")])
        .join(
            customer_demographics.filter(
                _.cd_gender == "M",
                _.cd_marital_status == "S",
                _.cd_education_status == "College",
            ),
            [("cs_bill_cdemo_sk", "cd_demo_sk")],
        )
        .join(
            promotion.filter((_.p_channel_email == "N") | (_.p_channel_event == "N")),
            [("cs_promo_sk", "p_promo_sk")],
        )
        .group_by(_.i_item_id)
        .agg(
            agg1=_.cs_quantity.mean(),
            agg2=_.cs_list_price.mean(),
            agg3=_.cs_coupon_amt.mean(),
            agg4=_.cs_sales_price.mean(),
        )
        .order_by(_.i_item_id)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(["datafusion"], reason="Failed to plan")
def test_27(store_sales, customer_demographics, date_dim, store, item):
    results = (
        store_sales.join(customer_demographics, [("ss_cdemo_sk", "cd_demo_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(
            _.cd_gender == "M",
            _.cd_marital_status == "S",
            _.cd_education_status == "College",
            _.d_year == 2002,
            _.s_state == "TN",
        )
        .select(
            _.i_item_id,
            _.s_state,
            g_state=lit(0),
            agg1=_.ss_quantity,
            agg2=_.ss_list_price,
            agg3=_.ss_coupon_amt,
            agg4=_.ss_sales_price,
        )
    )
    return (
        results.group_by(~s.startswith("agg"))
        .agg(s.across(s.startswith("agg"), _.mean()))
        .union(
            results.group_by(
                _.i_item_id, s_state=null(store.s_state.type()), g_state=lit(1)
            ).agg(s.across(s.startswith("agg"), _.mean()))
        )
        .union(
            results.group_by(
                i_item_id=null(item.i_item_id.type()),
                s_state=null(store.s_state.type()),
                g_state=lit(1),
            ).agg(s.across(s.startswith("agg"), _.mean()))
        )
        .order_by(_.i_item_id.asc(nulls_first=True), _.s_state.asc(nulls_first=True))
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse is off-by-one on the result",
)
def test_28(store_sales):
    quantity = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    list_price = [(lower, lower + 10) for lower in (8, 90, 142, 135, 122, 154)]
    coupon_amt = [
        (lower, lower + 1000) for lower in (459, 2323, 12214, 6071, 836, 7326)
    ]
    wholesale_cost = [(lower, lower + 20) for lower in (57, 31, 79, 38, 17, 7)]
    first, *rest = (
        store_sales.filter(
            _.ss_quantity.between(*qty_bounds),
            _.ss_list_price.between(*lp_bounds)
            | _.ss_coupon_amt.between(*cp_bounds)
            | _.ss_wholesale_cost.between(*ws_bounds),
        ).agg(
            **{
                f"B{i:d}_LP": _.ss_list_price.mean(),
                f"B{i:d}_CNT": _.ss_list_price.count(),
                f"B{i:d}_CNTD": _.ss_list_price.nunique(),
            }
        )
        for i, (qty_bounds, lp_bounds, cp_bounds, ws_bounds) in enumerate(
            zip(quantity, list_price, coupon_amt, wholesale_cost), start=1
        )
    )
    return first.cross_join(*rest).limit(100)


@tpc_test("ds")
def test_29(store_sales, store_returns, catalog_sales, date_dim, store, item):
    d1 = (
        date_dim.filter(_.d_moy == 9, _.d_year == 1999)
        .drop(~s.c("d_date_sk"))
        .rename(d1_date_sk="d_date_sk")
    )
    d2 = (
        date_dim.filter(_.d_moy.between(9, 9 + 3), _.d_year == 1999)
        .drop(~s.c("d_date_sk"))
        .rename(d2_date_sk="d_date_sk")
    )
    d3 = (
        date_dim.filter(_.d_year.isin((1999, 1999 + 1, 1999 + 2)))
        .drop(~s.c("d_date_sk"))
        .rename(d3_date_sk="d_date_sk")
    )
    return (
        store_sales.join(
            store_returns,
            [
                ("ss_customer_sk", "sr_customer_sk"),
                ("ss_item_sk", "sr_item_sk"),
                ("ss_ticket_number", "sr_ticket_number"),
            ],
        )
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(
            catalog_sales,
            [("sr_customer_sk", "cs_bill_customer_sk"), ("sr_item_sk", "cs_item_sk")],
        )
        .join(d1, [("ss_sold_date_sk", "d1_date_sk")])
        .join(d2, [("sr_returned_date_sk", "d2_date_sk")])
        .join(d3, [("cs_sold_date_sk", "d3_date_sk")])
        .group_by("i_item_id", "i_item_desc", "s_store_id", "s_store_name")
        .agg(
            store_sales_quantity=_.ss_quantity.sum(),
            store_returns_quantity=_.sr_return_quantity.sum(),
            catalog_sales_quantity=_.cs_quantity.sum(),
        )
        .order_by(~s.endswith("_quantity"))
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
def test_30(web_returns, date_dim, customer_address, customer):
    customer_total_return = (
        web_returns.join(
            date_dim.filter(_.d_year == 2002), [("wr_returned_date_sk", "d_date_sk")]
        )
        .join(customer_address, [("wr_returning_addr_sk", "ca_address_sk")])
        .group_by(ctr_customer_sk=_.wr_returning_customer_sk, ctr_state=_.ca_state)
        .agg(ctr_total_return=_.wr_return_amt.sum())
    )
    return (
        customer_total_return.filter(
            lambda ctr1: (
                ctr1.ctr_total_return
                > customer_total_return.view()
                .filter(lambda ctr2: ctr1.ctr_state == ctr2.ctr_state)
                .ctr_total_return.mean()
                .as_scalar()
                * 1.2
            )
        )
        .join(customer, [("ctr_customer_sk", "c_customer_sk")])
        .join(
            customer_address.filter(_.ca_state == "GA"),
            [("c_current_addr_sk", "ca_address_sk")],
        )
        .select(
            _.c_customer_id,
            _.c_salutation,
            _.c_first_name,
            _.c_last_name,
            _.c_preferred_cust_flag,
            _.c_birth_day,
            _.c_birth_month,
            _.c_birth_year,
            _.c_birth_country,
            _.c_login,
            _.c_email_address,
            _.c_last_review_date_sk,
            _.ctr_total_return,
        )
        .order_by(s.across(s.all(), _.asc(nulls_first=True)))
    )


@tpc_test("ds")
def test_31(store_sales, date_dim, customer_address, web_sales):
    ss = (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .group_by(_.ca_county, _.d_qoy, _.d_year)
        .agg(store_sales=_.ss_ext_sales_price.sum())
    )
    ws = (
        web_sales.join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ws_bill_addr_sk", "ca_address_sk")])
        .group_by(_.ca_county, _.d_qoy, _.d_year)
        .agg(web_sales=_.ws_ext_sales_price.sum())
    )
    ss1 = ss.filter(_.d_qoy == 1, _.d_year == 2000).select(
        "ca_county", "d_year", ss1_store_sales="store_sales"
    )
    ss2 = ss.filter(_.d_qoy == 2, _.d_year == 2000).select(
        "ca_county", ss2_store_sales="store_sales"
    )
    ss3 = ss.filter(_.d_qoy == 3, _.d_year == 2000).select(
        "ca_county", ss3_store_sales="store_sales"
    )
    ws1 = ws.filter(_.d_qoy == 1, _.d_year == 2000).select(
        "ca_county", ws1_web_sales="web_sales"
    )
    ws2 = ws.filter(_.d_qoy == 2, _.d_year == 2000).select(
        "ca_county", ws2_web_sales="web_sales"
    )
    ws3 = ws.filter(_.d_qoy == 3, _.d_year == 2000).select(
        "ca_county", ws3_web_sales="web_sales"
    )

    return (
        ss1.join(ss2, "ca_county")
        .join(ss3, "ca_county")
        .join(ws1, "ca_county")
        .join(ws2, "ca_county")
        .join(ws3, "ca_county")
        .filter(
            ifelse(
                _.ws1_web_sales > 0,
                (_.ws2_web_sales * 1.0000) / _.ws1_web_sales,
                null(),
            )
            > ifelse(
                _.ss1_store_sales > 0,
                (_.ss2_store_sales * 1.0000) / _.ss1_store_sales,
                null(),
            ),
            ifelse(
                _.ws2_web_sales > 0,
                (_.ws3_web_sales * 1.0000) / _.ws2_web_sales,
                null(),
            )
            > ifelse(
                _.ss2_store_sales > 0,
                (_.ss3_store_sales * 1.0000) / _.ss2_store_sales,
                null(),
            ),
        )
        .select(
            _.ca_county,
            _.d_year,
            web_q1_q2_increase=(_.ws2_web_sales * 1.0000) / _.ws1_web_sales,
            store_q1_q2_increase=(_.ss2_store_sales * 1.0000) / _.ss1_store_sales,
            web_q2_q3_increase=(_.ws3_web_sales * 1.0000) / _.ws2_web_sales,
            store_q2_q3_increase=(_.ss3_store_sales * 1.0000) / _.ss2_store_sales,
        )
        .order_by(ss1.ca_county)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
@pytest.mark.notyet(
    ["trino"],
    raises=TrinoUserError,
    reason="Given correlated subquery is not supported",
)
def test_32(catalog_sales, item, date_dim):
    return (
        catalog_sales.join(
            item.filter(_.i_manufact_id == 977), [("cs_item_sk", "i_item_sk")]
        )
        .join(
            date_dim.filter(_.d_date.between(date("2000-01-27"), date("2000-04-26"))),
            [("cs_sold_date_sk", "d_date_sk")],
        )
        .filter(
            lambda t: (
                t.cs_ext_discount_amt
                > catalog_sales.view()
                .filter(_.cs_item_sk == t.i_item_sk)
                .join(
                    date_dim.view().filter(
                        _.d_date.between(date("2000-01-27"), date("2000-04-26"))
                    ),
                    [("cs_sold_date_sk", "d_date_sk")],
                )
                .cs_ext_discount_amt.mean()
                .as_scalar()
                * 1.3
            )
        )
        .agg(_.cs_ext_discount_amt.sum().name("excess discount amount"))
        .limit(100)
    )


@tpc_test("ds")
def test_33(store_sales, date_dim, customer_address, item, catalog_sales, web_sales):
    electronics = _.i_manufact_id.isin(
        item.filter(_.i_category.isin(("Electronics",))).i_manufact_id
    )
    dates = date_dim.filter(_.d_year == 1998, _.d_moy == 5)
    addresses = customer_address.filter(_.ca_gmt_offset == -5)
    ss = (
        store_sales.join(dates, [("ss_sold_date_sk", "d_date_sk")])
        .join(addresses, [("ss_addr_sk", "ca_address_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(electronics)
        .group_by(_.i_manufact_id)
        .agg(total_sales=_.ss_ext_sales_price.sum())
    )
    cs = (
        catalog_sales.join(dates, [("cs_sold_date_sk", "d_date_sk")])
        .join(addresses, [("cs_bill_addr_sk", "ca_address_sk")])
        .join(item, [("cs_item_sk", "i_item_sk")])
        .filter(electronics)
        .group_by(_.i_manufact_id)
        .agg(total_sales=_.cs_ext_sales_price.sum())
    )
    ws = (
        web_sales.join(dates, [("ws_sold_date_sk", "d_date_sk")])
        .join(addresses, [("ws_bill_addr_sk", "ca_address_sk")])
        .join(item, [("ws_item_sk", "i_item_sk")])
        .filter(electronics)
        .group_by(_.i_manufact_id)
        .agg(total_sales=_.ws_ext_sales_price.sum())
    )
    return (
        union(ss, cs, ws)
        .group_by(_.i_manufact_id)
        .agg(total_sales=_.total_sales.sum())
        .order_by(_.total_sales)
        .limit(100)
    )


@tpc_test("ds")
def test_34(store_sales, date_dim, store, household_demographics, customer):
    return (
        store_sales.join(
            date_dim.filter(
                _.d_dom.between(1, 3) | _.d_dom.between(25, 28),
                _.d_year.isin((1999, 1999 + 1, 1999 + 2)),
            ),
            [("ss_sold_date_sk", "d_date_sk")],
        )
        .join(
            store.filter(_.s_county == "Williamson County"),
            [("ss_store_sk", "s_store_sk")],
        )
        .join(
            household_demographics.filter(
                (_.hd_buy_potential == ">10000") | (_.hd_buy_potential == "Unknown"),
                _.hd_vehicle_count > 0,
                ifelse(
                    _.hd_vehicle_count > 0,
                    (_.hd_dep_count * 1.0000) / _.hd_vehicle_count,
                    null(),
                )
                > 1.2,
            ),
            [("ss_hdemo_sk", "hd_demo_sk")],
        )
        .group_by(_.ss_ticket_number, _.ss_customer_sk)
        .agg(cnt=_.count())
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .filter(_.cnt.between(15, 20))
        .select(
            _.c_last_name,
            _.c_first_name,
            _.c_salutation,
            _.c_preferred_cust_flag,
            _.ss_ticket_number,
            _.cnt,
        )
        .order_by(
            _.c_last_name.asc(nulls_first=True),
            _.c_first_name.asc(nulls_first=True),
            _.c_salutation.asc(nulls_first=True),
            _.c_preferred_cust_flag.desc(nulls_first=True),
            _.ss_ticket_number.asc(nulls_first=True),
        )
    )


@tpc_test("ds")
@pytest.mark.notyet(["datafusion"], reason="internal error")
@pytest.mark.notyet(
    ["trino"],
    raises=TrinoUserError,
    reason="Given correlated subquery is not supported",
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
def test_35(
    customer,
    customer_address,
    customer_demographics,
    store_sales,
    web_sales,
    catalog_sales,
    date_dim,
):
    dates = date_dim.filter(_.d_year == 2002, _.d_qoy < 4)
    return (
        customer.join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(customer_demographics, [("c_current_cdemo_sk", "cd_demo_sk")])
        .filter(
            lambda t: (
                store_sales.filter(t.c_customer_sk == _.ss_customer_sk)
                .join(dates, [("ss_sold_date_sk", "d_date_sk")])
                .count()
                > 0
            ),
            lambda t: (
                (
                    web_sales.filter(t.c_customer_sk == _.ws_bill_customer_sk)
                    .join(dates, [("ws_sold_date_sk", "d_date_sk")])
                    .count()
                    > 0
                )
                | (
                    catalog_sales.filter(t.c_customer_sk == _.cs_ship_customer_sk)
                    .join(dates, [("cs_sold_date_sk", "d_date_sk")])
                    .count()
                    > 0
                )
            ),
        )
        .group_by(
            _.ca_state,
            _.cd_gender,
            _.cd_marital_status,
            _.cd_dep_count,
            _.cd_dep_employed_count,
            _.cd_dep_college_count,
        )
        .agg(
            cnt1=_.count(),
            min1=_.cd_dep_count.min(),
            max1=_.cd_dep_count.max(),
            avg1=_.cd_dep_count.mean(),
            cnt2=_.count(),
            min2=_.cd_dep_employed_count.min(),
            max2=_.cd_dep_employed_count.max(),
            avg2=_.cd_dep_employed_count.mean(),
            cnt3=_.count(),
            min3=_.cd_dep_college_count.min(),
            max3=_.cd_dep_college_count.max(),
            avg3=_.cd_dep_college_count.mean(),
        )
        .relocate("cd_dep_employed_count", before="cnt2")
        .relocate("cd_dep_college_count", before="cnt3")
        .order_by(
            s.across(s.startswith("cd_") | s.c("ca_state"), _.asc(nulls_first=True))
        )
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason=(
        "column name shadowing requires rewriting the query; "
        "this 'feature' is so annoying"
    ),
)
def test_36(store_sales, date_dim, item, store):
    results = (
        store_sales.join(
            date_dim.filter(_.d_year == 2001), [("ss_sold_date_sk", "d_date_sk")]
        )
        .join(item, [("ss_item_sk", "i_item_sk")])
        .join(store.filter(_.s_state == "TN"), [("ss_store_sk", "s_store_sk")])
        .group_by(_.i_category, _.i_class)
        .agg(
            ss_net_profit=_.ss_net_profit.sum(),
            ss_ext_sales_price=_.ss_ext_sales_price.sum(),
            gross_margin=(_.ss_net_profit.sum() * 1.0000) / _.ss_ext_sales_price.sum(),
            g_category=lit(0),
            g_class=lit(0),
        )
        .relocate(s.c("i_category", "i_class"), after="gross_margin")
    )
    return (
        results.select(
            _.gross_margin,
            _.i_category,
            _.i_class,
            t_category=lit(0),
            t_class=lit(0),
            lochierarchy=lit(0),
        )
        .union(
            results.group_by(_.i_category)
            .agg(
                gross_margin=(_.ss_net_profit.sum() * 1.0000)
                / _.ss_ext_sales_price.sum(),
                i_class=null("str"),
                t_category=lit(0),
                t_class=lit(1),
                lochierarchy=lit(1),
            )
            .relocate("i_category", after="gross_margin"),
            distinct=True,
        )
        .union(
            results.agg(
                gross_margin=(_.ss_net_profit.sum() * 1.0000)
                / _.ss_ext_sales_price.sum(),
                i_category=null("str"),
                i_class=null("str"),
                t_category=lit(1),
                t_class=lit(1),
                lochierarchy=lit(2),
            ),
            distinct=True,
        )
        .mutate(
            rank_within_parent=_.gross_margin.rank().over(
                group_by=(
                    _.lochierarchy,
                    ifelse(_.t_class == 0, _.i_category, null("str")),
                ),
                order_by=_.gross_margin.asc(),
            )
            + 1  # add one because ibis is 0-indexed
        )
        .relocate("lochierarchy", before="rank_within_parent")
        .drop("t_class", "t_category")
        .order_by(
            _.lochierarchy.desc(nulls_first=True),
            ifelse(_.lochierarchy == 0, _.i_category, null("str")).asc(
                nulls_first=True
            ),
            _.rank_within_parent.asc(nulls_first=True),
        )
        .limit(100)
    )


@tpc_test("ds")
def test_37(item, inventory, date_dim, catalog_sales):
    return (
        item.join(inventory, [("i_item_sk", "inv_item_sk")])
        .join(date_dim, [("inv_date_sk", "d_date_sk")])
        .join(catalog_sales, [("i_item_sk", "cs_item_sk")])
        .filter(
            _.i_current_price.between(68, 68 + 30),
            _.i_manufact_id.isin((677, 940, 694, 808)),
            _.inv_quantity_on_hand.between(100, 500),
            _.d_date.between(date("2000-02-01"), date("2000-04-01")),
        )
        .group_by(_.i_item_id, _.i_item_desc, _.i_current_price)
        .agg()
        .order_by(_.i_item_id)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=AssertionError,
    reason="clickhouse returns an incorrect result for this query",
)
def test_38(store_sales, catalog_sales, web_sales, date_dim, customer):
    dates = date_dim.filter(_.d_month_seq.between(1200, 1200 + 11))
    columns = "c_last_name", "c_first_name", "d_date"
    return (
        store_sales.join(dates, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .select(*columns)
        .distinct()
        .intersect(
            catalog_sales.join(dates, [("cs_sold_date_sk", "d_date_sk")])
            .join(customer, [("cs_bill_customer_sk", "c_customer_sk")])
            .select(*columns)
            .distinct()
        )
        .intersect(
            web_sales.join(dates, [("ws_sold_date_sk", "d_date_sk")])
            .join(customer, [("ws_bill_customer_sk", "c_customer_sk")])
            .select(*columns)
            .distinct(),
        )
        .agg(cnt=_.count())
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["datafusion"],
    raises=ArrowNotImplementedError,
    reason="Unsupported cast from double to null using function cast_null",
)
def test_39(inventory, item, warehouse, date_dim):
    inv = (
        inventory.join(item, [("inv_item_sk", "i_item_sk")])
        .join(warehouse, [("inv_warehouse_sk", "w_warehouse_sk")])
        .join(date_dim.filter(_.d_year == 2001), [("inv_date_sk", "d_date_sk")])
        .group_by(_.w_warehouse_name, _.w_warehouse_sk, _.i_item_sk, _.d_moy)
        .agg(
            stdev=_.inv_quantity_on_hand.std(how="sample") * 1.0000,
            mean=_.inv_quantity_on_hand.mean(),
        )
        .filter(ifelse(_.mean == 0, 0, _.stdev / _.mean) > 1)
        .mutate(cov=_.stdev / _.mean.nullif(0))
    )
    inv1 = inv.filter(_.d_moy == 1)
    inv2 = inv.filter(_.d_moy == 1 + 1)
    return (
        inv1.join(inv2, ["i_item_sk", "w_warehouse_sk"])
        .select(
            wsk1=inv1.w_warehouse_sk,
            isk1=inv1.i_item_sk,
            dmoy1=inv1.d_moy,
            mean1=inv1.mean,
            cov1=inv1.cov,
            w_warehouse_sk=inv2.w_warehouse_sk,
            i_item_sk=inv2.i_item_sk,
            d_moy=inv2.d_moy,
            mean=inv2.mean,
            cov=inv2.cov,
        )
        .order_by(
            s.across(
                s.c("wsk1", "isk1", "dmoy1", "mean1", "cov1", "d_moy", "mean", "cov"),
                _.asc(nulls_first=True),
            )
        )
    )


@tpc_test("ds")
def test_40(catalog_sales, catalog_returns, warehouse, item, date_dim):
    return (
        catalog_sales.left_join(
            catalog_returns,
            [("cs_order_number", "cr_order_number"), ("cs_item_sk", "cr_item_sk")],
        )
        .join(warehouse, [("cs_warehouse_sk", "w_warehouse_sk")])
        .join(
            item.filter(_.i_current_price.between(0.99, 1.49)),
            [("cs_item_sk", "i_item_sk")],
        )
        .join(
            date_dim.filter(_.d_date.between(date("2000-02-10"), date("2000-04-10"))),
            [("cs_sold_date_sk", "d_date_sk")],
        )
        .group_by(_.w_state, _.i_item_id)
        .agg(
            sales_before=ifelse(
                _.d_date < date("2000-03-11"),
                _.cs_sales_price - _.cr_refunded_cash.fill_null(0),
                0,
            ).sum(),
            sales_after=ifelse(
                _.d_date >= date("2000-03-11"),
                _.cs_sales_price - _.cr_refunded_cash.fill_null(0),
                0,
            ).sum(),
        )
        .order_by(_.w_state, _.i_item_id)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["datafusion"],
    raises=Exception,
    reason="Error during planning: Correlated column is not allowed in predicate",
)
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
def test_41(item):
    return (
        item.view()
        .filter(
            _.i_manufact_id.between(738, 738 + 40),
            lambda i1: item.filter(
                lambda s: (
                    (i1.i_manufact == s.i_manufact)
                    & (
                        (
                            (s.i_category == "Women")
                            & s.i_color.isin(("powder", "khaki"))
                            & s.i_units.isin(("Ounce", "Oz"))
                            & s.i_size.isin(("medium", "extra large"))
                        )
                        | (
                            (s.i_category == "Women")
                            & s.i_color.isin(("brown", "honeydew"))
                            & s.i_units.isin(("Bunch", "Ton"))
                            & s.i_size.isin(("N/A", "small"))
                        )
                        | (
                            (s.i_category == "Men")
                            & s.i_color.isin(("floral", "deep"))
                            & s.i_units.isin(("N/A", "Dozen"))
                            & s.i_size.isin(("petite", "petite"))
                        )
                        | (
                            (s.i_category == "Men")
                            & s.i_color.isin(("light", "cornflower"))
                            & s.i_units.isin(("Box", "Pound"))
                            & s.i_size.isin(("medium", "extra large"))
                        )
                    )
                )
                | (
                    (i1.i_manufact == s.i_manufact)
                    & (
                        (
                            (s.i_category == "Women")
                            & s.i_color.isin(("midnight", "snow"))
                            & s.i_units.isin(("Pallet", "Gross"))
                            & s.i_size.isin(("medium", "extra large"))
                        )
                        | (
                            (s.i_category == "Women")
                            & s.i_color.isin(("cyan", "papaya"))
                            & s.i_units.isin(("Cup", "Dram"))
                            & s.i_size.isin(("N/A", "small"))
                        )
                        | (
                            (s.i_category == "Men")
                            & s.i_color.isin(("orange", "frosted"))
                            & s.i_units.isin(("Each", "Tbl"))
                            & s.i_size.isin(("petite", "petite"))
                        )
                        | (
                            (s.i_category == "Men")
                            & s.i_color.isin(("forest", "ghost"))
                            & s.i_units.isin(("Lb", "Bundle"))
                            & s.i_size.isin(("medium", "extra large"))
                        )
                    )
                )
            ).count()
            > 0,
        )
        .select(_.i_product_name)
        .distinct()
        .order_by(s.all())
        .limit(100)
    )


@tpc_test("ds")
def test_42(date_dim, store_sales, item):
    return (
        date_dim.filter(_.d_moy == 11, _.d_year == 2000)
        .join(store_sales, [("d_date_sk", "ss_sold_date_sk")])
        .join(item.filter(_.i_manager_id == 1), [("ss_item_sk", "i_item_sk")])
        .group_by(_.d_year, _.i_category_id, _.i_category)
        .agg(total_sales=_.ss_ext_sales_price.sum())
        .order_by(_.total_sales.desc(), ~s.c("total_sales"))
        .limit(100)
    )


@tpc_test("ds")
def test_43(date_dim, store_sales, store):
    return (
        date_dim.filter(_.d_year == 2000)
        .join(store_sales, [("d_date_sk", "ss_sold_date_sk")])
        .join(store.filter(_.s_gmt_offset == -5), [("ss_store_sk", "s_store_sk")])
        .group_by(_.s_store_name, _.s_store_id)
        .agg(
            **{
                f"{name[:3].lower()}_sales": _.ss_sales_price.sum(
                    where=_.d_day_name == name
                )
                for name in map(cal.day_name.__getitem__, range(-1, 6))
            }
        )
        .order_by(s.all())
        .limit(100)
    )


@tpc_test("ds")
def test_44(store_sales, item):
    base = (
        store_sales.filter(_.ss_store_sk == 4)
        .group_by(item_sk=_.ss_item_sk)
        .having(
            _.ss_net_profit.mean()
            > 0.9
            * (
                store_sales.filter(_.ss_store_sk == 4, _.ss_addr_sk.isnull())
                .group_by(_.ss_store_sk)
                .agg(_.ss_net_profit.mean())
                .drop("ss_store_sk")
                .as_scalar()
            )
        )
        .agg(rank_col=_.ss_net_profit.mean())
    )
    ascending = base.select(
        "item_sk", rnk=rank().over(order_by=_.rank_col.asc()) + 1
    ).filter(_.rnk < 11)
    descending = base.select(
        "item_sk", rnk=rank().over(order_by=_.rank_col.desc()) + 1
    ).filter(_.rnk < 11)
    i1 = item
    i2 = item.view()
    return (
        ascending.join(descending, "rnk")
        .join(i1, ascending.item_sk == i1.i_item_sk)
        .join(i2, descending.item_sk == i2.i_item_sk)
        .select(
            ascending.rnk,
            best_performing=i1.i_product_name,
            worst_performing=i2.i_product_name,
        )
        .order_by(ascending.rnk)
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["datafusion"], raises=Exception, reason="Unsupported feature in DataFusion SQL"
)
def test_45(web_sales, customer, customer_address, date_dim, item):
    return (
        web_sales.join(customer, [("ws_bill_customer_sk", "c_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(item, [("ws_item_sk", "i_item_sk")])
        .join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .filter(
            _.ca_zip[:5].isin(
                (
                    "85669",
                    "86197",
                    "88274",
                    "83405",
                    "86475",
                    "85392",
                    "85460",
                    "80348",
                    "81792",
                )
            )
            | _.i_item_id.isin(
                item.view()
                .filter(_.i_item_sk.isin((2, 3, 5, 7, 11, 13, 17, 19, 23, 29)))
                .i_item_id
            ),
            _.d_qoy == 2,
            _.d_year == 2001,
        )
        .group_by(_.ca_zip, _.ca_city)
        .agg(total_web_sales=_.ws_sales_price.sum())
        .order_by(~s.c("total_web_sales"))
        .limit(100)
    )


@tpc_test("ds")
def test_46(
    store_sales, date_dim, store, household_demographics, customer_address, customer
):
    current_addr = customer_address.view()
    return (
        store_sales.join(
            date_dim.filter(
                _.d_dow.isin((6, 0)), _.d_year.isin((1999, 1999 + 1, 1999 + 2))
            ),
            [("ss_sold_date_sk", "d_date_sk")],
        )
        .join(
            store.filter(_.s_city.isin(("Fairview", "Midway"))),
            [("ss_store_sk", "s_store_sk")],
        )
        .join(
            household_demographics.filter(
                (_.hd_dep_count == 4) | (_.hd_vehicle_count == 3)
            ),
            [("ss_hdemo_sk", "hd_demo_sk")],
        )
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .group_by(
            _.ss_ticket_number, _.ss_customer_sk, _.ss_addr_sk, bought_city=_.ca_city
        )
        .agg(amt=_.ss_coupon_amt.sum(), profit=_.ss_net_profit.sum())
        .drop(_.ss_addr_sk)
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .join(
            current_addr,
            [
                ("c_current_addr_sk", "ca_address_sk"),
                current_addr.ca_city != _.bought_city,
            ],
        )
        .select(
            _.c_last_name,
            _.c_first_name,
            _.ca_city,
            _.bought_city,
            _.ss_ticket_number,
            _.amt,
            _.profit,
        )
        .order_by(s.across(~s.c("amt", "profit"), _.asc(nulls_first=True)))
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="correlated subqueries don't exist in clickhouse",
)
def test_47(item, store_sales, date_dim, store):
    v1 = (
        item.join(store_sales, [("i_item_sk", "ss_item_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .filter(
            (_.d_year == 1999)
            | ((_.d_year == 1999 - 1) & (_.d_moy == 12))
            | ((_.d_year == 1999 + 1) & (_.d_moy == 1))
        )
        .group_by(
            _.i_category, _.i_brand, _.s_store_name, _.s_company_name, _.d_year, _.d_moy
        )
        .agg(sum_sales=_.ss_sales_price.sum())
        .mutate(
            avg_monthly_sales=_.sum_sales.mean().over(
                # TODO: add support for selectors in window over specification
                # group_by=~s.c("sum_sales", "d_moy")
                group_by=(
                    _.i_category,
                    _.i_brand,
                    _.s_store_name,
                    _.s_company_name,
                    _.d_year,
                )
            ),
            rn=rank().over(
                group_by=(_.i_category, _.i_brand, _.s_store_name, _.s_company_name),
                order_by=(_.d_year, _.d_moy),
            ),
        )
    )
    v1_lag = v1.view()
    v1_lead = v1.view()
    v2 = v1.join(
        v1_lag,
        [
            "i_category",
            "i_brand",
            "s_store_name",
            "s_company_name",
            v1.rn == v1_lag.rn + 1,
        ],
    ).join(
        v1_lead,
        [
            "i_category",
            "i_brand",
            "s_store_name",
            "s_company_name",
            v1.rn == v1_lead.rn - 1,
        ],
    )
    return (
        v2.select(
            v1.i_category,
            v1.i_brand,
            v1.s_store_name,
            v1.s_company_name,
            v1.d_year,
            v1.d_moy,
            v1.avg_monthly_sales,
            v1.sum_sales,
            psum=v1_lag.sum_sales,
            nsum=v1_lead.sum_sales,
        )
        .filter(
            _.d_year == 1999,
            _.avg_monthly_sales > 0,
            ifelse(
                _.avg_monthly_sales > 0,
                (_.sum_sales - _.avg_monthly_sales).abs() / _.avg_monthly_sales,
                null(),
            )
            > 0.1,
        )
        .order_by(_.sum_sales - _.avg_monthly_sales, s.all())
        .limit(100)
    )


@tpc_test("ds")
def test_48(store_sales, store, customer_demographics, customer_address, date_dim):
    return (
        store_sales.join(store, [("ss_store_sk", "s_store_sk")])
        .join(date_dim.filter(_.d_year == 2000), [("ss_sold_date_sk", "d_date_sk")])
        .join(customer_demographics, [("ss_cdemo_sk", "cd_demo_sk")])
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .filter(
            (
                (_.cd_marital_status == "M")
                & (_.cd_education_status == "4 yr Degree")
                & _.ss_sales_price.between(100.00, 150.00)
            )
            | (
                (_.cd_marital_status == "D")
                & (_.cd_education_status == "2 yr Degree")
                & _.ss_sales_price.between(50.00, 100.00)
            )
            | (
                (_.cd_marital_status == "S")
                & (_.cd_education_status == "College")
                & _.ss_sales_price.between(150.00, 200.00)
            ),
            (
                (_.ca_country == "United States")
                & _.ca_state.isin(("CO", "OH", "TX"))
                & _.ss_net_profit.between(0, 2000)
            )
            | (
                (_.ca_country == "United States")
                & _.ca_state.isin(("OR", "MN", "KY"))
                & _.ss_net_profit.between(150, 3000)
            )
            | (
                (_.ca_country == "United States")
                & _.ca_state.isin(("VA", "CA", "MS"))
                & _.ss_net_profit.between(50, 25000)
            ),
        )
        .agg(total=_.ss_quantity.sum())
    )


@tpc_test("ds")
def test_49(
    web_sales,
    web_returns,
    date_dim,
    catalog_sales,
    catalog_returns,
    store_sales,
    store_returns,
):
    in_web = (
        web_sales.left_join(
            web_returns,
            [("ws_order_number", "wr_order_number"), ("ws_item_sk", "wr_item_sk")],
        )
        .join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .filter(
            _.wr_return_amt > 10000,
            _.ws_net_profit > 1,
            _.ws_net_paid > 0,
            _.ws_quantity > 0,
            _.d_year == 2001,
            _.d_moy == 12,
        )
        .group_by(item=_.ws_item_sk)
        .agg(
            return_ratio=_.wr_return_quantity.fill_null(0).sum().cast("decimal(15, 4)")
            / _.ws_quantity.fill_null(0).sum().cast("decimal(15, 4)"),
            currency_ratio=_.wr_return_amt.fill_null(0).sum().cast("decimal(15, 4)")
            / _.ws_net_paid.fill_null(0).sum().cast("decimal(15, 4)"),
        )
    )
    in_cat = (
        catalog_sales.left_join(
            catalog_returns,
            [("cs_order_number", "cr_order_number"), ("cs_item_sk", "cr_item_sk")],
        )
        .join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .filter(
            _.cr_return_amount > 10000,
            _.cs_net_profit > 1,
            _.cs_net_paid > 0,
            _.cs_quantity > 0,
            _.d_year == 2001,
            _.d_moy == 12,
        )
        .group_by(item=_.cs_item_sk)
        .agg(
            return_ratio=_.cr_return_quantity.fill_null(0).sum().cast("decimal(15, 4)")
            / _.cs_quantity.fill_null(0).sum().cast("decimal(15, 4)"),
            currency_ratio=_.cr_return_amount.fill_null(0).sum().cast("decimal(15, 4)")
            / _.cs_net_paid.fill_null(0).sum().cast("decimal(15, 4)"),
        )
    )
    in_store = (
        store_sales.left_join(
            store_returns,
            [("ss_ticket_number", "sr_ticket_number"), ("ss_item_sk", "sr_item_sk")],
        )
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .filter(
            _.sr_return_amt > 10000,
            _.ss_net_profit > 1,
            _.ss_net_paid > 0,
            _.ss_quantity > 0,
            _.d_year == 2001,
            _.d_moy == 12,
        )
        .group_by(item=_.ss_item_sk)
        .agg(
            return_ratio=_.sr_return_quantity.fill_null(0).sum().cast("decimal(15, 4)")
            / _.ss_quantity.fill_null(0).sum().cast("decimal(15, 4)"),
            currency_ratio=_.sr_return_amt.fill_null(0).sum().cast("decimal(15, 4)")
            / _.ss_net_paid.fill_null(0).sum().cast("decimal(15, 4)"),
        )
    )

    return (
        in_web.mutate(
            return_rank=rank().over(range=(None, 0), order_by=_.return_ratio) + 1,
            currency_rank=rank().over(range=(None, 0), order_by=_.currency_ratio) + 1,
        )
        .filter((_.return_rank <= 10) | (_.currency_rank <= 10))
        .mutate(channel=lit("web"))
        .relocate("channel", before="item")
        .union(
            in_cat.mutate(
                return_rank=rank().over(range=(None, 0), order_by=_.return_ratio) + 1,
                currency_rank=rank().over(range=(None, 0), order_by=_.currency_ratio)
                + 1,
            )
            .filter((_.return_rank <= 10) | (_.currency_rank <= 10))
            .mutate(channel=lit("catalog"))
            .relocate("channel", before="item")
        )
        .union(
            in_store.mutate(
                return_rank=rank().over(range=(None, 0), order_by=_.return_ratio) + 1,
                currency_rank=rank().over(range=(None, 0), order_by=_.currency_ratio)
                + 1,
            )
            .filter((_.return_rank <= 10) | (_.currency_rank <= 10))
            .mutate(channel=lit("store"))
            .relocate("channel", before="item")
        )
        .drop("currency_ratio")
        .order_by(
            _[0].asc(nulls_first=True),
            _[3].asc(nulls_first=True),
            _[4].asc(nulls_first=True),
            _[1].asc(nulls_first=True),
        )
        .limit(100)
    )


@tpc_test("ds")
def test_50(store_sales, store_returns, store, date_dim):
    return (
        store_sales.join(
            store_returns,
            [
                ("ss_ticket_number", "sr_ticket_number"),
                ("ss_item_sk", "sr_item_sk"),
                ("ss_customer_sk", "sr_customer_sk"),
            ],
        )
        .join(store, [("ss_store_sk", "s_store_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(
            date_dim.view().filter(_.d_year == 2001, _.d_moy == 8),
            [("sr_returned_date_sk", "d_date_sk")],
        )
        .group_by(
            _.s_store_name,
            _.s_company_id,
            _.s_street_number,
            _.s_street_name,
            _.s_street_type,
            _.s_suite_number,
            _.s_city,
            _.s_county,
            _.s_state,
            _.s_zip,
        )
        .agg(
            {
                "30 days": ifelse(
                    _.sr_returned_date_sk - _.ss_sold_date_sk <= 30, 1, 0
                ).sum(),
                "31-60 days": ifelse(
                    (_.sr_returned_date_sk - _.ss_sold_date_sk > 30)
                    & (_.sr_returned_date_sk - _.ss_sold_date_sk <= 60),
                    1,
                    0,
                ).sum(),
                "61-90 days": ifelse(
                    (_.sr_returned_date_sk - _.ss_sold_date_sk > 60)
                    & (_.sr_returned_date_sk - _.ss_sold_date_sk <= 90),
                    1,
                    0,
                ).sum(),
                "91-120 days": ifelse(
                    (_.sr_returned_date_sk - _.ss_sold_date_sk > 90)
                    & (_.sr_returned_date_sk - _.ss_sold_date_sk <= 120),
                    1,
                    0,
                ).sum(),
                ">120 days": ifelse(
                    _.sr_returned_date_sk - _.ss_sold_date_sk > 120, 1, 0
                ).sum(),
            }
        )
        .order_by(~s.endswith(" days"))
        .limit(100)
    )


@tpc_test("ds")
def test_51(web_sales, date_dim, store_sales):
    web_v1 = (
        web_sales.join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .filter(_.d_month_seq.between(1200, 1200 + 11), _.ws_item_sk.notnull())
        .group_by(item_sk=_.ws_item_sk, d_date=_.d_date)
        .agg(total_sales_price=_.ws_sales_price.sum())
        .mutate(
            cume_sales=_.total_sales_price.sum().over(
                cumulative_window(group_by="item_sk", order_by="d_date")
            )
        )
    )
    store_v1 = (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .filter(_.d_month_seq.between(1200, 1200 + 11), _.ss_item_sk.notnull())
        .group_by(item_sk=_.ss_item_sk, d_date=_.d_date)
        .agg(total_sales_price=_.ss_sales_price.sum())
        .mutate(
            cume_sales=_.total_sales_price.sum().over(
                cumulative_window(group_by="item_sk", order_by="d_date")
            )
        )
    )
    return (
        web_v1.outer_join(
            store_v1,
            [web_v1.item_sk == store_v1.item_sk, web_v1.d_date == store_v1.d_date],
        )
        .select(
            item_sk=coalesce(web_v1.item_sk, store_v1.item_sk),
            d_date=coalesce(web_v1.d_date, store_v1.d_date),
            web_sales=web_v1.cume_sales,
            store_sales=store_v1.cume_sales,
        )
        .mutate(
            web_cumulative=_.web_sales.max().over(
                cumulative_window(group_by="item_sk", order_by="d_date")
            ),
            store_cumulative=_.store_sales.max().over(
                cumulative_window(group_by="item_sk", order_by="d_date")
            ),
        )
        .filter(_.web_cumulative > _.store_cumulative)
        .order_by(_.item_sk.asc(nulls_first=True), _.d_date.asc(nulls_first=True))
        .limit(100)
    )


@tpc_test("ds")
def test_52(date_dim, store_sales, item):
    return (
        date_dim.join(store_sales, [("d_date_sk", "ss_sold_date_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(_.i_manager_id == 1, _.d_moy == 11, _.d_year == 2000)
        .group_by(_.d_year, brand=_.i_brand, brand_id=_.i_brand_id)
        .agg(ext_price=_.ss_ext_sales_price.sum())
        .relocate("brand_id", before="brand")
        .order_by(_.d_year, _.ext_price.desc(), _.brand_id)
        .limit(100)
    )


@tpc_test("ds")
def test_53(item, store_sales, date_dim, store):
    return (
        item.join(store_sales, [("i_item_sk", "ss_item_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .filter(
            _.d_month_seq.isin(
                (
                    1200,
                    1200 + 1,
                    1200 + 2,
                    1200 + 3,
                    1200 + 4,
                    1200 + 5,
                    1200 + 6,
                    1200 + 7,
                    1200 + 8,
                    1200 + 9,
                    1200 + 10,
                    1200 + 11,
                )
            ),
            (
                _.i_category.isin(("Books", "Children", "Electronics"))
                & _.i_class.isin(("personal", "portable", "reference", "self-help"))
                & _.i_brand.isin(
                    (
                        "scholaramalgamalg #14",
                        "scholaramalgamalg #7",
                        "exportiunivamalg #9",
                        "scholaramalgamalg #9",
                    )
                )
            )
            | (
                _.i_category.isin(("Women", "Music", "Men"))
                & _.i_class.isin(("accessories", "classical", "fragrances", "pants"))
                & _.i_brand.isin(
                    (
                        "amalgimporto #1",
                        "edu packscholar #1",
                        "exportiimporto #1",
                        "importoamalg #1",
                    )
                )
            ),
        )
        .group_by(_.i_manufact_id, _.d_qoy)
        .agg(sum_sales=_.ss_sales_price.sum())
        .drop("d_qoy")
        .mutate(avg_quarterly_sales=_.sum_sales.mean().over(group_by="i_manufact_id"))
        .filter(
            ifelse(
                _.avg_quarterly_sales > 0,
                (_.sum_sales - _.avg_quarterly_sales).abs() / _.avg_quarterly_sales,
                null(),
            )
            > 0.1
        )
        .order_by(_.avg_quarterly_sales, _.sum_sales, _.i_manufact_id)
        .limit(100)
    )


@tpc_test("ds", result_is_empty=True)
def test_54(
    catalog_sales,
    web_sales,
    item,
    date_dim,
    customer,
    store_sales,
    customer_address,
    store,
):
    cs_or_ws_sales = catalog_sales.select(
        sold_date_sk="cs_sold_date_sk",
        customer_sk="cs_bill_customer_sk",
        item_sk="cs_item_sk",
    ).union(
        web_sales.select(
            sold_date_sk="ws_sold_date_sk",
            customer_sk="ws_bill_customer_sk",
            item_sk="ws_item_sk",
        )
    )
    my_customers = (
        cs_or_ws_sales.join(item, [("item_sk", "i_item_sk")])
        .join(date_dim, [("sold_date_sk", "d_date_sk")])
        .join(customer, [("customer_sk", "c_customer_sk")])
        .filter(
            _.i_category == "Women",
            _.i_class == "maternity",
            _.d_moy == 12,
            _.d_year == 1998,
        )
        .select("c_customer_sk", "c_current_addr_sk")
        .distinct()
    )
    my_revenue = (
        my_customers.join(store_sales, [("c_customer_sk", "ss_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(store, [("ca_county", "s_county"), ("ca_state", "s_state")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .filter(
            lambda t: t.d_month_seq.between(
                date_dim.view()
                .filter(_.d_year == 1998, _.d_moy == 12)
                .select(_.d_month_seq + 1)
                .distinct()
                .as_scalar(),
                date_dim.view()
                .filter(_.d_year == 1998, _.d_moy == 12)
                .select(_.d_month_seq + 3)
                .distinct()
                .as_scalar(),
            )
        )
        .group_by(_.c_customer_sk)
        .agg(revenue=_.ss_ext_sales_price.sum())
    )
    segments = my_revenue.select(SEGMENT=(_.revenue / 50).round().cast("int32"))
    return (
        segments.group_by("SEGMENT", segment_base=_.SEGMENT * 50)
        .agg(num_customers=_.count())
        .relocate("segment_base", after="num_customers")
        .order_by(
            _.SEGMENT.asc(nulls_first=True),
            _.num_customers.asc(nulls_first=True),
            _.segment_base,
        )
        .limit(100)
    )


@tpc_test("ds")
def test_55(date_dim, store_sales, item):
    return (
        date_dim.join(store_sales, [("d_date_sk", "ss_sold_date_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(_.i_manager_id == 28, _.d_moy == 11, _.d_year == 1999)
        .group_by(brand=_.i_brand, brand_id=_.i_brand_id)
        .agg(ext_price=_.ss_ext_sales_price.sum())
        .relocate("brand_id", before="brand")
        .order_by(_.ext_price.desc(), _.brand_id)
        .limit(100)
    )


@tpc_test("ds")
def test_56(store_sales, date_dim, customer_address, item, catalog_sales, web_sales):
    ss = (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(
            _.ca_gmt_offset == -5,
            _.d_moy == 2,
            _.d_year == 2001,
            lambda t: t.i_item_id.isin(
                item.filter(
                    _.i_color.isin(("slate", "blanched", "burnished"))
                ).i_item_id
            ),
        )
        .group_by(_.i_item_id)
        .agg(total_sales=_.ss_ext_sales_price.sum())
    )
    cs = (
        catalog_sales.join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("cs_bill_addr_sk", "ca_address_sk")])
        .join(item, [("cs_item_sk", "i_item_sk")])
        .filter(
            _.ca_gmt_offset == -5,
            _.d_moy == 2,
            _.d_year == 2001,
            lambda t: t.i_item_id.isin(
                item.filter(
                    _.i_color.isin(("slate", "blanched", "burnished"))
                ).i_item_id
            ),
        )
        .group_by(_.i_item_id)
        .agg(total_sales=_.cs_ext_sales_price.sum())
    )
    ws = (
        web_sales.join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ws_bill_addr_sk", "ca_address_sk")])
        .join(item, [("ws_item_sk", "i_item_sk")])
        .filter(
            _.ca_gmt_offset == -5,
            _.d_moy == 2,
            _.d_year == 2001,
            lambda t: t.i_item_id.isin(
                item.filter(
                    _.i_color.isin(("slate", "blanched", "burnished"))
                ).i_item_id
            ),
        )
        .group_by(_.i_item_id)
        .agg(total_sales=_.ws_ext_sales_price.sum())
    )
    return (
        ss.union(cs)
        .union(ws)
        .group_by(_.i_item_id)
        .agg(total_sales=_.total_sales.sum())
        .order_by(
            _.total_sales.asc(nulls_first=True), _.i_item_id.asc(nulls_first=True)
        )
        .limit(100)
    )


@tpc_test("ds")
@pytest.mark.notyet(
    ["clickhouse"],
    raises=ClickHouseDatabaseError,
    reason="clickhouse can't parse the baseline input SQL text for this query",
)
def test_57(item, catalog_sales, date_dim, call_center):
    v1 = (
        item.join(catalog_sales, [("i_item_sk", "cs_item_sk")])
        .join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .join(call_center, [("cs_call_center_sk", "cc_call_center_sk")])
        .filter(
            (_.d_year == 1999)
            | ((_.d_year == 1999 - 1) & (_.d_moy == 12))
            | ((_.d_year == 1999 + 1) & (_.d_moy == 1))
        )
        .group_by(_.i_category, _.i_brand, _.cc_name, _.d_year, _.d_moy)
        .agg(sum_sales=_.cs_sales_price.sum())
        .mutate(
            avg_monthly_sales=_.sum_sales.mean().over(
                group_by=("i_category", "i_brand", "cc_name", "d_year")
            ),
            rn=rank().over(
                group_by=("i_category", "i_brand", "cc_name"),
                order_by=("d_year", "d_moy"),
            ),
        )
    )

    v1_lag = v1.view()
    v1_lead = v1.view()
    v2 = (
        v1.join(v1_lag, ["i_category", "i_brand", "cc_name", v1.rn == v1_lag.rn + 1])
        .join(v1_lead, ["i_category", "i_brand", "cc_name", v1.rn == v1_lead.rn - 1])
        .select(
            v1.i_category,
            v1.i_brand,
            v1.cc_name,
            v1.d_year,
            v1.d_moy,
            v1.avg_monthly_sales,
            v1.sum_sales,
            psum=v1_lag.sum_sales,
            nsum=v1_lead.sum_sales,
        )
    )
    return (
        v2.filter(
            _.d_year == 1999,
            _.avg_monthly_sales > 0,
            ifelse(
                _.avg_monthly_sales > 0,
                (_.sum_sales - _.avg_monthly_sales).abs() / _.avg_monthly_sales,
                null(),
            )
            > 0.1,
        )
        .order_by((_.sum_sales - _.avg_monthly_sales).asc(nulls_first=True), s.r[1:10])
        .limit(100)
    )


@tpc_test("ds")
def test_58(store_sales, item, date_dim, catalog_sales, web_sales):
    date_filter = lambda t: t.d_date.isin(
        date_dim.filter(
            lambda dd: dd.d_week_seq.isin(
                date_dim.filter(_.d_date == date("2000-01-03")).d_week_seq
            )
        ).d_date
    )
    ss_items = (
        store_sales.join(item, [("ss_item_sk", "i_item_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .filter(date_filter)
        .group_by(item_id=_.i_item_id)
        .agg(ss_item_rev=_.ss_ext_sales_price.sum())
    )
    cs_items = (
        catalog_sales.join(item, [("cs_item_sk", "i_item_sk")])
        .join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .filter(date_filter)
        .group_by(item_id=_.i_item_id)
        .agg(cs_item_rev=_.cs_ext_sales_price.sum())
    )
    ws_items = (
        web_sales.join(item, [("ws_item_sk", "i_item_sk")])
        .join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .filter(date_filter)
        .group_by(item_id=_.i_item_id)
        .agg(ws_item_rev=_.ws_ext_sales_price.sum())
    )
    return (
        ss_items.join(cs_items, "item_id")
        .join(ws_items, "item_id")
        .filter(
            _.ss_item_rev.between(0.9 * _.cs_item_rev, 1.1 * _.cs_item_rev),
            _.ss_item_rev.between(0.9 * _.ws_item_rev, 1.1 * _.ws_item_rev),
            _.cs_item_rev.between(0.9 * _.ss_item_rev, 1.1 * _.ss_item_rev),
            _.cs_item_rev.between(0.9 * _.ws_item_rev, 1.1 * _.ws_item_rev),
            _.ws_item_rev.between(0.9 * _.ss_item_rev, 1.1 * _.ss_item_rev),
            _.ws_item_rev.between(0.9 * _.cs_item_rev, 1.1 * _.cs_item_rev),
        )
        .select(
            ss_items.item_id,
            _.ss_item_rev,
            ss_dev=(
                _.ss_item_rev
                / ((_.ss_item_rev + _.cs_item_rev + _.ws_item_rev) / 3)
                * 100
            ),
            cs_item_rev=_.cs_item_rev,
            cs_dev=(
                _.cs_item_rev
                / ((_.ss_item_rev + _.cs_item_rev + _.ws_item_rev) / 3)
                * 100
            ),
            ws_item_rev=_.ws_item_rev,
            ws_dev=(
                _.ws_item_rev
                / ((_.ss_item_rev + _.cs_item_rev + _.ws_item_rev) / 3)
                * 100
            ),
            average=(_.ss_item_rev + _.cs_item_rev + _.ws_item_rev) / 3,
        )
        .order_by(
            ss_items.item_id.asc(nulls_first=True), _.ss_item_rev.asc(nulls_first=True)
        )
        .limit(100)
    )


@tpc_test("ds")
def test_59(store_sales, date_dim, store):
    days = [(cal.day_abbr[i].lower(), cal.day_name[i]) for i in range(-1, 6)]

    wss = (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .group_by(_.d_week_seq, _.ss_store_sk)
        .agg(
            {
                f"{abbr}_sales": _.ss_sales_price.sum(where=_.d_day_name == day)
                for abbr, day in days
            }
        )
    )
    y = (
        wss.join(store, [("ss_store_sk", "s_store_sk")])
        .join(date_dim, "d_week_seq")
        .filter(_.d_month_seq.between(1212, 1212 + 11))
        .mutate(_.s_store_name, _.d_week_seq, _.s_store_id, s.endswith("sales"))
        .rename("{name}1")
    )
    x = (
        wss.join(store, [("ss_store_sk", "s_store_sk")])
        .join(date_dim, "d_week_seq")
        .filter(_.d_month_seq.between(1212 + 12, 1212 + 23))
        .mutate(_.s_store_name, _.d_week_seq, _.s_store_id, s.endswith("sales"))
        .rename("{name}2")
    )
    return (
        y.join(x, [("s_store_id1", "s_store_id2"), y.d_week_seq1 == x.d_week_seq2 - 52])
        .select(
            _.s_store_name1,
            _.s_store_id1,
            _.d_week_seq1,
            **{
                f"{abbr}_sales_ratio": _[f"{abbr}_sales1"] / _[f"{abbr}_sales2"]
                for abbr in map(itemgetter(0), days)
            },
        )
        .order_by(s.across(~s.endswith("_ratio"), _.asc(nulls_first=True)))
        .limit(100)
    )


@tpc_test("ds")
def test_60(store_sales, date_dim, customer_address, item, catalog_sales, web_sales):
    filters = (
        _.i_item_id.isin(item.filter(lambda i: i.i_category == "Music").i_item_id),
        _.d_year == 1998,
        _.d_moy == 9,
        _.ca_gmt_offset == -5,
    )
    ss = (
        store_sales.join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ss_addr_sk", "ca_address_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(*filters)
        .group_by(_.i_item_id)
        .agg(total_sales=_.ss_ext_sales_price.sum())
    )
    cs = (
        catalog_sales.join(date_dim, [("cs_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("cs_bill_addr_sk", "ca_address_sk")])
        .join(item, [("cs_item_sk", "i_item_sk")])
        .filter(*filters)
        .group_by(_.i_item_id)
        .agg(total_sales=_.cs_ext_sales_price.sum())
    )
    ws = (
        web_sales.join(date_dim, [("ws_sold_date_sk", "d_date_sk")])
        .join(customer_address, [("ws_bill_addr_sk", "ca_address_sk")])
        .join(item, [("ws_item_sk", "i_item_sk")])
        .filter(*filters)
        .group_by(_.i_item_id)
        .agg(total_sales=_.ws_ext_sales_price.sum())
    )
    return (
        ss.union(cs)
        .union(ws)
        .group_by(_.i_item_id)
        .agg(total_sales=_.total_sales.sum())
        .order_by(s.all())
        .limit(100)
    )


@tpc_test("ds")
def test_61(store_sales, store, promotion, date_dim, customer, customer_address, item):
    promotional_sales = (
        store_sales.join(store, [("ss_store_sk", "s_store_sk")])
        .join(promotion, [("ss_promo_sk", "p_promo_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(
            _.ca_gmt_offset == -5,
            _.i_category == "Jewelry",
            (_.p_channel_dmail == "Y")
            | (_.p_channel_email == "Y")
            | (_.p_channel_tv == "Y"),
            _.s_gmt_offset == -5,
            _.d_year == 1998,
            _.d_moy == 11,
        )
        .agg(promotions=_.ss_ext_sales_price.sum())
    )
    all_sales = (
        store_sales.join(store, [("ss_store_sk", "s_store_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(customer, [("ss_customer_sk", "c_customer_sk")])
        .join(customer_address, [("c_current_addr_sk", "ca_address_sk")])
        .join(item, [("ss_item_sk", "i_item_sk")])
        .filter(
            _.ca_gmt_offset == -5,
            _.i_category == "Jewelry",
            _.s_gmt_offset == -5,
            _.d_year == 1998,
            _.d_moy == 11,
        )
        .agg(total=_.ss_ext_sales_price.sum())
    )
    return (
        promotional_sales.cross_join(all_sales)
        .mutate(
            perc_promotions=(
                _.promotions.cast("decimal(15, 4)")
                / _.total.cast("decimal(15, 4)")
                * 100
            )
        )
        .order_by(_.promotions, _.total)
        .limit(100)
    )


@tpc_test("ds")
def test_62(web_sales, warehouse, ship_mode, web_site, date_dim):
    return (
        web_sales.join(
            warehouse.mutate(w_substr=_.w_warehouse_name[:20]),
            [("ws_warehouse_sk", "w_warehouse_sk")],
        )
        .join(ship_mode, [("ws_ship_mode_sk", "sm_ship_mode_sk")])
        .join(web_site, [("ws_web_site_sk", ("web_site_sk"))])
        .join(date_dim, [("ws_ship_date_sk", "d_date_sk")])
        .filter(_.d_month_seq.between(1200, 1200 + 11))
        .group_by(_.w_substr, _.sm_type, _.web_name)
        .agg(
            # MEH
            **{
                f"{name} days": ifelse(
                    (
                        (_.ws_ship_date_sk - _.ws_sold_date_sk > lower)
                        if lower is not None
                        else True
                    )
                    & (
                        (_.ws_ship_date_sk - _.ws_sold_date_sk <= upper)
                        if upper is not None
                        else True
                    ),
                    1,
                    0,
                ).sum()
                for name, lower, upper in [
                    ("30", None, 30),
                    ("31-60", 30, 60),
                    ("61-90", 60, 90),
                    ("91-120", 90, 120),
                    (">120", 120, None),
                ]
            }
        )
        .order_by(s.across(~s.endswith(" days"), _.asc(nulls_first=True)))
    )


@tpc_test("ds")
def test_63(item, store_sales, date_dim, store):
    return (
        item.join(store_sales, [("i_item_sk", "ss_item_sk")])
        .join(date_dim, [("ss_sold_date_sk", "d_date_sk")])
        .join(store, [("ss_store_sk", "s_store_sk")])
        .filter(
            _.d_month_seq.isin(tuple(range(1200, 1212))),
            (
                _.i_category.isin(("Books", "Children", "Electronics"))
                & _.i_class.isin(("personal", "portable", "reference", "self-help"))
                & _.i_brand.isin(
                    (
                        "scholaramalgamalg #14",
                        "scholaramalgamalg #7",
                        "exportiunivamalg #9",
                        "scholaramalgamalg #9",
                    )
                )
            )
            | (
                _.i_category.isin(("Women", "Music", "Men"))
                & _.i_class.isin(("accessories", "classical", "fragrances", "pants"))
                & _.i_brand.isin(
                    (
                        "amalgimporto #1",
                        "edu packscholar #1",
                        "exportiimporto #1",
                        "importoamalg #1",
                    )
                )
            ),
        )
        .group_by(_.i_manager_id, _.d_moy)
        .agg(sum_sales=_.ss_sales_price.sum())
        .drop("d_moy")
        .mutate(avg_monthly_sales=_.sum_sales.mean().over(group_by=_.i_manager_id))
        .filter(
            ifelse(
                _.avg_monthly_sales > 0,
                (_.sum_sales - _.avg_monthly_sales).abs() / _.avg_monthly_sales,
                null(),
            )
            > 0.1
        )
        .order_by(_.i_manager_id, _.avg_monthly_sales, _.sum_sales)
        .limit(100)
    )
