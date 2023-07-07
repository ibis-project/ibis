from __future__ import annotations

import requests
import streamlit as st

from ibis import _
from ibis.streamlit import IbisConnection

st.set_page_config(page_title="Yummy Data", layout="wide")
st.title("Yummy Data :bacon:")


@st.cache_data
def get_emoji():
    resp = requests.get(
        "https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json"
    )
    resp.raise_for_status()
    emojis = resp.json()
    return emojis


options = [1, 5, 10, 25, 50, 100]


@st.cache_data
def query():
    return (
        con.tables.recipes.relabel("snake_case")
        .mutate(ner=_.ner.map(lambda n: n.lower()).unnest())
        .ner.topk(max(options))
        .relabel(dict(ner="ingredient"))
        .to_pandas()
        .assign(
            emoji=lambda df: df.ingredient.map(
                lambda emoji: f"{emojis.get(emoji, '-')}"
            )
        )
        .set_index("ingredient")
    )


emojis = get_emoji()

con = st.experimental_connection("ch", type=IbisConnection)

if n := st.radio("Ingredients", options, index=1, horizontal=True):
    table, whole = st.columns((2, 1))
    idx = options.index(n)
    k = 0
    base = query()
    for m in options[: idx + 1]:
        df = base.iloc[k:m]
        if not k:
            word = "first"
        elif m < n:
            word = "next"
        else:
            word = "last"

        uniq_emojis = " ".join(df.emoji[df.emoji != "-"].unique())
        table.header(f"{word.title()} {m - k:d}")
        table.subheader(uniq_emojis)

        table.dataframe(df, use_container_width=True)
        k = m

    b = base.iloc[:n]
    uniq_emojis = " ".join(b.emoji[b.emoji != "-"].unique())
    whole.header(f"Top {n:d}")
    whole.subheader(uniq_emojis)
    whole.dataframe(b, use_container_width=True)
