from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Automation Feasibility Analytics Dashboard", layout="wide")

st.title("Automation Feasibility Analytics Dashboard")
st.caption("The dashboard below provides auotmation feasiblity insights into the submissions made using this model. Additionally, it displays the data used to train the model.")

# === Power BI settings === #
POWER_BI_EMBED_URL = st.secrets("bi_dashboard")

DASHBOARD_HEIGHT = 800

# === Embed Power BI === #
if POWER_BI_EMBED_URL and POWER_BI_EMBED_URL.startswith("http"):
    components.iframe(POWER_BI_EMBED_URL, width="90%", height=DASHBOARD_HEIGHT, scrolling=True)
else:
    st.info("Paste a valid Power BI report URL (starts with https://...) to display the dashboard.")

st.markdown("---")