import streamlit as st

from utils.criteria_help import SYSTEM_OVERVIEW, HOW_TO_USE, CRITERIA_HELP, CRITERIA_DISPLAY_ORDER

st.title("Homepage")

# === System overview (what this system is) === #
st.markdown("## What is an Automation Feasiblity System?")
st.write(SYSTEM_OVERVIEW)

st.markdown("## How to use the system")
for step in HOW_TO_USE:
    st.write(step)

st.markdown("---")

# === Criteria explanations === #
st.markdown("## Task criteria")
st.write(
    "On the Automation Feasiblity Predictor page, you will be asked to fill in several criteria about the task you are interested in automating. "
    "Below explains what each criterion means."
)

for key in CRITERIA_DISPLAY_ORDER:
    if key in CRITERIA_HELP:
        with st.expander(key, expanded=False):
            st.write(CRITERIA_HELP[key])

extra_keys = [k for k in CRITERIA_HELP.keys() if k not in CRITERIA_DISPLAY_ORDER]
if extra_keys:
    st.markdown("### Outcome Labels")
    st.write("The labels below will be displayed when a prediction is made.")
    for key in sorted(extra_keys):
        with st.expander(key, expanded=False):
            st.write(CRITERIA_HELP[key])

st.markdown("---")

# === Button to go to Predictor === #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Go to Automation Feasibility Predictor ➜", use_container_width=True):
        st.switch_page("pages/2_Predictor.py")
