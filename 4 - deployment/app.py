import streamlit as st

from pathlib import Path

# === Global App Config === #
st.set_page_config(
    page_title="Automation Feasibility System",
    page_icon="🤖",
    layout="wide",
)

# === Define pages explicitly === #
current_folder = Path(__file__).resolve().parent
pages_folder = current_folder / "pages"

pages = {
    "": [
        st.Page(pages_folder / "1_Homepage.py", title="Homepage"),
        st.Page(pages_folder / "2_Predictor.py", title="Automation Feasibility Predictor"),
        st.Page(pages_folder / "3_Dashboard.py", title="Analytics Dashboard"),
    ]
}

st.navigation(pages)

# === Global header (shared) === #
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #2D3142;
    }

    [data-testid="stSidebarContent"] {
        background-color: #2D3142;
        opacity: 0.5;
    }

    header[data-testid="stHeader"] {
    }
    header[data-testid="stHeader"]::after {
        content: "Automation Feasibility System";
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-family: 'Verdana';
        font-size: 25px;
        font-weight: 600;
    }

        /* Reduce space below the top header */
    header[data-testid="stHeader"] {
        margin-bottom: 0px;
    }

    /* Pull the main content upward */
    section[data-testid="stMain"] > div:first-child {
        padding-top: 4.2rem;
    }

    /* Remove extra padding on first block */
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

pg = st.navigation(pages)
pg.run()

# Running `py -m streamlit run "4 - deployment/app.py"` will show the pages in the sidebar.
