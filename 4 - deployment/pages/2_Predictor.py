from __future__ import annotations

from pathlib import Path
from datetime import datetime

import joblib
import json
import pandas as pd
import streamlit as st
import gspread
import google.generativeai as genai

from oauth2client.service_account import ServiceAccountCredentials
from utils.criteria_help import get_help

# === Paths (robust relative paths) === #
PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = APP_DIR / "xgboost_model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

BASELINE_PATH = PROJECT_ROOT / "0 - data" / "train_clean.xlsx"

# === Google Sheets connector === #
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = st.secrets["google_credentials"]
creds = ServiceAccountCredentials.from_json_keyfile_name(creds_dict, scope)
client = gspread.authorize(creds)

sheet = client.open("new_requests").sheet1

LOG_DIR = PROJECT_ROOT / "0 - data"

MODEL_USED = "XGBoost (Tuned Model)"

TARGET_COL = "Automation Suitable"
DROP_COLS = ["Task ID"]

# Columns that must be standardized before sending to model
COLS_TO_STANDARDIZE = ["Time Taken (mins)", "Error Rate (%)"]

# === Google AI Studio path === #
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
AImodel = genai.GenerativeModel('gemini-flash-latest')

# === Helpers functions === #
@st.cache_resource
def load_tuned_model(bundle_path: Path):
    bundle = joblib.load(bundle_path)

    if not isinstance(bundle, dict):
        raise ValueError("Expected a bundle dict with keys like 'base' and 'tuned'.")

    if "tuned" not in bundle:
        raise KeyError("Bundle does not contain a 'tuned' model key.")

    return bundle["tuned"]

@st.cache_resource
def load_scaler(path: Path):
    return joblib.load(path)

@st.cache_data
def load_baseline(path: Path):
    return pd.read_excel(path)

TASK_NAME_COL = "Task Name"

def compute_feasibility_pct(model, X_one: pd.DataFrame):
    raw_pred = model.predict(X_one)[0]

    # ---- Normalize predicted label ----
    if raw_pred in [1, "Yes"]:
        pred_label = "Yes"
    elif raw_pred in [0, "No"]:
        pred_label = "No"
    else:
        pred_label = str(raw_pred)

    feasibility_pct = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_one)[0]

        yes_prob = None
        classes = list(model.classes_)

        if "Yes" in classes:
            yes_prob = float(proba[classes.index("Yes")])
        elif 1 in classes:
            yes_prob = float(proba[classes.index(1)])

        if yes_prob is not None:
            feasibility_pct = round(yes_prob * 100, 2)

    return pred_label, feasibility_pct

def standardize_user_inputs(user_inputs: dict, scaler) -> dict:
    """
    Takes user_inputs (raw values for time/error) and returns a new dict where
    Time Taken (mins) and Error Rate (%) are standardized using the saved scaler.
    The returned dict is what we send to the model AND what we log to CSV.
    """
    inputs = dict(user_inputs)  # copy

    # Ensure the two columns exist before transforming
    for c in COLS_TO_STANDARDIZE:
        if c not in inputs:
            raise KeyError(f"Missing required column for standardization: {c}")

    raw_time = float(inputs["Time Taken (mins)"])
    raw_err = float(inputs["Error Rate (%)"])

    scaled = scaler.transform([[raw_time, raw_err]])
    inputs["Time Taken (mins)"] = float(scaled[0][0])
    inputs["Error Rate (%)"] = float(scaled[0][1])

    return inputs

# === Page UI === #
st.title("Automation Feasibility Predictor")
st.caption("Fill in the task criteria to predict whether it is suitable for automation.")

# Load model, scaler & baseline
try:
    model = load_tuned_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load tuned model from:\n{MODEL_PATH}\n\n{e}")
    st.stop()

try:
    scaler = load_scaler(SCALER_PATH)
except Exception as e:
    st.error(
        f"Failed to load scaler from:\n{SCALER_PATH}\n\n"
        "Make sure you saved the fitted scaler (train-only) to 4 - deployment/scaler.pkl.\n\n"
        f"{e}"
    )
    st.stop()

try:
    baseline_df = load_baseline(BASELINE_PATH)
except Exception as e:
    st.error(
        "Failed to load the baseline TRAIN dataset from:\n"
        f"{BASELINE_PATH}\n\n"
        "Make sure train_clean.xlsx exists in 0 - data.\n\n"
        f"{e}"
    )
    st.stop()

drop_existing = [c for c in DROP_COLS if c in baseline_df.columns]

exclude_from_model = drop_existing + [TARGET_COL]
if TASK_NAME_COL in baseline_df.columns:
    exclude_from_model.append(TASK_NAME_COL)

feature_cols = [c for c in baseline_df.columns if c not in exclude_from_model]

cat_cols = baseline_df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
num_cols = baseline_df[feature_cols].select_dtypes(exclude=["object"]).columns.tolist()

with st.form("predict_form"):
    left, right = st.columns(2)
    user_inputs = {}

    # -------- Categorical --------
    with left:
        st.markdown("### Categorical Inputs")

        task_name = st.text_input(
            "Task Name",
            placeholder="e.g., Monthly invoice reconciliation",
            help="Used for logging only (not used by the prediction model)."
        )

        for col in cat_cols:
            options = sorted(baseline_df[col].dropna().unique().tolist())
            if options:
                user_inputs[col] = st.selectbox(col, options, help=get_help(col))
            else:
                user_inputs[col] = st.text_input(col, help=get_help(col))

    # -------- Numeric --------
    with right:
        st.markdown("### Numeric Inputs")
        for col in num_cols:
            if col == "Time Taken (mins)":
                user_inputs[col] = st.number_input(
                    col, min_value=0.0, max_value=600.0, value=30.0, step=1.0,
                    help=get_help(col)
                )

            elif col == "Complexity (1-5)":
                user_inputs[col] = st.number_input(
                    col,
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1,
                    format="%.1f",
                    help=get_help(col),
                )

            elif col == "Error Rate (%)":
                user_inputs[col] = st.number_input(
                    "Error Rate (1-10)",
                    min_value=1.0,
                    max_value=10.0,
                    value=5.0,
                    step=0.1,
                    format="%.1f",
                    help=get_help(col),
                )

            else:
                col_min = float(baseline_df[col].min())
                col_max = float(baseline_df[col].max())
                col_med = float(baseline_df[col].median())
                pad = (col_max - col_min) * 0.05 if col_max != col_min else 1.0

                user_inputs[col] = st.number_input(
                    col,
                    value=col_med,
                    min_value=col_min - pad,
                    max_value=col_max + pad,
                    help=get_help(col),
                )

    st.markdown("---")
    task_desc = st.text_area(
        "Optional: Task description (explain you proposed automation to receive an AI generated response according to the prediction and feasibility %.",
        placeholder="Briefly describe the task and any key rules/inputs/outputs...",
    )

    submitted = st.form_submit_button("Predict feasibility")

# === Prediction + Logging === #
if submitted:
    try:
        model_inputs = standardize_user_inputs(user_inputs, scaler)

        X_one = pd.DataFrame([model_inputs], columns=feature_cols)

        pred_label, feasibility_pct = compute_feasibility_pct(model, X_one)

    except Exception as e:
        st.error(
            "Prediction failed.\n\n"
            "Common causes:\n"
            "- Feature mismatch between deployment inputs and training features\n"
            "- scaler.pkl not matching the training-time scaler\n"
            "- Missing required columns\n\n"
            f"Error:\n{e}"
        )
        st.stop()

    st.subheader("Prediction Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Label", str(pred_label))
    c2.metric("Feasibility (%)", "N/A" if feasibility_pct is None else f"{feasibility_pct}%")
    c3.metric("Model Used", MODEL_USED)
    task_name_value = (task_name or "").strip()

    log_row = {
    "Timestamp": datetime.now().isoformat(timespec="seconds"),
    "Task Name": (task_name or "").strip()
    }

    log_row.update(user_inputs)

    log_row["Predicted Label"] = pred_label
    log_row["Feasibility (%)"] = feasibility_pct
    log_row["Model Used"] = MODEL_USED

    log_df = pd.DataFrame([log_row])

    st.write("Logging the following data:")
    st.write(log_df)

    log_values = log_df.values.tolist()
    
    try:
        sheet.append_row(log_values[0])
        st.success(f"Saved log to: new_requests.csv")
    except Exception as e:
        print(f"Prediction worked, but logging failed: {e}")

    st.markdown("---")

    # === Google AI Structured Prompt === #
    structured_prompt = f"""
        ROLE: Automation Consultant
        INPUT DATA:
        - Idea: {task_desc}
        - Feasibility Score: {feasibility_pct}%
        - Decision: {pred_label}

        You are an expert Automation Consultant. Your task is to analyze a proposed automation project based on three inputs:
            1. Feasibility Score %: A technical confidence score.
            2. Predicted Label (Decision): A "Yes" or "No" classification.
            3. Proposed Idea (Idea): The user's description of the task.

        Your Goal: Provide a detailed, professional explanation justifying the "Predicted Label."
            - If the label is Yes, highlight the ROI, efficiency gains, and why the feasibility score supports this.
            - If the label is No, explain the potential risks, technical blockers, or why the complexity might not be worth the effort.

        Always align your reasoning with the provided Feasibility percentage.
        """
    
    response = AImodel.generate_content(structured_prompt)

    st.write("### AI Analysis")
    st.write(response.text)
    st.markdown("---")
