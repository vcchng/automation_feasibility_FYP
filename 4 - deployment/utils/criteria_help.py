
# === High-level homepage text === #

SYSTEM_OVERVIEW = (
    "This web application helps you evaluate whether a business task is a good candidate for automation. "
    "By selecting or entering a few details about the task (e.g., how complex it is, how often it happens, "
    "how long it takes, and how error-prone it is). The system then predicts the automation feasibility "
    "and provides a confidence-like feasibility percentage."
)

HOW_TO_USE = [
    "1) Go to the Automation Feasibility Predictor page.",
    "2) Fill in all required task criteria.",
    "3) Click 'Predict feasibility' to view the result.",
    "4) (Optional) Review the Analytics Dashboard for aggregated insights.",
]

# === Criteria explanations === #

CRITERIA_HELP = {
    "Task Name": (
        "The type of task being evaluated (e.g., Invoice Processing, Email Reply, Report Generation, Approval). "
        "Use a short, clear name that matches what the task actually is."
    ),

    "Time Taken (mins)": (
        "Actual time to complete the task (in minutes). "
        "In the model pipeline, this may be standardized internally, but users should input the real time value."
    ),

    "Complexity (1-5)": (
        "A numerical complexity score from 1 to 5 (1 = very simple, 5 = very complex). "
        "Higher complexity usually means more rules/exceptions, making automation harder."
    ),

    "Frequency": (
        "How often the task is performed (e.g., Daily, Weekly, Monthly, Quarterly, Ad-Hoc). "
        "Frequent tasks typically give higher value when automated."
    ),

    "Tool Used": (
        "The current tool or method used to perform the task (e.g., RPA Tool, Power App, Excel Macro, etc.)."
    ),

    "Department": (
        "Which business unit is responsible for performing the task (e.g., Finance, HR, IT, Operations)."
    ),

    "Error Rate (1-10)": (
        "A numerical error rate score when the task is done manually on a scale from 1 to 10 (1= zero or a small amount of mistakes, 10= constant or a very high amount of mistakes). "
        "Higher error rates may indicate strong benefits from automation, but can also reflect unstable processes."
    ),

    "Rule-Based Indicator": (
        "Whether the task can be executed using predefined rules without human judgement. "
        "Examples: if-else decisions, clear approval thresholds, validation rules. (Yes/No)"
    ),

    "Process Stability": (
        "How consistent and repeatable the process is over time (Low / Medium / High). "
        "High Stability processes are generally easier to automate."
    ),

    "Data Structure": (
        "The format of data used in the task: Structured / Semi-Structured / Unstructured. "
        "Structured data (tables, databases) is easiest to automate; unstructured (free text, scanned docs) is harder."
    ),

    "Automation Suitable": (
        "The target label used to train the model: whether the task is appropriate for automation (Yes/No). "
        "The model predicts this label and the system displays it for the user."
    ),
    
        "Feasibility (%)": (
        "The predicted likelihood (expressed as a percentage) that the task is suitable for automation. "
        "This value is derived from XGBoost model's probability output and reflects the model's "
        "confidence in its prediction, not a guaranteed outcome."
    ),

    "Model Used": (
        "The machine learning model used to generate the prediction. "
        "An XGBoost classifier is used in this system because it performs well on structured tabular data, "
        "is capable of managing mixed feature types, and is good at capturing complex, non-linear correlations between task attributes and automation feasibility. "
    ),
}

CRITERIA_DISPLAY_ORDER = [
    "Task Name",
    "Department",
    "Frequency",
    "Tool Used",
    "Complexity (1-5)",
    "Time Taken (mins)",
    "Error Rate (1-10)",
    "Rule-Based Indicator",
    "Process Stability",
    "Data Structure",
]

ALIASES = {
    "Time": "Time Taken (mins)",
    "Time Taken": "Time Taken (mins)",
    "Error Rate (%)": "Error Rate (1-10)",
    "Rule Based Indicator": "Rule-Based Indicator",
    "Rule-based Indicator": "Rule-Based Indicator",
}

def get_help(label: str) -> str:
    if label in CRITERIA_HELP:
        return CRITERIA_HELP[label]
    if label in ALIASES and ALIASES[label] in CRITERIA_HELP:
        return CRITERIA_HELP[ALIASES[label]]
    return ""
