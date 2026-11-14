"""Engine health monitor: interactive demo of the final model.

Run from improved_implementation/:

    streamlit run app/streamlit_app.py

The app trains the final model (logistic regression, 40-cycle rolling features) on the 100
training engines at start-up, then scores the 100 unseen test engines cycle by cycle.
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pm.common import SENSORS, add_rolling, load_config, load_raw  # noqa: E402
from pm.models import make_classifier  # noqa: E402

WINDOW = 40           # rolling window of the final model
SEED = 42
SENSOR_NAMES = {
    "s2": "LPC outlet temperature", "s3": "HPC outlet temperature", "s4": "LPT outlet temperature",
    "s7": "HPC outlet pressure", "s8": "Physical fan speed", "s9": "Physical core speed",
    "s11": "HPC outlet static pressure", "s12": "Fuel flow / HPC pressure ratio",
    "s13": "Corrected fan speed", "s14": "Corrected core speed", "s15": "Bypass ratio",
    "s17": "Bleed enthalpy", "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}

st.set_page_config(page_title="Engine Health Monitor", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Training the model on 100 engines...")
def load_everything():
    cfg = load_config()
    cfg["data"]["roll_window"] = WINDOW
    cfg["data"]["kinds"] = ["raw", "mean", "std"]
    train, test = load_raw(cfg)
    train, test = add_rolling(train, SENSORS, WINDOW), add_rolling(test, SENSORS, WINDOW)
    model = make_classifier("logistic_regression", cfg, SEED).fit(train, train["label"])
    test = test.copy()
    test["probability"] = model.predict_proba(test)[:, 1]
    return test, cfg["data"]["fail_window"]


test, FAIL_WINDOW = load_everything()
last = test.loc[test.groupby("unit")["cycle"].idxmax()].set_index("unit")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
# optional URL parameters, e.g. ?engine=77&view=fleet, so a particular view can be linked
_q = st.query_params
_start = int(_q["engine"]) if _q.get("engine", "").isdigit() and 1 <= int(_q["engine"]) <= 100 else 34
unit = st.sidebar.selectbox("Test engine", sorted(test["unit"].unique()), index=_start - 1)
threshold = st.sidebar.slider("Alert threshold (probability)", 0.05, 0.95, 0.50, 0.05,
                              help="Raise an alert when the predicted probability is at or above this value.")
sensor = st.sidebar.selectbox("Sensor to plot", list(SENSOR_NAMES),
                              index=list(SENSOR_NAMES).index("s11"),
                              format_func=lambda s: f"{s}: {SENSOR_NAMES[s]}")
reveal = st.sidebar.checkbox("Show the true answer", value=True,
                             help="The true remaining life comes from NASA's answer file. "
                                  "The model never sees it.")
st.sidebar.caption(f"Model: logistic regression on raw readings plus {WINDOW}-cycle rolling mean "
                   f"and standard deviation of 14 sensors. Trained on 100 engines run to failure.")

# ---------------------------------------------------------------- header
st.title("Engine Health Monitor")
st.caption(f"Will this turbofan engine fail within the next {FAIL_WINDOW} cycles? "
           "NASA C-MAPSS FD001, 100 unseen test engines.")

tab_engine, tab_fleet = st.tabs(["One engine", "Whole fleet"],
                                 default="Whole fleet" if _q.get("view") == "fleet" else None)

# ---------------------------------------------------------------- one engine
with tab_engine:
    eng = test[test["unit"] == unit]
    row = last.loc[unit]
    alert = row["probability"] >= threshold

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cycles observed", int(row["cycle"]))
    c2.metric(f"Probability of failure within {FAIL_WINDOW} cycles", f"{row['probability']:.0%}")
    c3.metric("Decision", "INSPECT NOW" if alert else "Keep flying")
    if reveal:
        c4.metric("True cycles left", int(row["RUL"]))
    else:
        c4.metric("True cycles left", "hidden")

    if reveal:
        truth = bool(row["label"])
        if alert and truth:
            st.success(f"Correct alert: engine {unit} really fails in {int(row['RUL'])} cycles.")
        elif not alert and not truth:
            st.success(f"Correct: engine {unit} still has {int(row['RUL'])} cycles left, no alert needed.")
        elif alert and not truth:
            st.warning(f"False alarm: engine {unit} has {int(row['RUL'])} cycles left, "
                       f"which is more than {FAIL_WINDOW}.")
        else:
            st.error(f"Missed failure: engine {unit} fails in {int(row['RUL'])} cycles and was not flagged.")

    st.subheader("Predicted probability over the engine's life so far")
    base = alt.Chart(eng).encode(x=alt.X("cycle:Q", title="Cycle (flight number)"))
    prob = base.mark_line(color="#1f5fa8", strokeWidth=2.5).encode(
        y=alt.Y("probability:Q", title="Probability of failure soon",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=["cycle", alt.Tooltip("probability:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="#c0392b", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y": [threshold], "x": [eng["cycle"].min()],
                                    "t": [f"alert threshold {threshold:.2f}"]})).mark_text(
        align="left", dy=-8, color="#c0392b").encode(x="x:Q", y="y:Q", text="t:N")
    st.altair_chart((prob + rule + label).properties(height=300), width="stretch")

    st.subheader(f"Sensor {sensor}: {SENSOR_NAMES[sensor]}")
    long = eng[["cycle", sensor, f"{sensor}_mean{WINDOW}"]].rename(
        columns={sensor: "Raw reading", f"{sensor}_mean{WINDOW}": f"{WINDOW}-cycle rolling mean"}
    ).melt("cycle", var_name="Series", value_name="value")
    sens = alt.Chart(long).mark_line().encode(
        x=alt.X("cycle:Q", title="Cycle (flight number)"),
        y=alt.Y("value:Q", title="Sensor value", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Raw reading", f"{WINDOW}-cycle rolling mean"],
                                        range=["#9aa5b1", "#d35400"])),
        strokeWidth=alt.condition(alt.datum.Series == "Raw reading", alt.value(1.2), alt.value(2.8)),
        tooltip=["cycle", "Series", alt.Tooltip("value:Q", format=".3f")])
    st.altair_chart(sens.properties(height=280), width="stretch")
    st.caption("The raw reading is noisy. The rolling mean shows the slow drift, and the spread of "
               "readings inside the window (the rolling standard deviation) tells the model how fast "
               "the engine is degrading.")

# ---------------------------------------------------------------- fleet
with tab_fleet:
    fleet = last.reset_index()[["unit", "cycle", "probability", "RUL", "label"]].copy()
    fleet["alert"] = fleet["probability"] >= threshold
    tp = int((fleet["alert"] & (fleet["label"] == 1)).sum())
    fp = int((fleet["alert"] & (fleet["label"] == 0)).sum())
    fn = int((~fleet["alert"] & (fleet["label"] == 1)).sum())
    tn = int((~fleet["alert"] & (fleet["label"] == 0)).sum())

    st.subheader(f"One decision per engine at threshold {threshold:.2f}")
    c1, c2, c3, c4 = st.columns(4)
    if reveal:
        c1.metric("Failing engines caught", f"{tp} of {tp + fn}")
        c2.metric("Failing engines missed", fn)
        c3.metric("False alarms", fp)
        c4.metric("Healthy engines left alone", tn)
    else:                                   # real use: only the model's decisions are known
        c1.metric("Engines flagged for inspection", int(fleet["alert"].sum()))
        c2.metric("Engines left flying", int((~fleet["alert"]).sum()))
        c3.metric("Highest probability", f"{fleet['probability'].max():.0%}")
        c4.metric("Engines in the fleet", len(fleet))

    fleet["Outcome"] = "Correct, no alert"
    fleet.loc[fleet["alert"] & (fleet["label"] == 1), "Outcome"] = "Correct alert"
    fleet.loc[fleet["alert"] & (fleet["label"] == 0), "Outcome"] = "False alarm"
    fleet.loc[~fleet["alert"] & (fleet["label"] == 1), "Outcome"] = "Missed failure"
    dots = alt.Chart(fleet).mark_circle(size=90, opacity=0.85).encode(
        x=alt.X("RUL:Q", title="True cycles left (from NASA's answer file)"),
        y=alt.Y("probability:Q", title="Predicted probability", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Outcome:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Correct alert", "Correct, no alert", "False alarm",
                                                "Missed failure"],
                                        range=["#1e8449", "#9aa5b1", "#e67e22", "#c0392b"])),
        tooltip=["unit", "cycle", "RUL", alt.Tooltip("probability:Q", format=".1%"), "Outcome"])
    hline = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="#c0392b", strokeDash=[6, 4]).encode(y="y:Q")
    if reveal:
        vline = alt.Chart(pd.DataFrame({"x": [FAIL_WINDOW]})).mark_rule(
            color="#34495e", strokeDash=[2, 3]).encode(x="x:Q")
        st.altair_chart((dots + hline + vline).properties(height=380), width="stretch")
        st.caption(f"Each dot is one test engine at its last recorded cycle. Engines left of the vertical "
                   f"line really fail within {FAIL_WINDOW} cycles; engines above the dashed red line are flagged.")
    else:
        fleet["Decision"] = fleet["alert"].map({True: "Inspect now", False: "Keep flying"})
        blind = alt.Chart(fleet).mark_circle(size=90, opacity=0.85).encode(
            x=alt.X("cycle:Q", title="Cycles observed so far"),
            y=alt.Y("probability:Q", title="Predicted probability", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Decision:N", legend=alt.Legend(orient="top", title=None),
                            scale=alt.Scale(domain=["Inspect now", "Keep flying"],
                                            range=["#c0392b", "#9aa5b1"])),
            tooltip=["unit", "cycle", alt.Tooltip("probability:Q", format=".1%"), "Decision"])
        st.altair_chart((blind + hline).properties(height=380), width="stretch")
        st.caption("Each dot is one test engine at its last recorded cycle. Engines above the dashed red "
                   "line are flagged for inspection. Tick “Show the true answer” to see which "
                   "decisions were right.")

    st.subheader("Engines ranked by risk")
    table = fleet.sort_values("probability", ascending=False).rename(columns={
        "unit": "Engine", "cycle": "Cycles observed", "probability": "Probability",
        "RUL": "True cycles left"})[["Engine", "Cycles observed", "Probability", "True cycles left", "Outcome"]]
    if not reveal:
        table = table.drop(columns=["True cycles left", "Outcome"])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Probability": st.column_config.ProgressColumn(
                     "Probability", format="percent", min_value=0.0, max_value=1.0)})
