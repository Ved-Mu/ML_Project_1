"""Builds notebooks/01_data_understanding.ipynb (stdlib only)."""
import json
import sys

out_path = sys.argv[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


md("""
# Predictive Maintenance of Turbofan Engines using Machine Learning Classification
## Notebook 1 — Data Understanding

**Goal of this notebook:** load the NASA C-MAPSS FD001 data, give every column a name, and understand what one row means — before doing any analysis.

### The problem (in one line)
Given an engine's sensor readings, predict **whether it will fail within the next 30 cycles** (yes / no).

### What is predictive maintenance?
| Strategy | When you repair | Problem |
|---|---|---|
| Reactive | After it breaks | Unplanned downtime, can be dangerous |
| Preventive | On a fixed schedule | Replaces healthy parts too early → wasted money |
| **Predictive** | **When the data says failure is near** | Needs data + a model → **this project** |
""")

md("""
## 1. About the dataset

**Source:** NASA Ames Prognostics Center of Excellence — *C-MAPSS* (Commercial Modular Aero-Propulsion System Simulation).
Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM 2008.
Download: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (U.S. Government work, public).

**What it is:** a high-fidelity NASA simulation of a fleet of turbofan (jet) engines. Each engine starts healthy, with some random initial wear, and is run **until it fails**.

C-MAPSS has 4 subsets. **We use only FD001** — the simplest one:

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| **FD001** | **100** | **100** | **1** (sea level) | **1** (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 248 | 249 | 6 | 2 |

*HPC = High-Pressure Compressor.* One condition + one fault mode means any change in the sensors is due to wear, not due to flying conditions — easier to model and to explain.

**The 3 files we use:**
| File | What it contains |
|---|---|
| `train_FD001.txt` | Full life of 100 engines — every engine runs **until failure** |
| `test_FD001.txt` | 100 *other* engines — data **stops some time before failure** |
| `RUL_FD001.txt` | The true number of cycles each test engine had left when its data stopped (the "answer key") |
""")

code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)

# Folder that holds train_FD001.txt, test_FD001.txt, RUL_FD001.txt
# (on Google Colab: upload the 3 files and set DATA_DIR = ".")
DATA_DIR = "../data/raw/CMAPSSData"
""")

md("""
## 2. Naming the columns

The raw files have **no header** — just 26 numbers per row separated by spaces. From the dataset readme and the Saxena et al. paper, the columns are:

- **unit** — engine ID (1 to 100)
- **cycle** — time step; one cycle ≈ one flight
- **setting_1, setting_2, setting_3** — operational settings (altitude, Mach number, throttle). In FD001 these barely change (one operating condition).
- **s1 … s21** — 21 sensor measurements

| Sensor | Symbol | Meaning | Unit |
|---|---|---|---|
| s1 | T2 | Total temperature at fan inlet | °R |
| s2 | T24 | Total temperature at LPC outlet | °R |
| s3 | T30 | Total temperature at HPC outlet | °R |
| s4 | T50 | Total temperature at LPT outlet | °R |
| s5 | P2 | Pressure at fan inlet | psia |
| s6 | P15 | Total pressure in bypass duct | psia |
| s7 | P30 | Total pressure at HPC outlet | psia |
| s8 | Nf | Physical fan speed | rpm |
| s9 | Nc | Physical core speed | rpm |
| s10 | epr | Engine pressure ratio (P50/P2) | – |
| s11 | Ps30 | Static pressure at HPC outlet | psia |
| s12 | phi | Ratio of fuel flow to Ps30 | pps/psi |
| s13 | NRf | Corrected fan speed | rpm |
| s14 | NRc | Corrected core speed | rpm |
| s15 | BPR | Bypass ratio | – |
| s16 | farB | Burner fuel-air ratio | – |
| s17 | htBleed | Bleed enthalpy | – |
| s18 | Nf_dmd | Demanded fan speed | rpm |
| s19 | PCNfR_dmd | Demanded corrected fan speed | rpm |
| s20 | W31 | HPT coolant bleed | lbm/s |
| s21 | W32 | LPT coolant bleed | lbm/s |

*LPC/HPC = Low/High-Pressure Compressor, LPT/HPT = Low/High-Pressure Turbine, °R = degrees Rankine.*
""")

code("""
index_cols   = ["unit", "cycle"]
setting_cols = ["setting_1", "setting_2", "setting_3"]
sensor_cols  = [f"s{i}" for i in range(1, 22)]
col_names    = index_cols + setting_cols + sensor_cols

# sep=r"\\s+" : columns are separated by one or more spaces
train = pd.read_csv(f"{DATA_DIR}/train_FD001.txt", sep=r"\\s+", header=None, names=col_names)
test  = pd.read_csv(f"{DATA_DIR}/test_FD001.txt",  sep=r"\\s+", header=None, names=col_names)
rul   = pd.read_csv(f"{DATA_DIR}/RUL_FD001.txt",   sep=r"\\s+", header=None, names=["true_RUL"])

print("train:", train.shape, "| test:", test.shape, "| RUL:", rul.shape)
""")

md("""
## 3. What does one row mean?

**One row = one engine at one cycle.** It is a snapshot: *"engine 1, after flight number 1, had these 3 settings and these 21 sensor readings."*

So the data is a **time series per engine**: engine 1 has rows for cycle 1, 2, 3, … until it fails.
""")

code("""
train.head()
""")

code("""
# The full life of engine 1: cycles 1 ... last cycle (= the cycle it failed)
engine_1 = train[train["unit"] == 1]
print(f"Engine 1 has {len(engine_1)} rows -> it failed after cycle {engine_1['cycle'].max()}")
engine_1[["unit", "cycle", "s2", "s3", "s4", "s7", "s11"]].tail()
""")

md("""
## 4. Basic data-quality checks

Before any analysis we check: data types, missing values, duplicate rows, and that every engine's cycles run 1, 2, 3, … without gaps.
""")

code("""
train.info()
""")

code("""
print("Missing values (train):", train.isna().sum().sum())
print("Missing values (test): ", test.isna().sum().sum())
print("Duplicate rows (train):", train.duplicated().sum())

# Every engine should have cycles 1..N with no gaps
gaps = train.groupby("unit")["cycle"].apply(lambda c: (c.diff().dropna() != 1).sum()).sum()
print("Cycle gaps (train):    ", gaps)
""")

code("""
train.describe().T
""")

md("""
**What to look for in the table above:** columns whose `std` is 0 (or almost 0) **never change** — they carry no information about wear. We will confirm and drop them in Notebook 2 (EDA).
""")

code("""
# Quick preview: how many distinct values does each column have?
nunique = train[setting_cols + sensor_cols].nunique().sort_values()
nunique.head(10)
""")

md("""
## 5. How long do engines live?

In the training set every engine runs until failure, so **an engine's lifetime = its last cycle number**.
""")

code("""
lifetimes = train.groupby("unit")["cycle"].max()
print(lifetimes.describe().round(1))

plt.figure(figsize=(8, 4))
sns.histplot(lifetimes, bins=20)
plt.title("Engine lifetimes in the training set (FD001)")
plt.xlabel("Cycles until failure")
plt.ylabel("Number of engines")
plt.show()
""")

md("""
**Interpretation:** engines do **not** all fail at the same age. If they did, a simple fixed schedule ("service every N cycles") would be enough. Because lifetimes vary a lot, we need the **sensor data** to tell *this particular* engine's condition — which is exactly why predictive maintenance is useful.
""")

md("""
## 6. How is the test set different?

- **Train:** each engine runs **until failure** → we know exactly how many cycles were left at every row.
- **Test:** each engine's data **stops early**, at some random point before failure. `RUL_FD001.txt` tells us how many cycles it really had left at that last row.

**RUL = Remaining Useful Life** = number of cycles left before failure.
""")

code("""
test_last = test.groupby("unit")["cycle"].max().rename("last_cycle_seen").to_frame()
test_last["true_RUL"] = rul["true_RUL"].values          # row i of RUL file = engine i
test_last["total_life"] = test_last["last_cycle_seen"] + test_last["true_RUL"]
test_last.head()
""")

code("""
print("Test engines:", test["unit"].nunique())
print("Test engines that were within 30 cycles of failure when data stopped:",
      (test_last["true_RUL"] <= 30).sum())
""")

md("""
## 7. Preview: how the label will be built (done properly in Notebook 3)

For every training row:

$$\\text{RUL} = \\text{last cycle of that engine} - \\text{current cycle}$$

$$\\text{label} = \\begin{cases} 1 & \\text{if RUL} \\le 30 \\;\\; (\\text{will fail soon}) \\\\ 0 & \\text{otherwise} \\end{cases}$$

Below is a quick preview only, to see roughly how balanced the two classes will be.
""")

code("""
preview_rul = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
preview_label = (preview_rul <= 30).astype(int)
print(preview_label.value_counts().rename({0: "0 = healthy", 1: "1 = fails within 30 cycles"}))
print(f"\\nShare of 'fails soon' rows: {preview_label.mean():.1%}")
""")

md("""
**Interpretation:** only a minority of rows are "fails soon" → the classes are **imbalanced**. This is why, later, **accuracy alone will be misleading** and we will focus on **recall, precision and F1**.

## 8. Key takeaways (Notebook 1)

1. FD001 = 100 training engines run to failure + 100 test engines cut off early, with an answer key (`RUL_FD001.txt`).
2. One row = one engine at one cycle → each engine is a **time series**.
3. 26 columns: engine ID, cycle, 3 settings, 21 sensors. The data is clean: no missing values, no duplicates, no gaps.
4. Some columns barely or never change → candidates to drop (Notebook 2).
5. Engine lifetimes vary a lot → a fixed maintenance schedule would not work well; sensor data is needed.
6. The yes/no label will be imbalanced → choose metrics carefully.

## Likely viva questions on this part
- *What is one row of your data?* → One engine at one cycle (≈ one flight).
- *Why FD001 and not the other subsets?* → One operating condition and one fault mode, so sensor changes reflect wear only; simplest to model and explain.
- *How is the test set different from the training set?* → Test engines stop before failure; the true remaining life is given separately in `RUL_FD001.txt`.
- *Is the data real?* → It is a high-fidelity NASA simulation, the standard benchmark for predictive maintenance research.
- *What is RUL?* → Remaining Useful Life: cycles left before failure.

**Next → Notebook 2: Exploratory Data Analysis.**
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out_path, "with", len(cells), "cells")
