# Predictive Maintenance of Turbofan Engines using Machine Learning Classification

Predict whether an aircraft engine will **fail within the next 30 cycles** (yes / no) from its sensor readings,
using the NASA C-MAPSS FD001 dataset.

The project has two stages. **Stage 1** (the notebooks) builds the baseline with 5-cycle rolling features.
**Stage 2** (`improved_implementation/`) fixes the validation so that it resembles the test set and moves to a
40-cycle window; the final model is logistic regression (test F1 0.870, all 25 failing test engines caught).

## Dataset
- **NASA C-MAPSS** turbofan engine degradation simulation, subset **FD001** (100 training engines run to failure, 100 test engines, 1 operating condition, 1 fault mode).
- Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (U.S. Government work, public).
- Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM 2008.

## Project structure
```
data/raw/CMAPSSData/        original NASA files (train/test/RUL for FD001-FD004, readme)
data/processed/             features + labels created by notebook 03
notebooks/
  01_data_understanding.ipynb           what the data is, what one row means
  02_eda.ipynb                          Exploratory Data Analysis
  03_features_and_labels.ipynb          RUL, label, rolling features, engine-wise split
  04_models_regularization_tuning.ipynb Logistic Regression, Decision Tree, L1/L2, depth, GridSearchCV
  05_ensemble_and_performance.ipynb     Random Forest, Gradient Boosting, final comparison
src/                        scripts that generated the notebooks (not needed to run the project)
improved_implementation/    stage 2: corrected validation, window study, final model, demo app
```

## How to run
```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks **in order 01 → 05** (03 creates the files that 04 and 05 read; 04 saves the models to
`models/` that 05 loads).

Run the demo app (Engine Health Monitor):
```
cd improved_implementation
streamlit run app/streamlit_app.py
```
