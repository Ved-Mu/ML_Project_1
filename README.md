# Predictive Maintenance of Turbofan Engines using Machine Learning Classification

Predict whether an aircraft engine will **fail within the next 30 cycles** (yes / no) from its sensor readings,
using the NASA C-MAPSS FD001 dataset.

## Dataset
- **NASA C-MAPSS** turbofan engine degradation simulation, subset **FD001** (100 training engines run to failure, 100 test engines, 1 operating condition, 1 fault mode).
- Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data (U.S. Government work, public).
- Reference: A. Saxena, K. Goebel, D. Simon, N. Eklund, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, PHM 2008.

## Project structure
```
data/raw/CMAPSSData/        original NASA files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt, readme, paper)
data/processed/             features + labels created by notebook 03
notebooks/
  01_data_understanding.ipynb          what the data is, what one row means
  02_eda.ipynb                         Exploratory Data Analysis
  03_features_and_labels.ipynb         RUL, label, rolling features, engine-wise split
  04_models_regularization_tuning.ipynb Logistic Regression, Decision Tree, L1/L2, depth, GridSearchCV
  05_ensemble_and_performance.ipynb    Random Forest, Gradient Boosting, final comparison
models/                     saved tuned models (.joblib)
results/final_comparison.csv  final comparison table of all models
results/figures/            every plot from the notebooks as PNG (for the report and slides);
                            extra_*.png = two presentation charts (boundary effect, model comparison)
docs/Synopsis_Predictive_Maintenance.docx / .pdf   project synopsis (course template)
docs/Project_Guide_Predictive_Maintenance.pdf      full guide: data, code, reasoning, results, how to present
docs/report_outline.md      report + slides structure
docs/viva_qa.md             39 likely viva questions with short answers
src/                        scripts that generated the notebooks (not needed to run the project)
```

## Rubric coverage
| Requirement | Where |
|---|---|
| 1. EDA | Notebook 02 |
| 2. Feature derivation | Notebook 03 |
| 3. Classification | Notebook 04 (Logistic Regression, Decision Tree) |
| 4. Regularization | Notebook 04 (L1 vs L2, tree depth / leaf size) |
| 5. Hyperparameter tuning | Notebook 04 + 05 (GridSearchCV with GroupKFold) |
| 6. Ensemble | Notebook 05 (Random Forest, Gradient Boosting) |
| 7. Performance analysis | Notebook 05 (metrics, ROC/PR curves, error analysis, threshold tuning, engine-level evaluation) |

## How to run
**Locally**
```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks **in order 01 → 05** (03 creates the files that 04 and 05 read; 04 saves models that 05 loads).

**Google Colab**
Upload the notebooks, the `data/` folder and keep the same folder layout (or change `DATA_DIR` / the `../data/...` paths at the top of each notebook).
