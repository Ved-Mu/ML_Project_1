# Report Outline (≈ 10–15 pages)

**Title:** Predictive Maintenance of Turbofan Engines using Machine Learning Classification

Each section maps to one notebook. Copy the plots and tables from the notebooks; write the text in your own words using the markdown explanations in the notebooks.

---

## 1. Introduction (1 page)
- What predictive maintenance is; reactive vs preventive vs predictive (table in Notebook 01).
- Why it matters for aircraft engines: safety, cost of unplanned downtime, cost of replacing healthy parts too early.
- **Problem statement:** predict whether an engine will fail within the next 30 cycles (binary classification).
- Objectives = the 7 rubric items.

## 2. Dataset (1 page) — *Notebook 01*
- NASA C-MAPSS, FD001: 100 train engines (run to failure), 100 test engines (cut off early), RUL answer key.
- Columns: unit, cycle, 3 settings, 21 sensors (sensor table).
- What one row means; data quality (no missing values, duplicates or gaps).
- Engine lifetime histogram (128–362 cycles) → why a fixed schedule doesn't work.

## 3. Exploratory Data Analysis (2 pages) — *Notebook 02*
- Constant columns dropped (10 of 24) → 14 useful sensors.
- Sensor trends vs RUL plot: rising and falling sensors; noise; s9/s14 engine-dependent.
- Healthy vs near-failure distributions.
- Correlation with RUL + heatmap → multicollinearity.
- Class balance: 15% positive → accuracy is misleading.
- Table "Finding → Decision".

## 4. Methodology (3 pages)
### 4.1 Labelling and feature derivation — *Notebook 03*
- RUL formula (train and test), label = RUL ≤ 30, why 30.
- Rolling mean / rolling std (window 5), why only past cycles, evidence table (AUC raw vs rolling mean).
- Engine-wise split + GroupKFold; scaling inside a Pipeline.
### 4.2 Classification models — *Notebook 04*
- Logistic Regression (sigmoid, weights), Decision Tree (Gini splits).
### 4.3 Regularization — *Notebook 04*
- L1 vs L2 (formulas, C), feature-count plot; tree depth overfitting curve.
### 4.4 Hyperparameter tuning — *Notebooks 04–05*
- Parameters vs hyperparameters; GridSearchCV grids; F1 as the selection metric; class_weight result.
### 4.5 Ensemble methods — *Notebook 05*
- Bagging vs boosting table; Random Forest and Gradient Boosting grids.

## 5. Results and Performance Analysis (3 pages) — *Notebook 05*
- Metric definitions (confusion matrix, recall, precision, F1, ROC-AUC, PR-AUC).
- Cross-validation comparison table; test-set table; ROC and PR curves; confusion matrices.
- Error analysis: recall by RUL bucket → boundary cases; false alarms are near the boundary.
- Threshold tuning (chosen from out-of-fold predictions) → recall vs precision trade-off.
- Engine-level evaluation (100 decisions).
- Feature importance → derived features dominate.
- Final comparison table + bar chart.

## 6. Conclusion (1 page)
- Final model choice and justification.
- Key insights (derived features, regularization, the boundary effect, threshold choice).
- Limitations and future work.

## References
1. A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008.
2. NASA Prognostics Center of Excellence, C-MAPSS dataset, https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 2011.
4. L. Breiman, "Random Forests", Machine Learning 45, 2001.
5. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", Annals of Statistics, 2001.

---

## Slides (≈ 12)
1. Title + team
2. Problem: what predictive maintenance is and why it matters
3. Dataset: FD001, what one row means
4. EDA: sensor trends plot
5. EDA: correlations + class imbalance
6. Feature derivation: raw vs rolling mean plot + AUC table
7. Engine-wise split: why, and what leakage is
8. Models + regularization: tree overfitting curve, L1 feature count
9. Tuning + ensembles: bagging vs boosting
10. Results: final comparison table + ROC curves
11. Error analysis + threshold tuning
12. Conclusion, limitations, future work
