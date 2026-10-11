# Viva Q&A Sheet — Predictive Maintenance of Turbofan Engines

Short answers to the questions most likely to come up. **Every group member should be able to answer all of them.**

---

## A. Problem and dataset

**1. What is your project about, in one sentence?**
We predict whether an aircraft engine will fail within the next 30 cycles, from its sensor readings, so that maintenance can be scheduled before the failure.

**2. What is predictive maintenance and why is it better than the alternatives?**
Reactive maintenance repairs after a breakdown (dangerous, unplanned downtime). Preventive maintenance repairs on a fixed schedule (replaces healthy parts too early). Predictive maintenance uses data to repair only when failure is actually near.

**3. Which dataset did you use?**
NASA C-MAPSS, subset FD001: a high-fidelity NASA simulation of turbofan engines. 100 training engines run until failure, 100 test engines stopped before failure, plus a file with the true remaining life of each test engine. It is public (U.S. Government work).

**4. Why FD001?**
It has one operating condition and one fault mode (High-Pressure Compressor degradation), so changes in the sensors are caused by wear only. It is the simplest subset to model and explain.

**5. What does one row mean?**
One engine at one cycle (≈ one flight): engine ID, cycle number, 3 operating settings and 21 sensor readings.

**6. Is the data real?**
It is simulated by NASA's C-MAPSS engine model with realistic noise. It is the standard benchmark for predictive maintenance research (Saxena et al., 2008).

## B. EDA

**7. What did you learn from the EDA?**
- 10 of 24 settings/sensors are constant → dropped, 14 sensors kept.
- Useful sensors drift up or down as failure approaches, but individual readings are noisy.
- Many sensors are strongly correlated with each other (e.g. s9 & s14: r = 0.96).
- Only 15% of training rows are "fails soon" → imbalanced classes.
- Engine lifetimes vary from 128 to 362 cycles → a fixed schedule would not work.

**8. Why did you drop those columns?**
Their standard deviation is ≈ 0 — they never change, so they cannot tell a healthy engine from a failing one.

**9. Why did you plot sensors against RUL instead of cycle?**
Engines live for different lengths of time. Aligning them at failure (RUL = 0) makes the degradation pattern comparable across engines.

## C. Labels and features

**10. What is RUL? How did you build the label?**
RUL = Remaining Useful Life = cycles left before failure. Train: last cycle − current cycle. Test: (last cycle seen + given RUL) − current cycle. Label = 1 if RUL ≤ 30, else 0.

**11. Why classification and not regression?**
The maintenance decision is yes/no: "send this engine for maintenance now or not". Predicting the exact RUL is possible (regression) but the decision is what matters.

**12. Why 30 cycles?**
A realistic planning window — enough time to order parts and schedule the repair. Microsoft's Azure predictive-maintenance template uses the same window on this data.

**13. What features did you derive and why?**
Rolling mean and rolling standard deviation of each sensor over the last 5 cycles. Readings are noisy; the recent trend reflects wear. Evidence: the rolling mean separates the classes better than the raw reading for 12 of 14 sensors, and rolling means carry ≈78% of the Random Forest's feature importance.

**14. Isn't using a rolling window cheating?**
No — it uses only the current and previous 4 cycles, which are known at prediction time. Using future cycles would be leakage.

**15. How did you split the data and why?**
By engine, never by random row. The final test uses NASA's 100 separate test engines; tuning uses GroupKFold (5 folds, grouped by engine). Rows of the same engine are nearly identical, so a random split would leak information and inflate scores.

**16. Why scale the features, and why inside a Pipeline?**
Sensors have very different ranges; logistic regression and its penalty are scale-sensitive. Inside a Pipeline the scaler is fitted only on the training folds, never on validation/test data.

## D. Models

**17. How does logistic regression work?**
It computes a weighted sum of the features and passes it through the sigmoid function to get a probability between 0 and 1; it predicts "fails soon" if the probability ≥ 0.5 (or another chosen threshold).

**18. How does a decision tree work?**
It asks a sequence of yes/no questions (e.g. "is s11_mean5 > 47.8?"), each chosen to best separate the classes, measured by Gini impurity. Our tree's first question uses s11_mean5 — the smoothed pressure at the HPC outlet, matching the HPC-degradation fault.

## E. Regularization

**19. What is overfitting? How did you detect it?**
The model memorises the training data and does worse on new data. The unlimited decision tree had train F1 = 1.000 but validation F1 = 0.815.

**20. What is regularization? What did you use?**
Limiting model complexity so it learns the general pattern. Logistic regression: L1 and L2 penalties (strength controlled by C). Decision tree: max_depth and min_samples_leaf.

**21. L1 vs L2?**
L2 adds Σw² and shrinks all weights smoothly (good with correlated features). L1 adds Σ|w| and sets useless weights to exactly 0 (feature selection) — at strong regularization it used only 8 of 42 features.

**22. What does C mean?**
Inverse regularization strength: small C = strong penalty = simpler model. Our best was C = 0.01 with L2.

## F. Hyperparameter tuning

**23. Parameter vs hyperparameter?**
Parameters are learned from data (weights, split thresholds). Hyperparameters are chosen before training (C, max_depth, number of trees) and tuned with cross-validation.

**24. How did you tune?**
GridSearchCV: tries every combination in a grid, scores each with 5-fold GroupKFold, keeps the best by F1. The test set is never used for tuning.

**25. Why F1 for tuning, not recall?**
A model that flags every engine has 100% recall but is useless. F1 balances recall and precision. We then raise recall deliberately with threshold tuning.

**26. Did class_weight="balanced" help?**
It raised recall but lowered precision more, so F1 dropped; the grid search chose class_weight=None. We handled recall with the decision threshold instead.

## G. Ensembles

**27. Bagging vs boosting?**
Bagging (Random Forest) trains many trees independently on bootstrap samples and random feature subsets, then averages them → reduces variance. Boosting (Gradient Boosting) trains small trees one after another, each correcting the previous errors → reduces bias.

**28. Why did you use Random Forest here?**
The single tree overfit badly. Random Forest averages many decorrelated trees, so individual errors cancel. Validation F1 went from 0.815 (single unlimited tree) to 0.871.

**29. Can more trees overfit a Random Forest?**
No — more trees only make the average more stable; it costs computation time.

## H. Performance analysis

**30. Why not accuracy?**
Only 2.5% of test rows are positive; a model that always says "healthy" would get 97.5% accuracy and catch zero failures.

**31. Which metric matters most and why?**
Recall — a missed failure is far more costly and dangerous than an extra inspection. We report precision, F1, ROC-AUC and PR-AUC alongside it.

**32. ROC-AUC vs PR-AUC?**
ROC-AUC measures how well the model ranks failing engines above healthy ones over all thresholds. PR-AUC focuses only on the rare positive class, so it is more informative under heavy imbalance.

**33. What are your results?**
| Model | CV F1 | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.866 | 0.989 | 0.803 |
| Decision Tree | 0.859 | 0.966 | 0.724 |
| **Random Forest** | **0.871** | **0.991** | **0.807** |
| Gradient Boosting | 0.866 | 0.990 | 0.788 |

**34. Why is test recall (≈0.61) lower than cross-validation recall (≈0.85)?**
Test engines are cut off before failure, so 63% of their positive rows are hard boundary cases (RUL 21–30) versus 32% in training. The model catches 100% of rows with RUL 0–10 and 90% with RUL 11–20; it struggles only near the 30-cycle boundary, where wear is gradual but the label is sharp.

**35. What is threshold tuning and how did you choose the threshold?**
Lowering the probability cut-off flags more engines → higher recall, lower precision. We chose the highest threshold giving ≥90% recall on out-of-fold training predictions (≈0.33). On the test set recall rose from 0.61 to 0.72 and precision fell from 0.82 to 0.73.

**36. What is the engine-level evaluation?**
One decision per test engine at its last available cycle ("maintenance now?"). 25 of 100 engines truly needed it. The Random Forest at threshold 0.33 flagged them with ≈95% precision and 72% recall. The missed engines were those with the most cycles left inside the window.

**37. Which model would you deploy?**
Random Forest with threshold ≈0.33: best cross-validated F1, best test ROC-AUC and PR-AUC, and no single-tree overfitting. Logistic regression is a near-equal, simpler alternative if explainability is the priority.

**38. Limitations?**
Simulated data; only one operating condition and fault mode; the 30-cycle window is a design choice; rows of one engine are not independent.

**39. Future work?**
Predict exact RUL (regression or LSTM); extend to FD002–FD004 (needs operating-condition normalisation); choose the threshold from a real cost model.
