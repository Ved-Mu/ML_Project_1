# improved_implementation

Stage 2 of the turbofan predictive-maintenance project: corrected validation, rolling-window and
feature-set studies, the final model, and an interactive demo. It reads the raw data from
`../data/raw/CMAPSSData/`.

## Layout

| Path | Content |
|---|---|
| `config/baseline.yaml` | Settings for the baseline reproduction and corrected baseline |
| `config/improved.yaml` | Settings for the improvement experiments |
| `src/pm/` | Data loading, causal features, grouped folds, metrics, result logging, model factories |
| `src/run_phase1_2.py` | Integrity checks, baseline re-measure, corrected baseline, regression reference |
| `src/run_phase3.py` | Window and feature-set ablations, final model |
| `src/analysis.py` | Figures and error analysis |
| `src/summarize.py`, `src/ablation_table.py` | Mean ± std over seeds and ablation tables from `results.csv` |
| `app/streamlit_app.py`, `.streamlit/config.toml` | Engine Health Monitor: interactive demo of the final model |

## Reproduce

All commands run from `improved_implementation/`. Each run appends rows to `results.csv`
(created on the first run); logs and figures are written to `logs/` and `figures/`.

```
# 1. Integrity / leakage checks (run notebook 03 first so data/processed/ exists)
python src/run_phase1_2.py checks

# 2. As-found protocol re-measured, one process per model
python src/run_phase1_2.py original logistic_regression
python src/run_phase1_2.py original decision_tree
python src/run_phase1_2.py original random_forest
python src/run_phase1_2.py original gradient_boosting

# 3. Corrected baseline (classification, 5 seeds) and nested-CV diagnostic
python src/run_phase1_2.py corrected
python src/run_phase1_2.py nested

# 4. Regression reference (RUL), 5 seeds
python src/run_phase1_2.py regression

# 5. Rolling-window ablation, then feature-set ablation at the chosen window
python src/run_phase3.py window 5 10 20 30 40 50
python src/run_phase3.py kinds 40

# 6. FINAL MODEL: logistic regression, window 40, raw + mean + std features
python src/run_phase3.py final logistic_regression 40 raw,mean,std

# 7. Figures, error analysis, tables
python src/analysis.py
python src/summarize.py
python src/ablation_table.py window_ablation

# Demo app
streamlit run app/streamlit_app.py
```

Run time is about 45 minutes in total, 30 of which are the optional nested CV in step 3.
