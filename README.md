# Credit Risk Modelling — PD, LGD, EAD & Risk-Based Provisioning
 
An end-to-end credit risk modelling pipeline built on Lending Club's public loan
dataset, predicting **Probability of Default (PD)**, **Loss Given Default (LGD)**,
and **Exposure at Default (EAD)**, then combining them into **Expected Loss**
to compare risk-based capital provisioning against a naive flat-rate baseline —
the same PD/LGD/EAD framework banks use under IFRS9 / Ind AS 109.
 
**Live dashboard:** _[add your Streamlit Cloud link here]_
 
---
 
## 1. Problem
 
Most credit-risk portfolios stop at "can I predict default." Real banks need to
go one step further: turn that prediction into a **capital decision** — how
much money should be reserved against expected losses, and is that better than
a flat, one-size-fits-all reserve rate.
 
This project builds that full chain:
 
```
raw loan data → cleaning → feature engineering → PD / LGD / EAD models
→ Expected Loss per loan → naive vs. risk-adjusted provisioning comparison
```
 
## 2. Data
 
- **Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
  (`accepted_2007_to_2018Q4.csv`), Kaggle, public and free.
- **~2M raw rows** → filtered to ~27 relevant columns at ingestion → cleaned
  down to **~610,000 loans with resolved outcomes** (`Fully Paid`, `Charged
  Off`, `Default`). Loans still `Current` or `Late` were excluded, since their
  true outcome isn't known yet.
- Class balance: ~79% good / ~21% defaulted.
## 3. Pipeline architecture
 
```
Data Ingestion → Data Validation → Data Transformation → Model Training → Expected Loss / Provisioning
```
 
- **Ingestion** — loads only the required columns (`usecols`), splits into
  train/test.
- **Validation** — checks column count, column names, and train/test schema
  match before anything downstream runs.
- **Transformation** — shared cleaning (missing values, sentinel-value fixes,
  feature engineering), then branches:
  - **PD branch:** Weight of Evidence (WoE) binning + Information Value (IV)
    filtering (`scorecardpy`), SMOTETomek to balance the training set only.
  - **LGD/EAD branch:** `StandardScaler` + `OneHotEncoder`, fit on defaulted
    loans only (where real loss/exposure values exist), plus a full-portfolio
    feature set for downstream Expected Loss scoring.
- **Model Training** — trains all three models, logs every run to **MLflow**
  (params, metrics, model artifacts).
- **Expected Loss** — applies all three trained models to the full test
  portfolio and computes the naive-vs-risk-adjusted provisioning comparison.
## 4. Data cleaning highlights
 
A few real issues found and handled during EDA, not assumed:
 
- **DTI sentinel values** — `999.00` appeared 76 times, a clear placeholder
  (confirmed by checking the value distribution, not just the max) vs. genuine
  high-DTI outliers further down the tail, which were left untouched.
- **Income outliers** — capped at the 99th percentile rather than dropped.
- **2-digit year date bug** — `earliest_cr_line` values like `'Mar-05'` were
  being parsed with the wrong century (`%Y` vs `%y`), producing dates as far
  out as 2068. Fixed by parsing with `%b-%y` and correcting any date that fell
  after the loan's own `issue_d` (impossible) by shifting back 100 years.
- **FICO range** — `fico_range_low`/`fico_range_high` are perfectly correlated
  (same underlying band), combined into a single `fico_avg` feature.
## 5. Modelling
 
| Model | Target | Type | Trained on | Metric |
|---|---|---|---|---|
| PD | `default_flag` (0/1) | Classification | All resolved loans | AUC / Gini |
| LGD | Loss % on default | Regression | Defaulted loans only | RMSE |
| EAD | Outstanding balance at default | Regression | Defaulted loans only | RMSE |
 
- **PD:** Logistic Regression (interpretable, regulator-preferred baseline)
  compared against XGBoost; the better model by AUC is selected.
- **Class imbalance:** handled with SMOTETomek on the training set. This
  shifted predicted probabilities away from the real-world default rate
  (mean predicted PD of ~45.6% vs. an actual ~21.2%), so a
  **`CalibratedClassifierCV`** (sigmoid) step was added, fit on the real,
  unbalanced test distribution — bringing predicted and actual rates back
  into alignment.
- **Validation:** chronological train/test split (by `issue_d`), not random —
  the model is trained on earlier loan vintages and tested on later ones, to
  mirror how it would actually be used.
### Results (test set)
 
- PD AUC: **~0.72**, Gini: **~0.44**
- LGD RMSE: **~0.185**
- EAD RMSE: **~$3,040**
## 6. Expected Loss & provisioning
 
```
Expected Loss = PD × LGD × EAD   (per loan)
```
 
Computed for every loan in the test portfolio (not just historically defaulted
ones — LGD/EAD are applied prospectively as forward-looking severity/exposure
estimates), then compared against a naive flat-rate reserve assumption.
 
**Known limitation, found and documented, not hidden:** the LGD model is
trained only on loans that actually defaulted, so its predictions extrapolate
less reliably onto the full portfolio — observed as a materially higher mean
LGD when applied broadly than would be expected for a blended, healthy
portfolio. Real IFRS9 implementations handle exactly this with downturn LGD
floors or segment-level overlays rather than trusting raw model output
directly; this project reports the finding honestly rather than
artificially forcing it to a "cleaner" number.
 
## 7. Dashboard
 
A Streamlit app (`app.py`) presents:
- Model performance (AUC, RMSE)
- Naive vs. risk-adjusted provisioning comparison
- Portfolio overview (default rate by grade, from a sampled subset of the
  cleaned test data)
Deployed on Streamlit Community Cloud — see link above.
 
## 8. Tooling
 
- **Python**, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scorecardpy`,
  `imbalanced-learn`
- **MLflow** for experiment tracking (params, metrics, model versioning)
- **Streamlit** for the results dashboard
- Modular pipeline structure (`src/components`, `src/entity`,
  `src/pipeline`) with dataclass-based config/artifact objects passed between
  stages
## 9. Limitations & next steps
 
- LGD/EAD extrapolation onto non-defaulted loans (see Section 6) — next step
  would be segment-level LGD or a downturn LGD floor.
- Dataset is US-based (Lending Club); the PD/LGD/EAD/Expected Loss
  methodology maps directly onto other IFRS9-aligned frameworks (e.g. India's
  Ind AS 109), but the specific figures wouldn't transfer without
  region-specific data.
- No macroeconomic scenario overlay (base/adverse/downturn PD adjustment) —
  a natural extension for future work.
- Live single-loan prediction (feeding new input through the same WoE/
  scaler pipeline) was scoped but not included in the deployed dashboard, in
  favour of a simpler, more robust static-results view.
## 10. How to run locally
 
```bash
git clone https://github.com/ShrutiShravani/Credit-Risk-Modelling.git
cd Credit-Risk-Modelling
pip install -r requirements.txt
 
python main.py          # runs the full training pipeline
streamlit run app.py    # launches the dashboard locally
```
 
