# Telco Customer Churn Prediction — Python + SQL

An end-to-end machine learning system that predicts which telecom customers are likely to churn, writes churn-risk scores back into a SQL database, and turns them into stakeholder-ready analysis.

```
Raw customer data ──► SQL database ──► cleaning & feature engineering
      ──► ML model training & threshold selection ──► churn probabilities
      ──► predictions written back to SQL ──► reporting views & figures
```

## Business problem

Acquiring a new telecom customer costs far more than keeping an existing one. If the retention team knew *which* customers were likely to leave — and *why* — they could intervene with targeted offers before the cancellation call. This project builds that early-warning system.

## Objective

Predict, for every customer, the probability of churning; identify the drivers behind each high-risk score; and deliver the results where an operations team would actually consume them: a SQL database with ready-made reporting views.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`): 7,043 customers, 21 columns — demographics, subscribed services, contract/billing details, charges, and whether the customer churned in the last month. 26.5% of customers churned.

Known data quirks handled by the cleaning step (`src/data_cleaning.py`):

| Issue | Fix |
|---|---|
| `TotalCharges` stored as text with 11 blank values | Converted to numeric; blanks are all tenure-0 (never-billed) customers → set to 0.0 |
| `Churn` stored as `Yes`/`No` | Converted to binary `churn` (1/0); text column dropped so it can't leak into features |
| `SeniorCitizen` stored as 0/1 | Mapped to `No`/`Yes` for consistency with other categoricals |
| Duplicates / ID integrity | Verified: 0 duplicate rows, all 7,043 `customerID`s unique (enforced with a hard check) |

## Technologies

Python (pandas, NumPy, scikit-learn, matplotlib, joblib) · SQL (SQLite via SQLAlchemy, with a stdlib-sqlite3 fallback; PostgreSQL-ready via a single connection URL) · Jupyter · pytest

## Project structure

```
telco-churn-project/
├── data/
│   ├── raw/                  # original dataset (never modified)
│   └── processed/            # cleaned data + feature table (regenerated)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # cleaning walkthrough + EDA with insights
│   ├── 02_feature_engineering.ipynb  # features, rationale, validation
│   └── 03_model_evaluation.ipynb     # metrics, thresholds, explainability
├── src/
│   ├── config.py             # paths, DB URL, seeds, business assumptions
│   ├── data_cleaning.py      # load + clean the raw data
│   ├── database.py           # SQL engine, table/view creation, read/write
│   ├── feature_engineering.py# engineered features + documented rationale
│   ├── eda.py                # presentation-ready EDA figures
│   ├── train.py              # pipelines, model comparison, threshold choice
│   ├── evaluate.py           # metrics, curves, cost analysis, explanations
│   └── score_customers.py    # batch scoring → churn_predictions table
├── sql/
│   ├── create_tables.sql     # raw_customers, customer_features, model_runs, churn_predictions
│   ├── create_views.sql      # reporting views (latest run only)
│   └── analysis_queries.sql  # example stakeholder queries
├── models/                   # saved pipeline (joblib) + run metadata (json)
├── reports/                  # model comparison, threshold table, explanations
│   └── figures/              # all EDA + evaluation charts
├── tests/                    # data-quality, leakage, pipeline and DB tests
├── main.py                   # single-command pipeline
└── requirements.txt
```

## Installation

```bash
git clone <your-repo-url>
cd telco-churn-project
python -m venv venv
venv\Scripts\activate        # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## How to run

```bash
python main.py           # full pipeline: setup → EDA → train → score
```

or stage by stage:

```bash
python main.py setup     # raw CSV → SQL, clean, engineer features → SQL
python main.py eda       # regenerate EDA figures
python main.py train     # train + evaluate models, save the best pipeline
python main.py score     # score all customers, write predictions to SQL
pytest                   # run the test suite
```

Everything uses relative paths and a fixed random seed (42), so results are reproducible on any machine.

## Models

| Model | Why |
|---|---|
| **Logistic Regression** (baseline) | Interpretable coefficients, strong baseline for churn problems |
| **HistGradientBoostingClassifier** (challenger) | Gradient-boosted trees — the same model family as XGBoost — built into scikit-learn, so no extra dependency or install issues |

Both use identical scikit-learn `Pipeline`s: stratified 80/20 train/test split, median imputation + standard scaling for numeric features, most-frequent imputation + one-hot encoding for categoricals. **All preprocessing is fitted inside the pipeline on training data only — nothing is fitted on the full dataset, preventing leakage.**

The final model is selected by test ROC-AUC. Its operating threshold is chosen on **out-of-fold training predictions** (never the test set): the F1-optimal threshold is the default operating point, and a cost-optimal threshold is computed alongside it for the business analysis.

## Evaluation

Reported for both models on the held-out test set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, plus confusion matrix, ROC and precision-recall curves.

**Why accuracy is not enough:** only 26.5% of customers churn, so a model that predicts "nobody churns" is already ~73.5% accurate while catching zero churners. Recall (share of churners caught), precision (share of flags that are real), and the AUC metrics tell the real story.

### Results (held-out test set, n = 1,409)

Model comparison at the default 0.50 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.794 | 0.637 | 0.516 | 0.570 | **0.842** | 0.636 |
| HistGradientBoosting | 0.791 | 0.630 | 0.513 | 0.566 | 0.834 | 0.636 |

The interpretable baseline **won**: logistic regression was selected (test ROC-AUC 0.842) — a useful reminder that on modest tabular datasets a well-preprocessed linear model can match or beat boosted trees.

Final model at its F1-optimal operating threshold of **0.35** (chosen on out-of-fold training predictions, never the test set):

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.774 | 0.558 | 0.709 | 0.624 | 0.842 | 0.636 |

Lowering the threshold from 0.50 to 0.35 raised recall from 52% to **71%** — catching an extra ~1 in 5 churners — at an acceptable precision cost: the right trade when a missed churner costs far more than a wasted contact.

All artifacts are regenerated by `python main.py`: `reports/model_comparison.csv`, `models/model_metadata.json`, `reports/figures/10–14_*.png`, and the queryable `model_runs` table in SQL.

### Threshold analysis & business cost

A 0.50 threshold is a default, not a decision. `reports/threshold_analysis.csv` and `reports/figures/12_threshold_analysis.png` show precision, recall, false positives, false negatives, and estimated campaign cost at every threshold, using **hypothetical** economics: $5 per retention contact, $200 lost per missed churner. Because a missed churner costs 40× a contact, the cost-optimal threshold is low (flag aggressively); notebook 03 shows how the optimum shifts as the intervention gets more expensive. A real deployment would plug in actual campaign economics.

## What drives churn (EDA findings — verified on this dataset)

| Segment | Churn rate |
|---|---|
| Overall | **26.5%** |
| Month-to-month contract | **42.7%** (one-year: 11.3%, two-year: 2.8%) |
| Tenure 0–12 months | **47.4%** (49–72 months: 9.5%) |
| Electronic-check payers | **45.3%** (automatic payments: ~15–17%) |
| Fiber-optic internet | **41.9%** (DSL: 19.0%, no internet: 7.4%) |
| Senior citizens | **41.7%** (non-seniors: 23.6%) |

**Business interpretation:** churn concentrates in new, month-to-month, electronic-check-paying, fiber-optic customers with high monthly bills. The levers that follow: early-tenure onboarding offers, contract-upgrade incentives, auto-pay enrollment discounts, and a hard look at fiber pricing/experience.

## Feature engineering

12 engineered features, each with a documented rationale (see `FEATURE_RATIONALE` in `src/feature_engineering.py` and notebook 02): tenure groups, month-to-month flag, auto-payment flag, number of subscribed services, paperless-billing flag, average charges per tenure month, high-monthly-charge flag, internet/fiber indicators, streaming and support bundles, and spend groups. All thresholds are fixed business values (e.g. $70), never dataset statistics — another leakage guard.

## SQL integration

Four tables (`sql/create_tables.sql`):

| Table | Contents |
|---|---|
| `raw_customers` | the untouched source data |
| `customer_features` | cleaned + engineered customer table (model input) |
| `model_runs` | one row per training run: model, version, threshold, metrics |
| `churn_predictions` | customer ID, churn probability, predicted class, model name, run ID, timestamp (append-only history) |

Five reporting views (`sql/create_views.sql`), all scoped to the **latest** model run: `v_highest_risk_customers`, `v_risk_by_contract`, `v_risk_by_payment_method`, `v_risk_by_internet_service`, `v_risk_by_tenure_group`. Ready-made stakeholder queries live in `sql/analysis_queries.sql` (e.g. monthly revenue at risk, month-to-month high-risk target list).

SQLite is the default (zero install). To use PostgreSQL, set one environment variable — `TELCO_DB_URL=postgresql+psycopg2://user:pass@host:5432/telco` — and add `psycopg2-binary` to requirements; no code changes needed.

### Example prediction output (schema)

```
customer_id | churn_probability | predicted_churn | model_name              | run_id   | prediction_ts
------------+-------------------+-----------------+-------------------------+----------+---------------------
7590-VHVEG  | 0.81              | 1               | hist_gradient_boosting  | 3f9a12bc | 2026-07-13T18:40:00
5575-GNVDE  | 0.07              | 0               | hist_gradient_boosting  | 3f9a12bc | 2026-07-13T18:40:00
```
*(illustrative rows — your run produces real values)*

## Explainability

- **Global (linear):** strongest positive/negative logistic-regression coefficients (`reports/figures/13_logreg_coefficients.png`).
- **Global (final model):** permutation importance — how much test ROC-AUC drops when each raw feature is shuffled (`14_final_model_feature_importance.png`).
- **Individual:** `reports/high_risk_explanations.csv` lists the riskiest customers with their top personal risk drivers (per-feature contribution to the logistic-regression log-odds). SHAP was deliberately left out to keep the dependency footprint small; the permutation + contribution approach covers the same questions with stdlib-stable tools.

## Testing

`pytest` covers: required dataset columns, unique customer IDs, successful `TotalCharges` conversion, valid binary target, **no target/ID in model features** (leakage guard), model predictions in {0,1}, probabilities in [0,1], and a SQL insert/read round-trip on an in-memory database.

## Limitations

- Single snapshot in time — no behavioral time series, usage logs, or support-ticket signals.
- Cost figures in the threshold analysis are hypothetical placeholders.
- Assumes a contacted churner is retained; real campaigns have partial success rates.
- The dataset is a stylized IBM sample; real telco data would be messier and larger.
- No hyperparameter tuning — defaults are strong here, but a search could add a point or two of AUC.

## Future improvements

Hyperparameter tuning (e.g. `RandomizedSearchCV`), probability calibration for more trustworthy scores, a scheduled scoring job + drift monitoring, SHAP explanations, a PostgreSQL + dashboard (e.g. Metabase/Power BI) deployment, and uplift modelling to target customers who are *persuadable*, not just at-risk.

## Resume bullet points

> - Built an end-to-end churn prediction system in Python and SQL (test ROC-AUC 0.84, 71% churner recall): raw data ingestion to SQLite, leakage-safe scikit-learn pipelines (logistic regression vs. gradient boosting), and batch scoring that writes churn probabilities back to SQL reporting views used for retention targeting.
> - Engineered 12 documented, business-driven customer features and delivered EDA identifying the highest-risk segments (month-to-month contracts: 43% churn vs. 3% on two-year; first-year customers: 47%), directly informing retention strategy recommendations.
> - Replaced the default 0.50 classification threshold with a cost-based threshold analysis (contact cost vs. customer-loss cost), demonstrating how campaign economics change the optimal operating point; validated the pipeline with automated pytest data-quality and anti-leakage checks.

## Interview talking points

1. **Leakage prevention** — split before fitting; all preprocessing inside the `Pipeline`; threshold chosen on out-of-fold training predictions; fixed (not data-derived) feature thresholds; target column physically dropped after encoding.
2. **Why two models** — an interpretable baseline earns stakeholder trust and explains drivers; the boosted-tree challenger tests whether non-linearity buys real lift. Selection used ROC-AUC *and* business usefulness.
3. **Why accuracy misleads here** — 26.5% base rate; "no one churns" scores 73.5%. Talk precision/recall trade-offs in terms of wasted offers vs. lost customers.
4. **Thresholds are business decisions** — walk through the $5/$200 cost curve and why cheap interventions justify low thresholds (high recall), expensive ones demand precision.
5. **Why SQL in the loop** — predictions live where analysts and dashboards already look; views are versioned by run ID so a retrain never silently changes a dashboard.
6. **What you'd do with real data** — usage trends, support contacts, payment failures as features; measured campaign uplift instead of an assumed retention rate.
