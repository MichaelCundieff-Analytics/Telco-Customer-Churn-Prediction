"""
Telco Customer Churn - end-to-end pipeline.

Usage:
    python main.py            # run everything: setup -> eda -> train -> score
    python main.py setup      # load raw data into SQL, clean, engineer features
    python main.py eda        # regenerate EDA figures
    python main.py train      # train, evaluate and save the model
    python main.py score      # score customers and write predictions to SQL

Workflow:
    raw CSV -> raw_customers (SQL) -> cleaning -> feature engineering
    -> customer_features (SQL) -> model training + threshold selection
    -> churn_predictions (SQL) -> reporting views + figures
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe; scripts only save figures

from src import config
from src.data_cleaning import clean_customers, load_raw_data, save_clean_data
from src.database import (
    create_views, get_engine, init_tables, read_query, write_dataframe,
)
from src.eda import run_eda
from src.feature_engineering import add_features, save_features

# Note: src.train and src.score_customers are imported inside their stage
# functions so the data/EDA stages work even before scikit-learn is
# installed, and so a broken ML environment fails with a clear message.


def validate_paths() -> None:
    """Fail fast with a clear message if the project layout is broken."""
    if not config.RAW_DATA_FILE.exists():
        sys.exit(f"ERROR: raw dataset not found at {config.RAW_DATA_FILE}.\n"
                 "Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/.")
    for d in (config.PROCESSED_DATA_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def stage_setup(engine):
    """Steps 1-5: raw data -> SQL -> cleaning -> features -> SQL."""
    print("[1/4] Setup: loading data and building the database...")
    init_tables(engine)

    raw = load_raw_data()
    write_dataframe(raw, "raw_customers", engine, if_exists="replace")
    print(f"  raw_customers: {len(raw):,} rows written to SQL")

    clean = clean_customers(raw)
    save_clean_data(clean)

    features = add_features(clean)
    save_features(features)
    write_dataframe(features, "customer_features", engine, if_exists="replace")
    print(f"  customer_features: {len(features):,} rows "
          f"({features.shape[1]} columns) written to SQL")
    return features


def stage_eda(engine):
    """EDA figures, generated from the feature table in SQL."""
    print("[2/4] EDA: generating figures in reports/figures ...")
    features = read_query("SELECT * FROM customer_features", engine)
    run_eda(features)
    print(f"  figures saved to {config.FIGURES_DIR}")


def stage_train(engine):
    print("[3/4] Training and evaluating models...")
    from src.train import run_training
    return run_training(engine)


def stage_score(engine):
    print("[4/4] Scoring customers and writing predictions to SQL...")
    from src.score_customers import score_customers
    score_customers(engine)
    create_views(engine)

    print("\nSample insight - churn risk by contract type (from SQL view):")
    print(read_query("SELECT * FROM v_risk_by_contract", engine).to_string(index=False))
    print("\nTop 5 highest-risk customers (from SQL view):")
    print(read_query("SELECT * FROM v_highest_risk_customers LIMIT 5", engine)
          .to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Telco churn pipeline")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["all", "setup", "eda", "train", "score"])
    args = parser.parse_args()

    validate_paths()
    engine = get_engine()

    if args.stage in ("all", "setup"):
        stage_setup(engine)
    if args.stage in ("all", "eda"):
        stage_eda(engine)
    if args.stage in ("all", "train"):
        stage_train(engine)
    if args.stage in ("all", "score"):
        stage_score(engine)

    print("\nDone. Database:", config.DB_FILE)


if __name__ == "__main__":
    main()
