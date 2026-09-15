"""
03_classify_intents.py

Step 3 of the AmazonHelp support-agent pipeline.

What this script does:
  1. Loads full customer messages (inbound_customer_messages.csv) and labeled sample (clustered_sample.csv).
  2. Maps cluster numbers (0-14) to human-defined intent categories.
  3. Trains a Logistic Regression classifier on sentence embeddings of the labeled sample.
  4. Evaluates classifier performance on an 80/20 train/test split for a sanity check.
  5. Detects the language of every customer message in the full dataset using langdetect.
  6. Predicts intent categories for all messages using sentence embeddings + trained classifier.
  7. Overrides intent to 'other_non_english' for any message not detected as English.
  8. Saves final output to 'classified_messages.csv' and prints final summary counts.

How to run:
  python 03_classify_intents.py
  python 03_classify_intents.py --inbound data/inbound_customer_messages.csv --sample data/clustered_sample.csv

Outputs:
  - data/classified_messages.csv  (Full dataset with detected_language and predicted_intent)
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Ensure deterministic language detection results
DetectorFactory.seed = 0

# Manual mapping from cluster numbers (0-14) to intent categories
CLUSTER_INTENT_MAP = {
    0: "follow_up_unresolved",
    1: "other_non_english",
    2: "billing_refund",
    3: "order_issue",
    4: "other_non_english",
    5: "other_non_english",
    6: "delivery_issue",
    7: "device_technical_issue",
    8: "general_complaint",
    9: "delivery_issue",
    10: "other_non_english",
    11: "order_issue",
    12: "delivery_issue",
    13: "account_access",
    14: "delivery_issue",
}


def load_file_path(file_path: str) -> str:
    """Resolves relative file path whether running from project root or subfolder."""
    if not os.path.exists(file_path):
        subfolder_path = os.path.join("Amazon_helpcare_twitter_bot", file_path)
        if os.path.exists(subfolder_path):
            return subfolder_path
        raise FileNotFoundError(f"Required file not found at '{file_path}'.")
    return file_path


def detect_language(text: str) -> str:
    """Detects text language, falling back to 'unknown' if detection fails or text is too short."""
    try:
        text_str = str(text).strip()
        if len(text_str) < 4:
            return "unknown"
        return detect(text_str)
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Classify customer tweets into intent categories.")
    parser.add_argument("--inbound", default="data/inbound_customer_messages.csv", help="Path to inbound messages CSV")
    parser.add_argument("--sample", default="data/clustered_sample.csv", help="Path to clustered sample CSV")
    parser.add_argument("--outdir", default="data", help="Output directory to save results (default: data)")
    args = parser.parse_args()

    # Step 1: Load CSV files
    inbound_path = load_file_path(args.inbound)
    sample_path = load_file_path(args.sample)

    print(f"Loading full dataset from: {inbound_path}")
    full_df = pd.read_csv(inbound_path)
    full_df["text"] = full_df["text"].fillna("").astype(str)

    print(f"Loading labeled sample from: {sample_path}")
    sample_df = pd.read_csv(sample_path)
    sample_df["text"] = sample_df["text"].fillna("").astype(str)

    # Determine cluster column name dynamically (cluster_id or cluster)
    cluster_col = "cluster_id" if "cluster_id" in sample_df.columns else "cluster"

    # Step 2: Apply manual cluster-to-intent mapping
    sample_df["intent"] = sample_df[cluster_col].map(CLUSTER_INTENT_MAP)
    print(f"Mapped {len(sample_df):,} sample rows into intent labels.")

    # Step 3: Embed sample text using offline SentenceTransformer model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings for labeled training sample...")
    X_sample = model.encode(sample_df["text"].tolist(), show_progress_bar=True)
    y_sample = sample_df["intent"].values

    # Step 4: Sanity check evaluation with 80/20 train/test split
    print("Evaluating Logistic Regression model on 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
    )

    eval_classifier = LogisticRegression(max_iter=1000, random_state=42)
    eval_classifier.fit(X_train, y_train)
    y_pred_eval = eval_classifier.predict(X_test)

    print("\n--- CLASSIFIER SANITY CHECK ---")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_eval):.2%}")
    print(classification_report(y_test, y_pred_eval))

    # Train final classifier on the complete labeled sample
    print("Training final classifier on all labeled sample data...")
    final_classifier = LogisticRegression(max_iter=1000, random_state=42)
    final_classifier.fit(X_sample, y_sample)

    # Save trained classifier, label list, and model config to disk
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(final_classifier, os.path.join(args.outdir, "intent_classifier.joblib"))
    joblib.dump(list(final_classifier.classes_), os.path.join(args.outdir, "intent_labels.joblib"))
    with open(os.path.join(args.outdir, "model_config.json"), "w", encoding="utf-8") as config_file:
        json.dump({"model_name": "all-MiniLM-L6-v2"}, config_file, indent=2)
    print("Saved trained classifier, label list, and model config to disk.")

    # Step 5: Detect language for all messages in the full dataset
    print(f"Detecting language for {len(full_df):,} messages in full dataset...")
    full_df["detected_language"] = [detect_language(text) for text in full_df["text"]]

    # Step 6: Embed full dataset and predict intents
    print(f"Generating embeddings for full dataset ({len(full_df):,} messages)...")
    X_full = model.encode(full_df["text"].tolist(), batch_size=256, show_progress_bar=True)

    print("Predicting intent categories for full dataset...")
    full_df["predicted_intent"] = final_classifier.predict(X_full)

    # Step 7: Override non-English messages to 'other_non_english'
    non_english_mask = full_df["detected_language"] != "en"
    full_df.loc[non_english_mask, "predicted_intent"] = "other_non_english"
    print(f"Applied non-English override to {non_english_mask.sum():,} messages.")

    # Step 8: Save classified dataset to CSV
    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, "classified_messages.csv")
    output_columns = ["tweet_id", "text", "created_at", "detected_language", "predicted_intent"]
    
    full_df[output_columns].to_csv(output_path, index=False)
    print(f"\nSaved classified dataset to: {output_path}")

    # Step 9: Print summary count of final intent distribution
    print("\n=========================================")
    print(" FINAL INTENT DISTRIBUTION SUMMARY       ")
    print("=========================================")
    intent_counts = full_df["predicted_intent"].value_counts()
    for intent_label, count in intent_counts.items():
        percentage = (count / len(full_df)) * 100
        print(f"  {intent_label:<25} : {count:6,} ({percentage:5.1f}%)")
    print("=========================================\n")


if __name__ == "__main__":
    main()
