"""
01_build_threads.py

Step 1 of the AmazonHelp support-agent pipeline.

What this script does:
  1. Loads raw Twitter customer support data (twcs.csv).
  2. Finds messages sent to AmazonHelp (or specified brand).
  3. Matches customer questions with Amazon's replies.
  4. Quality Filter (AI Model or Smart Rules):
     - AI Classification (--use-llm): Uses a HuggingFace NLI Zero-Shot AI model with GPU batching
       to evaluate true semantic intent (Solution vs. Deflection/Boilerplate).
     - Smart Heuristics (Default): Fast rule-based filter inspecting solution signals vs. deflections.
  5. Saves two cleaned CSV files into the 'data/' folder:
     - inbound_customer_messages.csv (First customer messages to AmazonHelp)
     - resolved_pairs.csv (Customer message + Amazon response pairs for AI training/retrieval)

Usage:
  python 01_build_threads.py             (Fast Smart Heuristics filter)
  python 01_build_threads.py --use-llm   (AI-based Zero-Shot classification using HuggingFace)
"""

import argparse
import os
import re
import pandas as pd

# --- FILTER CONFIGURATION ---

# 1. Signals that indicate a response contains an actual solution or useful info
RESOLUTION_SIGNALS = [
    r"https?://\S+",                      # Links to help pages, troubleshooting, or forms
    r"return(s|ing)?\s+or\s+replacement",  # Return or replacement guidance
    r"replacement/refund",                # Replacement / refund options
    r"refund\s+options",                  # Refund guidance
    r"troubleshooting",                   # Direct troubleshooting instructions
    r"steps\s+(outlined|mentioned|here)", # Direct instructions
    r"transit\s+speed",                   # Policy explanations (e.g. 2-day shipping)
    r"business\s+days",                   # Business day policy explanations
    r"estimated\s+delivery\s+date",       # Tracking update info
]

# 2. Signals that indicate pure deflection / non-resolving boilerplate
DEFLECTION_PATTERNS = [
    r"unable to (access|affect) your account via twitter",
    r"(reach|contact) us by (phone|chat)",
    r"for real time support",
    r"please send us a dm",
    r"can you dm us",
    r"shoot us a dm",
    r"reach out to us directly",
    r"provide your details here:",
    r"without providing any personal information",
    r"we'?d like to (take a )?look into this with you",
    r"thanks for bringing this to our attention",
    r"^hi[, ]\s*\^\w+$",                    # Bare greeting with only staff signature
]

MIN_RESOLVING_REPLY_LEN = 30  # Anything shorter than 30 chars is almost never a resolution


def load_data(file_path: str) -> pd.DataFrame:
    """Reads the CSV file and ensures correct data types."""
    print(f"Loading CSV data from: {file_path} ...")

    if not os.path.exists(file_path):
        subfolder_path = os.path.join("Amazon_helpcare_twitter_bot", file_path)
        if os.path.exists(subfolder_path):
            file_path = subfolder_path
        else:
            raise FileNotFoundError(
                f"Could not find input file '{file_path}'. "
                "Please make sure twcs.csv is in the directory or pass --input path/to/twcs.csv"
            )

    df = pd.read_csv(
        file_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
    )

    if "inbound" in df.columns:
        df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])

    print(f"Successfully loaded {len(df):,} total rows.")
    return df


def clean_text(text: str) -> str:
    """Cleans tweet text by removing @mentions and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_boilerplate_heuristics(reply: str) -> bool:
    """
    Smart rule-based filter:
    - Keeps replies with concrete solution signals (links, return policies, instructions).
    - Removes pure deflection phrases (DM us, call/chat us, cannot access account).
    """
    if not isinstance(reply, str) or len(reply.strip()) < MIN_RESOLVING_REPLY_LEN:
        return True

    lowered = reply.strip().lower()

    is_deflection = any(re.search(pat, lowered) for pat in DEFLECTION_PATTERNS)
    has_resolution_signal = any(re.search(pat, lowered) for pat in RESOLUTION_SIGNALS)

    if has_resolution_signal and not is_deflection:
        return False

    if is_deflection:
        return True

    if len(reply) < 90 and re.match(r"^(we'?re|i'?m) sorry", lowered):
        return True

    return False


def classify_with_llm_batch(pairs_df: pd.DataFrame, batch_size: int = 64) -> list[bool]:
    """
    AI Model Classifier (HuggingFace Zero-Shot NLI Classification).
    Processes customer-reply pairs in GPU-accelerated batches to evaluate semantic intent.
    Returns a list of boolean flags (True = Boilerplate/Deflection, False = Resolving).
    """
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU (CUDA)" if device == 0 else "CPU"
        print(f"Initializing AI Zero-Shot Classifier model on {device_name}...")

        classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-xsmall",
            device=device
        )

        candidate_labels = [
            "Provides helpful solution or actionable steps",
            "Generic boilerplate apology or deflection to call/chat/DM"
        ]

        sequences = [
            f"Customer: {c} | Amazon Reply: {a}"
            for c, a in zip(pairs_df["customer_message"], pairs_df["amazon_reply"])
        ]

        is_boilerplate_flags = []
        total_rows = len(sequences)
        print(f"Running AI Classification on {total_rows:,} pairs (batch_size={batch_size})...")

        for i in range(0, total_rows, batch_size):
            batch = sequences[i:i + batch_size]
            results = classifier(batch, candidate_labels)

            # Standardize output format from pipeline batching
            if isinstance(results, dict):
                results = [results]

            for res in results:
                top_label = res["labels"][0]
                is_boilerplate_flags.append(top_label == "Generic boilerplate apology or deflection to call/chat/DM")

            if (i // batch_size) % 25 == 0 and i > 0:
                print(f"  Processed {min(i + batch_size, total_rows):,}/{total_rows:,} pairs...")

        return is_boilerplate_flags

    except ImportError:
        print("[Warning] 'transformers' or 'torch' module not found. Install via 'pip install transformers torch'.")
        print("Falling back to Smart Heuristics rule-based filter...")
        return [is_boilerplate_heuristics(r) for r in pairs_df["amazon_reply"]]
    except Exception as e:
        print(f"[Warning] AI classification error: {e}. Falling back to Smart Heuristics...")
        return [is_boilerplate_heuristics(r) for r in pairs_df["amazon_reply"]]


def build_threads(df: pd.DataFrame, brand: str, use_llm: bool = False):
    """Reconstructs customer support threads and pairs customer messages with brand replies."""
    df = df.copy()
    print("Cleaning text messages...")
    df["text_clean"] = df["text"].apply(clean_text)

    brand_replies = df[df["author_id"] == brand].copy()
    print(f"Found {len(brand_replies):,} tweets posted by {brand}.")

    by_id = df.set_index("tweet_id", drop=False)

    pairs = []
    for _, reply_row in brand_replies.iterrows():
        parent_id = reply_row.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue

        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]

        if not parent.get("inbound", True):
            continue

        pairs.append(
            {
                "customer_tweet_id": parent["tweet_id"],
                "customer_message": parent["text_clean"],
                "customer_created_at": parent.get("created_at"),
                "amazon_tweet_id": reply_row["tweet_id"],
                "amazon_reply": reply_row["text_clean"],
                "amazon_created_at": reply_row.get("created_at"),
                "customer_is_thread_start": pd.isna(parent.get("in_response_to_tweet_id")),
            }
        )

    pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["customer_tweet_id"])
    print(f"Reconstructed {len(pairs_df):,} customer -> {brand} reply pairs.")

    # 1. Inbound customer messages (first message in thread)
    inbound_msgs = pairs_df[pairs_df["customer_is_thread_start"]][
        ["customer_tweet_id", "customer_message", "customer_created_at"]
    ].rename(
        columns={
            "customer_tweet_id": "tweet_id",
            "customer_message": "text",
            "customer_created_at": "created_at",
        }
    )
    inbound_msgs = inbound_msgs[inbound_msgs["text"].str.len() > 5].reset_index(drop=True)
    print(f"Found {len(inbound_msgs):,} first-turn inbound customer messages.")

    # 2. Quality-filtered resolved pairs
    print("Filtering boilerplate & non-resolving replies...")
    if use_llm:
        print("Mode: AI Zero-Shot Model Classifier")
        pairs_df["is_boilerplate_reply"] = classify_with_llm_batch(pairs_df)
    else:
        print("Mode: Smart Heuristics Rule Filter")
        pairs_df["is_boilerplate_reply"] = pairs_df["amazon_reply"].apply(is_boilerplate_heuristics)

    resolved_pairs = pairs_df[~pairs_df["is_boilerplate_reply"]].reset_index(drop=True)

    dropped = len(pairs_df) - len(resolved_pairs)
    pct_dropped = (dropped / max(len(pairs_df), 1)) * 100
    print(f"Dropped {dropped:,} boilerplate/non-resolving replies ({pct_dropped:.1f}% of total).")
    print(f"Kept {len(resolved_pairs):,} quality resolved pairs for retrieval.")

    return inbound_msgs, resolved_pairs


def main():
    parser = argparse.ArgumentParser(description="Build customer support conversation threads from Twitter CS dataset.")
    parser.add_argument("--input", default="twcs.csv", help="Path to twcs.csv (default: twcs.csv)")
    parser.add_argument("--brand", default="AmazonHelp", help="Twitter support account author_id (default: AmazonHelp)")
    parser.add_argument("--outdir", default="data", help="Output directory to save results (default: data)")
    parser.add_argument("--use-llm", action="store_true", help="Enable AI Zero-Shot Model classification pass")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.input)
    inbound_msgs, resolved_pairs = build_threads(df, args.brand, use_llm=args.use_llm)

    inbound_path = os.path.join(args.outdir, "inbound_customer_messages.csv")
    pairs_path = os.path.join(args.outdir, "resolved_pairs.csv")

    inbound_msgs.to_csv(inbound_path, index=False)
    resolved_pairs.to_csv(pairs_path, index=False)

    print("\nProcessing complete!")
    print(f"  -> Saved: {inbound_path} ({len(inbound_msgs):,} rows)")
    print(f"  -> Saved: {pairs_path} ({len(resolved_pairs):,} rows)")


if __name__ == "__main__":
    main()
