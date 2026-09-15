"""
04_tag_and_index.py

Step 4 of the AmazonHelp support-agent pipeline.

What this script does:
  1. Loads clean customer-reply resolved pairs (data/resolved_pairs.csv).
  2. Detects message language using langdetect (non-English flagged as 'other_non_english').
  3. Loads pre-trained intent classifier (intent_classifier.joblib) and encodes customer messages
     using sentence-transformers (all-MiniLM-L6-v2) to predict intent categories.
  4. Stores embeddings, text, and metadata (intent, amazon_reply, customer_tweet_id) into a local
     ChromaDB vector database collection ('amazon_support_pairs') in efficient batches.
  5. Persists the database to disk (data/chroma_db) for fast downstream retrieval.
  6. Prints progress and a final breakdown count of indexed pairs per intent category.
  7. Includes an example query function demonstrating intent-filtered similarity retrieval.

How to run:
  python 04_tag_and_index.py
  python 04_tag_and_index.py --pairs data/resolved_pairs.csv --batch-size 500

Outputs:
  - data/chroma_db/  (Local persistent Chroma vector database)
"""

import argparse
import os
import chromadb
import joblib
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer

# Ensure deterministic language detection results
DetectorFactory.seed = 0


def resolve_file_path(path_str: str) -> str:
    """Resolves relative file path whether running from project root or subfolder."""
    if not os.path.exists(path_str):
        subfolder_path = os.path.join("Amazon_helpcare_twitter_bot", path_str)
        if os.path.exists(subfolder_path):
            return subfolder_path
        raise FileNotFoundError(f"Required file or folder not found at '{path_str}'.")
    return path_str


def detect_language(text: str) -> str:
    """Detects text language, returning 'unknown' if detection fails or text is too short."""
    try:
        text_str = str(text).strip()
        if len(text_str) < 4:
            return "unknown"
        return detect(text_str)
    except Exception:
        return "unknown"


def query_similar_cases(
    query_text: str,
    target_intent: str,
    chroma_dir: str = "data/chroma_db",
    top_k: int = 3
):
    """
    Example function demonstrating how to query the Chroma vector database for similar
    past customer support cases filtered by a specific predicted intent category.
    """
    chroma_path = resolve_file_path(chroma_dir)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name="amazon_support_pairs")

    # Encode user query text using the same embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query_text])[0].tolist()

    # Query Chroma collection filtered by intent metadata
    results = collection.query(
        query_embeddings=[query_embedding],
        where={"intent": target_intent},
        n_results=top_k
    )

    print(f"\n=========================================")
    print(f" SIMILAR CASE RETRIEVAL DEMO             ")
    print(f" Query: '{query_text}'                   ")
    print(f" Filter Intent: {target_intent}          ")
    print(f"=========================================")

    if results and results.get("documents") and len(results["documents"][0]) > 0:
        for idx in range(len(results["ids"][0])):
            doc = results["documents"][0][idx]
            metadata = results["metadatas"][0][idx]
            dist = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
            print(f"\n--- Match #{idx + 1} (Distance: {dist:.4f}) ---")
            print(f"  Customer Message : {doc}")
            print(f"  Amazon Reply     : {metadata.get('amazon_reply')}")
    else:
        print("No matching cases found for this query.")
    print("=========================================\n")


def main():
    parser = argparse.ArgumentParser(description="Tag resolved support pairs with intent and index into ChromaDB.")
    parser.add_argument("--pairs", default="data/resolved_pairs.csv", help="Path to resolved pairs CSV")
    parser.add_argument("--model-dir", default="data", help="Directory containing trained classifier artifacts")
    parser.add_argument("--chroma-dir", default="data/chroma_db", help="Directory to save persistent Chroma database")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for vector DB additions (default: 500)")
    args = parser.parse_args()

    # Step 1: Load resolved pairs CSV
    pairs_file_path = resolve_file_path(args.pairs)
    print(f"Loading resolved pairs dataset from: {pairs_file_path}")
    pairs_df = pd.read_csv(pairs_file_path)

    # Ensure text fields are string types
    pairs_df["customer_message"] = pairs_df["customer_message"].fillna("").astype(str)
    pairs_df["amazon_reply"] = pairs_df["amazon_reply"].fillna("").astype(str)
    print(f"Loaded {len(pairs_df):,} total customer-reply pairs.")

    # Step 2: Detect language for each customer message
    print(f"Detecting message languages for {len(pairs_df):,} rows...")
    pairs_df["detected_language"] = [detect_language(msg) for msg in pairs_df["customer_message"]]

    is_english = pairs_df["detected_language"] == "en"
    non_english_count = (~is_english).sum()
    print(f"Found {is_english.sum():,} English messages and {non_english_count:,} non-English messages.")

    # Step 3: Embed messages and predict intents with pre-trained classifier
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating sentence embeddings for all {len(pairs_df):,} messages...")
    all_embeddings = embedding_model.encode(
        pairs_df["customer_message"].tolist(),
        batch_size=256,
        show_progress_bar=True
    )

    classifier_path = resolve_file_path(os.path.join(args.model_dir, "intent_classifier.joblib"))
    print(f"Loading trained intent classifier from: {classifier_path}")
    classifier = joblib.load(classifier_path)

    # Default all rows to 'other_non_english'
    pairs_df["predicted_intent"] = "other_non_english"

    # Predict intents for English messages using embeddings
    if is_english.sum() > 0:
        english_indices = np.where(is_english)[0]
        english_embeddings = all_embeddings[english_indices]
        predicted_english_intents = classifier.predict(english_embeddings)
        pairs_df.loc[is_english, "predicted_intent"] = predicted_english_intents

    # Step 4 & 5: Initialize ChromaDB persistent client and collection
    chroma_output_dir = args.chroma_dir
    if not os.path.isabs(chroma_output_dir) and not os.path.exists("data"):
        chroma_output_dir = os.path.join("Amazon_helpcare_twitter_bot", chroma_output_dir)

    os.makedirs(chroma_output_dir, exist_ok=True)
    print(f"Initializing local persistent Chroma vector DB at: {chroma_output_dir}")
    chroma_client = chromadb.PersistentClient(path=chroma_output_dir)
    collection = chroma_client.get_or_create_collection(name="amazon_support_pairs")

    # Step 6 & 7: Batch insert resolved pairs into ChromaDB collection
    total_rows = len(pairs_df)
    batch_size = args.batch_size
    print(f"Indexing {total_rows:,} pairs into Chroma collection 'amazon_support_pairs' (batch size = {batch_size})...")

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = pairs_df.iloc[start_idx:end_idx]
        batch_embeds = all_embeddings[start_idx:end_idx].tolist()

        # Generate unique string IDs for Chroma items
        batch_ids = [f"{row['customer_tweet_id']}_{start_idx + idx}" for idx, (_, row) in enumerate(batch_df.iterrows())]
        batch_documents = batch_df["customer_message"].tolist()
        
        batch_metadatas = [
            {
                "intent": str(row["predicted_intent"]),
                "amazon_reply": str(row["amazon_reply"]),
                "customer_tweet_id": str(row["customer_tweet_id"]),
            }
            for _, row in batch_df.iterrows()
        ]

        # Add batch to vector collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeds,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

        if (start_idx // batch_size) % 10 == 0 or end_idx == total_rows:
            print(f"  Indexed {end_idx:,} / {total_rows:,} rows...")

    # Step 8: Print indexing summary and breakdown
    print("\n=========================================")
    print(" VECTOR DATABASE INDEXING SUMMARY        ")
    print("=========================================")
    print(f" Total Rows Indexed : {collection.count():,}")
    print(" Breakdown by Intent Category:")

    intent_counts = pairs_df["predicted_intent"].value_counts()
    for intent_name, count in intent_counts.items():
        pct = (count / total_rows) * 100
        print(f"  {intent_name:<25} : {count:6,} ({pct:5.1f}%)")
    print("=========================================\n")


if __name__ == "__main__":
    main()

    # To test similarity retrieval example, uncomment the line below after indexing:
    # query_similar_cases(
    #     query_text="My package hasn't arrived yet and it's 2 days late",
    #     target_intent="delivery_issue",
    #     top_k=3
    # )
