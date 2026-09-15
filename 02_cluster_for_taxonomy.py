"""
02_cluster_for_taxonomy.py

Step 2 of the AmazonHelp support-agent pipeline.

What this script does:
  1. Loads customer support messages from 'inbound_customer_messages.csv'.
  2. Takes a random sample of messages (default: 3,000) for fast processing.
  3. Converts text into numerical embeddings using a small offline model (all-MiniLM-L6-v2).
  4. Groups similar messages into clusters using the KMeans algorithm.
  5. Finds the most representative example messages for each cluster.
  6. Saves a human-readable text report ('cluster_report.txt') and a CSV file ('clustered_sample.csv').
  7. Prints clear next-step instructions for manual intent taxonomy discovery.

How to run:
  python 02_cluster_for_taxonomy.py
  python 02_cluster_for_taxonomy.py --sample-size 5000 --num-clusters 15 --examples-per-cluster 10

Outputs:
  - data/cluster_report.txt    (Text report summarizing each cluster size and top representative examples)
  - data/clustered_sample.csv  (CSV containing sampled customer messages + assigned cluster_id)
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def load_input_data(input_path: str) -> pd.DataFrame:
    """Loads the input CSV file, handling relative directory paths if needed."""
    # Check if file exists at specified path or inside subfolder
    if not os.path.exists(input_path):
        subfolder_path = os.path.join("Amazon_helpcare_twitter_bot", input_path)
        if os.path.exists(subfolder_path):
            input_path = subfolder_path
        else:
            raise FileNotFoundError(f"Input file not found at '{input_path}'.")

    print(f"Loading customer messages from: {input_path}")
    dataframe = pd.read_csv(input_path)
    
    # Ensure text column contains valid strings
    dataframe["text"] = dataframe["text"].fillna("").astype(str)
    print(f"Loaded {len(dataframe):,} total messages.")
    return dataframe


def sample_data(dataframe: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Takes a random sample of N messages from the dataset."""
    # If dataset has fewer rows than requested sample size, use all rows
    actual_sample_size = min(sample_size, len(dataframe))
    print(f"Taking a random sample of {actual_sample_size:,} messages...")
    
    # Use fixed random_state for reproducible sampling
    sampled_df = dataframe.sample(n=actual_sample_size, random_state=42).reset_index(drop=True)
    return sampled_df


def generate_embeddings(text_list: list[str]) -> np.ndarray:
    """Converts a list of text messages into numerical vector embeddings."""
    print("Loading offline embedding model (all-MiniLM-L6-v2)...")
    # Load small, fast sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Generating embeddings for {len(text_list):,} messages...")
    # Convert text messages into dense vector representations
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings


def perform_clustering(embeddings: np.ndarray, num_clusters: int) -> tuple[KMeans, np.ndarray]:
    """Clusters vector embeddings into K groups using KMeans."""
    print(f"Clustering messages into {num_clusters} groups with KMeans...")
    # Initialize KMeans clustering algorithm
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    
    # Fit model on embeddings and get cluster label for each message
    cluster_labels = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def find_representative_examples(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    text_list: list[str],
    cluster_id: int,
    top_n: int
) -> list[str]:
    """Finds top N messages in a cluster closest to the cluster center."""
    # Get indices of all messages assigned to this cluster
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    
    # Extract embeddings for messages in this cluster
    cluster_embeddings = embeddings[cluster_indices]
    
    # Calculate Euclidean distance from each message embedding to the cluster center
    center = cluster_centers[cluster_id]
    distances = np.linalg.norm(cluster_embeddings - center, axis=1)
    
    # Sort messages by distance (closest first)
    closest_relative_indices = np.argsort(distances)[:top_n]
    
    # Map back to original text messages
    top_examples = [text_list[cluster_indices[i]] for i in closest_relative_indices]
    return top_examples


def save_results(
    sampled_df: pd.DataFrame,
    embeddings: np.ndarray,
    kmeans: KMeans,
    cluster_labels: np.ndarray,
    num_clusters: int,
    examples_per_cluster: int,
    output_dir: str
):
    """Saves the cluster report text file and the clustered CSV sample."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Assign cluster labels to the sampled dataframe
    sampled_df["cluster_id"] = cluster_labels
    
    # 1. Save clustered CSV
    csv_output_path = os.path.join(output_dir, "clustered_sample.csv")
    sampled_df.to_csv(csv_output_path, index=False)
    print(f"Saved clustered dataset to: {csv_output_path}")

    # 2. Build text report
    report_output_path = os.path.join(output_dir, "cluster_report.txt")
    text_list = sampled_df["text"].tolist()
    
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        report_file.write("=========================================\n")
        report_file.write(" CUSTOMER SUPPORT INTENT CLUSTER REPORT   \n")
        report_file.write("=========================================\n\n")
        
        for cluster_id in range(num_clusters):
            # Calculate how many messages belong to this cluster
            cluster_size = int(np.sum(cluster_labels == cluster_id))
            
            report_file.write(f"--- CLUSTER {cluster_id} ({cluster_size:,} messages) ---\n")
            
            # Find closest example messages for this cluster
            examples = find_representative_examples(
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                cluster_centers=kmeans.cluster_centers_,
                text_list=text_list,
                cluster_id=cluster_id,
                top_n=examples_per_cluster
            )
            
            for idx, msg in enumerate(examples, 1):
                report_file.write(f"  {idx}. {msg}\n")
            report_file.write("\n")

    print(f"Saved human-readable report to: {report_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster customer tweets to discover intent categories.")
    parser.add_argument("--input", default="data/inbound_customer_messages.csv", help="Path to inbound messages CSV")
    parser.add_argument("--sample-size", type=int, default=3000, help="Number of messages to sample (default: 3000)")
    parser.add_argument("--num-clusters", type=int, default=12, help="Number of clusters K (default: 12)")
    parser.add_argument("--examples-per-cluster", type=int, default=8, help="Top example messages per cluster (default: 8)")
    parser.add_argument("--outdir", default="data", help="Output directory to save results (default: data)")
    args = parser.parse_args()

    # Step 1: Load CSV data
    dataframe = load_input_data(args.input)
    
    # Step 2: Random sample of N messages
    sampled_df = sample_data(dataframe, args.sample_size)
    
    # Step 3: Convert text to embeddings
    text_list = sampled_df["text"].tolist()
    embeddings = generate_embeddings(text_list)
    
    # Step 4: Cluster embeddings with KMeans
    kmeans, cluster_labels = perform_clustering(embeddings, args.num_clusters)
    
    # Step 5 & 6: Find representative examples and save output files
    save_results(
        sampled_df=sampled_df,
        embeddings=embeddings,
        kmeans=kmeans,
        cluster_labels=cluster_labels,
        num_clusters=args.num_clusters,
        examples_per_cluster=args.examples_per_cluster,
        output_dir=args.outdir
    )
    
    # Step 7: Print next-step instructions for intent taxonomy discovery
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR MANUAL INTENT TAXONOMY DISCOVERY:")
    print("=" * 60)
    print("1. Open the generated report:")
    print(f"   -> {os.path.join(args.outdir, 'cluster_report.txt')}")
    print("2. Read through the representative example messages for each cluster.")
    print("3. Manually assign a short intent label to each cluster number.")
    print("   (e.g., Cluster 0 -> 'Refund Request', Cluster 1 -> 'Delivery Delay')")
    print("4. Use these intent categories to build your taxonomy or classification prompt!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
