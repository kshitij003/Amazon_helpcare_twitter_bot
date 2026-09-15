# AmazonHelp AI Support Agent

An automated customer support bot built to handle `@AmazonHelp` Twitter customer inquiries. It combines fast machine learning intent classification, RAG vector retrieval, safety escalation rules, and a local LLM (`llama3.2:3b` via Ollama) to draft grounded support replies.

---

## Overview

Handling high-volume customer support on Twitter requires balancing speed, accuracy, and safety. A small ML classifier is fast but can be overconfident or confused on tricky inputs. An LLM is smart but slow and prone to hallucinating facts or copying stray links from past examples.

This project uses a hybrid architecture:
1. **Language Check**: Filters out non-English queries right away.
2. **Intent Classifier + LLM Tie-Breaker**: Uses a fast Scikit-Learn Logistic Regression model on sentence embeddings. If confidence is low (< 0.55) or the gap between top-1 and top-2 predicted classes is narrow (< 0.15), it calls the local LLM to disambiguate the intent.
3. **Escalation Rules**: Instantly routes sensitive issues (account security, open follow-ups) to human agents.
4. **RAG Vector Search**: Queries a local ChromaDB vector database for past resolved support interactions to ground the reply.
5. **Grounded Reply Drafting**: Prompts Ollama (`llama3.2:3b`) with sanitized precedent examples to draft a Twitter-ready response under 280 characters, attaching verified Amazon Help links automatically.

---

## Repository Structure

```text
Amazon_helpcare_twitter_bot/
├── 01_build_threads.py           # Extracts customer-agent message pairs from TWCS dataset
├── 02_cluster_for_taxonomy.py    # Unsupervised embedding clustering to build intent taxonomy
├── 03_classify_intents.py        # Trains the Logistic Regression intent classifier
├── 04_tag_and_index.py           # Embeds resolved support pairs into a local ChromaDB database
├── 05_agent.py                   # Main CLI script running the live agent loop
├── README.md                     # Documentation
├── twcs.csv                      # Raw Kaggle Twitter Customer Support dataset
└── data/                         # Data directory (models, ChromaDB, processed CSVs)
```

---

## Pipeline Breakdown

### 1. Thread Building (`01_build_threads.py`)
Parses the raw `twcs.csv` dataset, filters for tweets addressed to `@AmazonHelp`, pairs customer questions with Amazon's replies, and removes low-quality or non-resolving boilerplate responses.

### 2. Intent Taxonomy Clustering (`02_cluster_for_taxonomy.py`)
Encodes customer messages with `all-MiniLM-L6-v2` embeddings and clusters them into distinct intent categories (e.g. delivery issues, order cancellations, billing refunds, device troubleshooting).

### 3. Intent Classifier Training (`03_classify_intents.py`)
Trains a Logistic Regression model on sentence embeddings to classify incoming customer tweets into one of 8 intent categories:
- `delivery_issue`
- `order_issue`
- `billing_refund`
- `account_access` *(Escalated to human support)*
- `device_technical_issue`
- `general_complaint`
- `follow_up_unresolved` *(Escalated to human support)*
- `other_non_english` *(Escalated to human support)*

### 4. Vector Database Indexing (`04_tag_and_index.py`)
Indexes resolved support pairs into a local persistent ChromaDB collection (`amazon_support_pairs`) for vector similarity search during inference.

### 5. Live Support Agent (`05_agent.py`)
Runs the interactive support loop:
- **Language Detection**: Escalates non-English messages immediately.
- **Intent Prediction & Margin Check**: Computes top-1 probability and margin (`top1_prob - top2_prob`). If confidence < 0.55 or margin < 0.15, triggers the LLM tie-breaker.
- **Escalation Logic**: Evaluates business safety rules.
- **Precedent Retrieval**: Queries ChromaDB for 3 similar past resolved cases matching the intent.
- **Sanitized Reply Generation**: Strips stray URLs and agent initials (`^JO`, `-AB`) from historical examples, prompts Ollama for a short reply, cleans surrounding quotes, and appends a verified Amazon Help link if within the 280-character limit.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Install dependencies:
  ```bash
  pip install pandas sentence-transformers langdetect joblib chromadb requests scikit-learn numpy
  ```

### 2. Install and Start Ollama
Download Ollama from [ollama.com](https://ollama.com), then start the server and pull the model:
```bash
ollama serve
ollama pull llama3.2:3b
```

### 3. Run the Agent CLI
In a separate terminal window:
```bash
python 05_agent.py
```

---

## Example Usage

```text
Customer: @AmazonHelp my package says delivered but it's not here, driver probably left it at wrong house

Agent Output:
{
  "customer_message": "@AmazonHelp my package says delivered but it's not here, driver probably left it at wrong house",
  "detected_language": "en",
  "predicted_intent": "delivery_issue",
  "intent_confidence": 0.9626,
  "intent_margin": 0.9312,
  "second_place_intent": "order_issue",
  "used_llm_tiebreaker": false,
  "retrieved_case_count": 3,
  "escalate": false,
  "escalation_reason": "Clear intent with strong historical precedent",
  "draft_reply": "We're sorry to hear your package hasn't arrived. Please send us a DM with your order details so we can look into this for you! https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7"
}
```

---

## License & Data Credits

Dataset sourced from the public Kaggle Twitter Customer Support (`twcs.csv`) dataset.
