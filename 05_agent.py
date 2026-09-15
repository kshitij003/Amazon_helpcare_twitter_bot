"""
05_agent.py - Live Customer Support AI Agent

This script runs an automated customer support agent for AmazonHelp.
Given a customer message, it:
  1. Detects the language (non-English messages are immediately escalated).
  2. Classifies customer intent using a pre-trained Logistic Regression model and sentence embeddings.
     If classifier confidence is low (< 0.55) OR top-1 vs top-2 probability margin is narrow (< 0.15),
     an LLM tie-breaker is invoked before escalating.
  3. Retrieves top matching historical resolved cases from a local ChromaDB vector database.
  4. Applies clear business rules to decide whether to auto-handle or escalate to a human agent.
  5. If auto-handling, drafts a grounded Twitter support reply using a local Ollama LLM (llama3.2:3b).

Required Local Setup:
  - Install Ollama: https://ollama.com
  - Start Ollama service: run 'ollama serve' in a terminal
  - Pull model: run 'ollama pull llama3.2:3b' in a terminal
  - Dependencies: pip install pandas sentence-transformers langdetect joblib chromadb requests

How to run:
  python 05_agent.py
"""

import json
import os
import re
import chromadb
import joblib
import numpy as np
import requests
from langdetect import DetectorFactory, detect
from sentence_transformers import SentenceTransformer

# Ensure deterministic language detection results
DetectorFactory.seed = 0

# Starting value for margin threshold; tune based on real misclassified examples
# (i.e. difference between top-1 and top-2 predicted class probabilities)
MARGIN_THRESHOLD = 0.15

# Double-check / replace these with real verified links before using in demo
INTENT_HELP_LINKS = {
    "delivery_issue": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7",
    "billing_refund": "https://www.amazon.com/gp/help/customer/display.html?nodeId=GRFRJ8UDDMKTQCMQ",
    "order_issue": "https://www.amazon.com/gp/css/order-history",
    "device_technical_issue": "https://www.amazon.com/gp/help/customer/display.html?nodeId=201952840",
    "general_complaint": "https://www.amazon.com/gp/help/customer/contact-us",
}

# Plain-English descriptions for each supported intent label
VALID_INTENT_DESCRIPTIONS = {
    "delivery_issue": "problem with delivery timing or location",
    "order_issue": "problem with the order itself like cancellation or wrong seller",
    "billing_refund": "charged incorrectly or refund problems",
    "account_access": "login or account lockout problems",
    "device_technical_issue": "Alexa/Echo/Prime Video/Music technical problems",
    "general_complaint": "frustration with support with no clear specific issue",
    "follow_up_unresolved": "customer is following up on an already-open unresolved issue",
}


def resolve_path(path_str: str) -> str:
    """Resolves relative file path whether running from project root or subfolder."""
    if os.path.exists(path_str):
        return path_str
    for prefix in ["Amazon_helpcare_twitter_bot", ".."]:
        alt = os.path.join(prefix, path_str)
        if os.path.exists(alt):
            return alt
    return path_str


# Initialize pre-trained intent classifier, embedding model, and vector database
CLASSIFIER_PATH = resolve_path("data/intent_classifier.joblib")
CHROMA_PATH = resolve_path("data/chroma_db")

classifier = joblib.load(CLASSIFIER_PATH)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_collection(name="amazon_support_pairs")


def get_few_shot_examples_per_intent(intent_list: list) -> dict:
    """
    Queries ChromaDB collection for 2 historical example customer messages per intent label.
    Excludes 'other_non_english' as that is handled by language detection.
    """
    examples = {}
    for intent in intent_list:
        if intent == "other_non_english":
            continue
        try:
            results = chroma_collection.get(where={"intent": intent}, limit=2)
            docs = results.get("documents", [])
            examples[intent] = docs[:2]
        except Exception:
            examples[intent] = []
    return examples


# Cache few-shot examples once at module startup
FEW_SHOT_EXAMPLES = get_few_shot_examples_per_intent(list(VALID_INTENT_DESCRIPTIONS.keys()))


def call_local_llm(prompt: str) -> str:
    """Sends prompt to local Ollama service (llama3.2:3b) and returns response text."""
    url = "http://127.0.0.1:11434/api/generate"
    payload = {"model": "llama3.2:3b", "prompt": prompt, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        # Fallback error string if Ollama fails or times out
        return "[LLM unavailable]"


def strip_urls_and_signoffs(text: str) -> str:
    """
    Removes any URLs (http/https/t.co links) and short trailing agent sign-offs
    (patterns like '^XX' or '-XX' at the end of a string, 2-3 uppercase letters)
    from historical examples before inserting them into the LLM prompt.
    """
    if not text:
        return ""
    # Remove http/https and t.co URLs
    text = re.sub(r'https?://\S+|t\.co/\S+', '', text, flags=re.IGNORECASE)
    # Remove short trailing agent sign-offs at the end of the text (e.g. ^JO, -AB)
    text = re.sub(r'\s*[\^|-][A-Z]{2,3}\s*$', '', text.strip())
    # Clean up leftover multiple whitespace spaces
    return re.sub(r'\s+', ' ', text).strip()


def clean_llm_output(text: str) -> str:
    """
    Strips leading/trailing whitespace and any leading/trailing quote characters
    (", ', ”, “, ’, ‘) from the response text.
    """
    if not text:
        return ""
    cleaned = text.strip()
    quote_chars = '"\'“”-------------‘’'
    return cleaned.strip(quote_chars).strip()


def llm_classify_intent(customer_message: str, few_shot_examples: dict) -> str:
    """
    Uses local LLM to disambiguate intent when the classifier confidence score is low (< 0.55)
    or when the top-1 vs top-2 probability margin is narrow (< MARGIN_THRESHOLD).
    Returns the exact matching intent label string or 'unclear' if ambiguous/invalid.
    """
    # Build list of valid intents and plain-English descriptions
    intent_desc_str = ""
    for intent, desc in VALID_INTENT_DESCRIPTIONS.items():
        intent_desc_str += f"- {intent}: {desc}\n"

    # Build few-shot examples block
    examples_block = ""
    for intent, msgs in few_shot_examples.items():
        examples_block += f"\nIntent Category: {intent}\n"
        for i, msg in enumerate(msgs, 1):
            examples_block += f"  Example {i}: \"{msg}\"\n"

    prompt = (
        f"You are a customer support intent classifier. Classify the following customer message into EXACTLY ONE of these valid intent categories:\n"
        f"{intent_desc_str}\n"
        f"Here are example customer messages for each category:\n"
        f"{examples_block}\n"
        f"Customer Message to classify: \"{customer_message}\"\n\n"
        f"Task: Respond with ONLY the single matching intent label (e.g. delivery_issue). Do NOT include preamble, punctuation, quotes, or explanations."
    )

    raw_response = call_local_llm(prompt)
    cleaned_intent = clean_llm_output(raw_response)

    if cleaned_intent in VALID_INTENT_DESCRIPTIONS:
        return cleaned_intent
    return "unclear"


def handle_message(customer_message: str) -> dict:
    """Processes customer message, predicts intent, retrieves precedent, decides escalation, and drafts reply."""
    # Step 1: Detect language (escalate immediately if non-English)
    try:
        text_clean = str(customer_message).strip()
        detected_lang = detect(text_clean) if len(text_clean) >= 4 else "unknown"
    except Exception:
        detected_lang = "unknown"

    if detected_lang != "en":
        return {
            "customer_message": customer_message,
            "detected_language": detected_lang,
            "predicted_intent": "other_non_english",
            "intent_confidence": 1.0,
            "intent_margin": 1.0,
            "second_place_intent": None,
            "used_llm_tiebreaker": False,
            "retrieved_case_count": 0,
            "escalate": True,
            "escalation_reason": "Message is not in English; outside current agent scope",
            "draft_reply": None
        }

    # Step 2: Embed message and classify intent with confidence score & margin calculation
    embedding = embedder.encode([customer_message])[0]
    probs = classifier.predict_proba([embedding])[0]

    # Sort class probabilities in descending order
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx

    predicted_intent = str(classifier.classes_[top1_idx])
    second_place_intent = str(classifier.classes_[top2_idx]) if len(sorted_indices) > 1 else predicted_intent

    intent_confidence = float(probs[top1_idx])
    top2_confidence = float(probs[top2_idx])
    intent_margin = float(intent_confidence - top2_confidence)

    # Distinguish trigger reason internally for debugging/logging
    low_confidence = intent_confidence < 0.55
    low_margin = intent_margin < MARGIN_THRESHOLD

    if low_confidence and low_margin:
        tiebreaker_trigger_reason = "both"
    elif low_confidence:
        tiebreaker_trigger_reason = "low_confidence"
    elif low_margin:
        tiebreaker_trigger_reason = "low_margin"
    else:
        tiebreaker_trigger_reason = None

    used_llm_tiebreaker = False
    llm_tiebreaker_unclear = False

    # LLM Tie-breaker: Triggered when confidence is low (< 0.55) OR margin is narrow (< MARGIN_THRESHOLD)
    if tiebreaker_trigger_reason is not None:
        used_llm_tiebreaker = True
        tiebreaker_intent = llm_classify_intent(customer_message, FEW_SHOT_EXAMPLES)
        if tiebreaker_intent in VALID_INTENT_DESCRIPTIONS:
            predicted_intent = tiebreaker_intent
        else:
            llm_tiebreaker_unclear = True

    # Step 3: Query ChromaDB for top 3 similar past cases matching predicted intent
    retrieved_cases = []
    try:
        results = chroma_collection.query(
            query_embeddings=[embedding.tolist()],
            where={"intent": predicted_intent},
            n_results=3
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            retrieved_cases.append({"customer_message": doc, "amazon_reply": meta.get("amazon_reply", "")})
    except Exception:
        retrieved_cases = []

    retrieved_case_count = len(retrieved_cases)

    # Step 4: Escalation decision rules (checked in strict priority order, first match wins)
    # Rules filter out sensitive, out-of-scope, low-confidence, or poorly grounded cases.
    if predicted_intent == "other_non_english":
        escalate = True
        escalation_reason = "Non-English message, outside current scope"
    elif predicted_intent == "account_access":
        escalate = True
        escalation_reason = "Account security issue requires human verification"
    elif predicted_intent == "follow_up_unresolved":
        escalate = True
        escalation_reason = "Customer is following up on an already-open issue"
    elif llm_tiebreaker_unclear:
        escalate = True
        escalation_reason = "Both classifier and LLM tie-breaker were uncertain about intent"
    elif (low_confidence or low_margin) and not used_llm_tiebreaker:
        escalate = True
        escalation_reason = "Low classifier confidence or narrow margin in predicted intent"
    elif retrieved_case_count < 2:
        escalate = True
        escalation_reason = "Insufficient historical precedent to draft a grounded reply"
    else:
        escalate = False
        escalation_reason = "Clear intent with strong historical precedent"

    # Step 5: If not escalating, draft reply using local Ollama model
    draft_reply = None
    if not escalate:
        examples_str = ""
        for idx, case in enumerate(retrieved_cases, 1):
            clean_reply = strip_urls_and_signoffs(case.get("amazon_reply", ""))
            examples_str += f"Example {idx}:\nCustomer: {case['customer_message']}\nAmazonHelp: {clean_reply}\n\n"

        prompt = (
            f"You are an AmazonHelp Twitter support agent. Below are past resolved customer support examples for similar issues:\n\n"
            f"{examples_str}"
            f"New Customer Message: \"{customer_message}\"\n\n"
            f"Task: Write a short, on-brand, helpful Twitter reply (under 280 characters) to the new customer message, consistent in tone and approach with the examples above.\n"
            f"Rules:\n"
            f"- Do NOT invent specific facts (order numbers, refund amounts, tracking info, dates) not present in the customer message or examples.\n"
            f"- Do not include any URLs or links in your reply. Do not sign off with initials or a name.\n"
            f"- Output ONLY the final Twitter reply text. Do NOT include preamble, headers, quotes, or explanations."
        )
        raw_reply = call_local_llm(prompt)
        draft_reply = clean_llm_output(raw_reply)

        # If predicted intent has a verified help link, append it if combined length <= 280 chars
        if predicted_intent in INTENT_HELP_LINKS:
            help_link = INTENT_HELP_LINKS[predicted_intent]
            candidate_reply = f"{draft_reply} {help_link}"
            # Known limitation: If adding the link causes the total reply length to exceed 280 characters,
            # we leave the link off rather than truncating the response text. Note this in write-ups.
            if len(candidate_reply) <= 280:
                draft_reply = candidate_reply

    # Step 6: Return compiled agent output dictionary
    return {
        "customer_message": customer_message,
        "detected_language": detected_lang,
        "predicted_intent": predicted_intent,
        "intent_confidence": round(intent_confidence, 4),
        "intent_margin": round(intent_margin, 4),
        "second_place_intent": second_place_intent,
        "used_llm_tiebreaker": used_llm_tiebreaker,
        "retrieved_case_count": retrieved_case_count,
        "escalate": escalate,
        "escalation_reason": escalation_reason,
        "draft_reply": draft_reply
    }


if __name__ == "__main__":
    # Startup check: verify local Ollama HTTP endpoint is reachable
    try:
        res = requests.get("http://127.0.0.1:11434", timeout=3)
        if res.status_code != 200:
            print(f"[WARNING] Ollama server returned status code {res.status_code}.")
    except Exception:
        print("\n[WARNING] Could not connect to local Ollama service at http://127.0.0.1:11434.")
        print("Please ensure Ollama is installed and running ('ollama serve') with model 'llama3.2:3b'.\n")

    print("\n=======================================================")
    print("      AmazonHelp Support Agent - Interactive CLI       ")
    print("=======================================================")
    print("Type a customer message to test the agent, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting agent CLI.")
            break

        if not user_input or user_input.lower() == "quit":
            print("Exiting agent CLI.")
            break

        output = handle_message(user_input)
        print("\nAgent Output:")
        print(json.dumps(output, indent=2))
        print("-" * 55 + "\n")
