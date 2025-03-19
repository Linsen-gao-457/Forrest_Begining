import logging
from openai import AzureOpenAI
import json
from datasets import load_dataset
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Initialize Azure OpenAI client
client =
)

def load_npy_file(file_path):
    """Load the .npy file containing training batches."""
    try:
        training_batches = np.load(file_path, allow_pickle=True)
        logger.info(f"Loaded {len(training_batches)} batches from {file_path}")
        return training_batches
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_dataset(dataset):
    """
    Process the dataset to extract queries, positives, and corpus.

    Returns:
        queries (dict): {query_id: query_text}
        positives (dict): {query_id: positive_docid}
        corpus (dict): {passage_id: passage_text}
    """
    data = dataset["train"]
    queries = {item["_id"]: item["query"] for item in data}
    positives = {item["_id"]: item["_id"] for item in data}
    corpus = {item["_id"]: item["title"] + item["text"] for item in data}
    return queries, positives, corpus

def create_relevance_prompt(query_id: str, query_text: str, doc_id: str, corpus) -> str:
    """Create a prompt for the OpenAI model to assess document relevance."""
    doc_content = corpus.get(doc_id, "Content not found")
    prompt = (
        f"Query ID: '{query_id}'\n"
        f"Query Text: '{query_text}'\n"
        f"Doc ID: '{doc_id}'\n"
        f"Doc Content: '{doc_content}'\n"
        f"Task\n"
        f"Determine whether this document is relevant or irrelevant to the given query, the positive ID may be negative.(the positive doc not always relevant, but can be a reference).\n"
        f"Provide:\n"
        f"Document ID: [Insert Document ID]\n"
        f"Justification: a brief reason for the decision\n"
        f"Relevance: [Relevant / Irrelevant]\n"
        f"Please respond in the following format:\n"
        f"Document ID: [the document ID]\n"
        f"Justification: [a brief reason for your decision]\n"
        f"Relevance: [Relevant or Irrelevant]\n"
    )
    return prompt

def parse_response(response_text):
    """Parse the model's response to extract document ID, justification, and relevance."""
    lines = response_text.split("\n")
    if len(lines) < 3:
        return None, "Error parsing response", "Unknown"
    doc_id_line = lines[0]
    justification_line = lines[1]
    relevance_line = lines[2]
    try:
        doc_id = doc_id_line.split(":")[1].strip()
        justification = justification_line.split(":")[1].strip()
        relevance_part = relevance_line.split(":")[1].strip()
        if relevance_part.startswith("[") and relevance_part.endswith("]"):
            relevance = relevance_part[1:-1]
        else:
            relevance = relevance_part
        return doc_id, justification, relevance
    except IndexError:
        return None, "Error parsing response", "Unknown"

def evaluate_relevance_azure(training_batches, corpus):
    """Evaluate the relevance of documents for each query using Azure OpenAI."""
    results = []
    total_queries = len(training_batches)
    for index, batch in enumerate(training_batches, start=1):
        logger.info(f"Processing query {index}/{total_queries}")
        query_id = batch["query_id"]
        query_text = batch["query_text"]
        positive_id = batch["positive_id"]
        doc_ids = [positive_id] + batch["hard_negative_ids"]  # Include positive ID with hard negatives
        batch_result = {
            "Query ID": query_id,
            "Query Text": query_text,
            "Positive ID": positive_id,
            "Positive Content": corpus.get(positive_id, "Content not found"),
            "Documents for Evaluation": []
        }
        for doc_id in doc_ids:
            prompt = create_relevance_prompt(query_id, query_text, doc_id, corpus)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error getting response for doc_id {doc_id}: {str(e)}")
                continue
            doc_id_parsed, justification, relevance = parse_response(response_text)
            max_attempts = 3
            attempt = 1
            while ((doc_id_parsed, justification, relevance) == (None, "Error parsing response", "Unknown") and attempt < max_attempts):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.choices[0].message.content
                doc_id_parsed, justification, relevance = parse_response(response_text)
                attempt += 1

            if doc_id_parsed != doc_id:
                logger.warning(f"Document ID mismatch: expected {doc_id}, got {doc_id_parsed}")
                doc_id_parsed = doc_id
            batch_result["Documents for Evaluation"].append({
                "Query ID": query_id,
                "Document ID": doc_id,
                "Content": corpus[doc_id],
                "Justification": justification,
                "Relevance": relevance
            })
        results.append(batch_result)
    return results

def main():
    """Main function to run the relevance evaluation and save results."""
    # Load training batches
    hard_negatives_data = "./swim_ir_yo_training_batches.npy"
    logger.info("Loading HN data")
    training_batches = load_npy_file(hard_negatives_data)

    # Load and preprocess dataset
    logger.info("Loading dataset")
    dataset_name = "nthakur/swim-ir-monolingual"
    dataset = load_dataset(dataset_name, 'yo')
    _, _, corpus = preprocess_dataset(dataset)

    # Evaluate relevance
    results = evaluate_relevance_azure(training_batches, corpus)

    # Save results to JSON
    output_file = "judgments_pro.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved relevance judgments to {output_file}")

if __name__ == "__main__":
    main()