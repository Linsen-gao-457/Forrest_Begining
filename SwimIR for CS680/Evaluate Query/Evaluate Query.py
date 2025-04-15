import logging
import json
from openai import AzureOpenAI

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Initialize Azure OpenAI client
client = AzureOpenAI()

def create_relevance_prompt(query_id: str, query_text: str, doc_id: str, doc_content: str) -> str:
    """Create a prompt for the OpenAI model to assess document relevance."""
    prompt = (
        f"Query ID: '{query_id}'\n"
        f"Query Text: '{query_text}'\n"
        f"Doc ID: '{doc_id}'\n"
        f"Doc Content: '{doc_content}'\n"
        f"Task\n"
        f"Determine whether this document is relevant or irrelevant to the given query.\n"
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

def parse_response(response_text: str) -> tuple:
    """Parse the model's response to extract document ID, justification, and relevance."""
    lines = response_text.split("\n")
    if len(lines) < 3:
        return None, "Error parsing response", "Unknown"
    try:
        doc_id = lines[0].split(":", 1)[1].strip()
        justification = lines[1].split(":", 1)[1].strip()
        relevance_part = lines[2].split(":", 1)[1].strip()
        relevance = relevance_part[1:-1] if relevance_part.startswith("[") and relevance_part.endswith("]") else relevance_part
        return doc_id, justification, relevance
    except IndexError:
        return None, "Error parsing response", "Unknown"

def reevaluate_queries(input_file: str, output_file: str):
    """Reevaluate the relevance of documents for queries in the input JSON file."""
    # Load the input JSON file
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} queries from {input_file}")
    except Exception as e:
        logger.error(f"Error loading {input_file}: {str(e)}")
        return

    reevaluated_results = []
    total_queries = len(queries)
    # Process each query
    for index, query in enumerate(queries, start=1):
        logger.info(f"Processing query {index}/{total_queries}: {query['Query ID']}")
        query_id = query["Query ID"]
        query_text = query["Query Text"]
        positive_id = query["Positive ID"]
        positive_content = query["Positive Content"]
        documents = query["Documents for Evaluation"]
        reevaluated_docs = []

        # Process each document for the query
        for doc in documents:
            doc_id = doc["Document ID"]
            doc_content = doc["Content"]
            prompt = create_relevance_prompt(query_id, query_text, doc_id, doc_content)
            max_attempts = 3

            # Retry logic for API calls
            for attempt in range(1, max_attempts + 1):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = response.choices[0].message.content
                    doc_id_parsed, justification, relevance = parse_response(response_text)
                    if doc_id_parsed is not None:
                        break
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed for doc_id {doc_id}: {str(e)}")
            else:
                logger.warning(f"Failed to evaluate doc_id {doc_id} after {max_attempts} attempts")
                doc_id_parsed = doc_id
                justification = "Failed to parse response after multiple attempts"
                relevance = "Unknown"

            # Check for document ID mismatch
            if doc_id_parsed != doc_id:
                logger.warning(f"Document ID mismatch: expected {doc_id}, got {doc_id_parsed}")
                doc_id_parsed = doc_id

            reevaluated_docs.append({
                "Query ID": query_id,
                "Document ID": doc_id,
                "Content": doc_content,
                "Justification": justification,
                "Relevance": relevance
            })

        # Append the query with its reevaluated documents
        reevaluated_results.append({
            "Query ID": query_id,
            "Query Text": query_text,
            "Positive ID": positive_id,
            "Positive Content": positive_content,
            "Documents for Evaluation": reevaluated_docs
        })

    # Save the results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reevaluated_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved reevaluated results to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to {output_file}: {str(e)}")

def main():
    """Main function to run the reevaluation process."""
    input_file = "queries_with_other_relevant.json"
    output_file = "reevaluated_queries.json"
    reevaluate_queries(input_file, output_file)

if __name__ == "__main__":
    main()