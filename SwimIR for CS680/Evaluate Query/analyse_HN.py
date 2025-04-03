import json
import logging
from datasets import load_dataset
logging.basicConfig(level=logging.INFO,force=True, format='%(asctime)s - %(levelname)s - %(message)s')
def analyze_relevance(file_path):
    """
    Analyze a JSON file with 5,000 queries, each with 21 documents, to count 'Relevant' documents per query,
    calculate the average number of relevant documents per query, and count queries with exactly one relevant document.

    Args:
        file_path (str): Path to the JSON file containing relevance judgments.
    """
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        return

    # Initialize variables
    total_relevant = 0
    num_queries = len(data)
    queries_with_one_relevant = 0  # Counter for queries with exactly one relevant document

    # Process each query
    for query in data:
        query_id = query.get("Query ID", "Unknown")
        documents = query.get("Documents for Evaluation", [])


        # Count relevant documents for this query
        relevant_count = sum(1 for doc in documents if doc.get("Relevance") == "Relevant")


        # Add to the running total
        total_relevant += relevant_count

        # Increment counter if this query has exactly one relevant document
        if relevant_count == 1:
            queries_with_one_relevant += 1

        print(f"Query ID: {query_id}, Relevant documents: {relevant_count}")
    # Calculate and print the average
    if num_queries > 0:
        average = total_relevant / num_queries
        print(f"Average number of relevant documents per query: {average:.2f}")
    else:
        print("No queries found in the file.")

    # Print the number of queries with only one relevant document
    print(f"Number of queries with only one relevant document: {queries_with_one_relevant}")

def divive_into_one_relevant_query_and_others(file_path):
    """
    Analyze a JSON file with queries and their documents to:
    - Count 'Relevant' documents per query.
    - Calculate the average number of relevant documents per query.
    - Count queries with exactly one relevant document.
    - Filter queries into two files: one for queries with exactly one relevant document,
      and another for queries with any other number of relevant documents.

    Args:
        file_path (str): Path to the JSON file containing relevance judgments.
    """
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        return

    # Initialize variables
    total_relevant = 0
    num_queries = len(data)
    queries_with_one_relevant = 0  # Counter for queries with exactly one relevant document
    one_relevant_queries = []      # List for queries with exactly one relevant document
    other_queries = []             # List for queries with any other number of relevant documents

    # Process each query
    for query in data:
        query_id = query.get("Query ID", "Unknown")
        documents = query.get("Documents for Evaluation", [])

        # Count relevant documents for this query
        relevant_count = sum(1 for doc in documents if doc.get("Relevance") == "Relevant")

        # Print the count for this query
        print(f"Query ID: {query_id}, Relevant documents: {relevant_count}")

        # Add to the running total
        total_relevant += relevant_count

        # Filter the query into the appropriate list and increment counter if applicable
        if relevant_count == 1:
            queries_with_one_relevant += 1
            one_relevant_queries.append(query)
        else:
            other_queries.append(query)

    # Calculate and print the average
    if num_queries > 0:
        average = total_relevant / num_queries
        print(f"Average number of relevant documents per query: {average:.2f}")


    # Print the number of queries with only one relevant document
    print(f"Number of queries with only one relevant document: {queries_with_one_relevant}")

    # Write the filtered queries to separate JSON files
    with open("queries_with_one_relevant.json", 'w', encoding='utf-8') as f:
        json.dump(one_relevant_queries, f, indent=4)

    with open("queries_with_other_relevant.json", 'w', encoding='utf-8') as f:
        json.dump(other_queries, f, indent=4)

    # Optional: Confirm the files have been written
    print(f"Written {len(one_relevant_queries)} queries to queries_with_one_relevant.json")
    print(f"Written {len(other_queries)} queries to queries_with_other_relevant.json")

def combine_json_files(file1_path, file2_path, output_path):
    """
    Combine two JSON files into one.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.
        output_path (str): Path to save the combined JSON file.
    """
        # Load data from both files
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    # Combine the data
    combined_data = data1 + data2

    # Save the combined data
    with open(output_path, 'w', encoding='utf-8') as out_file:
        json.dump(combined_data, out_file, indent=4)

    print(f"Successfully combined files! Total queries: {len(combined_data)}")
    print(f"Combined file saved to: {output_path}")

def process_and_save_json(input_file, output_file):
    """Load, transform, and save the JSON data."""
    logging.info("Loading judgments data")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            judgments_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        return
    
    logging.info("Transforming data format")
    formatted_data = []
    
    for entry in judgments_data:
        query_id = entry["Query ID"]
        query_text = entry["Query Text"]
        
        positive_passages = []
        negative_passages = []
        
        for doc in entry["Documents for Evaluation"]:
            doc_id = doc["Document ID"]
            title, text = "", doc["Content"]  # Assuming no separate title field in input
            
            if doc["Relevance"].lower() == "relevant":
                positive_passages.append({"docid": doc_id, "text": text,"title": title})
            else:
                negative_passages.append({"docid": doc_id, "text": text,"title": title})
        
        formatted_data.append({
            "query_id": query_id,
            "query": query_text,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages
        })
    
    logging.info("Saving transformed data")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved transformed data to {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")
    
    
def check_empty_positive_passages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    empty_cases = [entry["query_id"] for entry in data if not entry["positive_passages"]]
    
    print(f"Total queries with no positive passages: {len(empty_cases)}")
    if empty_cases:
        print("Query IDs with no positive passages:")
        for qid in empty_cases:
            print(qid)

def check_weak_training_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    weak_cases = [
        entry["query_id"]
        for entry in data
        if not entry["positive_passages"] or len(entry["negative_passages"]) < 8
    ]
    
    print(f"Total weak training pairs ( <8 negatives): {len(weak_cases)}")
    if weak_cases:
        print("Query IDs of weak training pairs:")
        for qid in weak_cases:
            print(qid)
            
def process_and_save_json(input_file,
                          output_filtered_file, 
                          output_no_positive_file, 
                          output_few_negative_file, 
                          lang='yo'):
    """Load, transform, and save JSON into three files: filtered, no-positive, few-negatives."""
    logging.info("Loading judgments data")
    
    try:
        dataset = load_dataset("nthakur/swim-ir-monolingual", lang, trust_remote_code=True)
    except Exception as e:
        logging.error(f"Error loading HuggingFace dataset: {e}")
        return

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            judgments_data = json.load(f)
    except Exception as e:
        logging.error(f"Error loading input JSON file: {e}")
        return

    # Build document lookup
    doc_lookup = {
        doc['_id']: {'title': doc['text'], 'text': doc['title']}
        for doc in dataset['train']
    }

    logging.info("Transforming and filtering data")
    filtered = []
    no_positives = []
    few_negatives = []

    for entry in judgments_data:
        query_id = entry["Query ID"]
        query_text = entry["Query Text"]

        positive_passages = []
        negative_passages = []

        for doc in entry["Documents for Evaluation"]:
            doc_id = doc["Document ID"]
            metadata = doc_lookup.get(doc_id, {"title": "", "text": ""})
            title, text = metadata["title"], metadata["text"]

            passage = {"docid": doc_id, "text": text, "title": title}
            if doc["Relevance"].lower() == "relevant":
                positive_passages.append(passage)
            else:
                negative_passages.append(passage)

        processed_entry = {
            "query_id": query_id,
            "query": query_text,
            "positive_passages": positive_passages,
            "negative_passages": negative_passages,
            "language": lang
        }

        if not positive_passages:
            no_positives.append(processed_entry)
        elif len(negative_passages) < 8:
            few_negatives.append(processed_entry)
        else:
            filtered.append(processed_entry)

    # Helper to save
    def save_json(data, path, description):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"{description} saved to {path} ({len(data)} entries)")
        except Exception as e:
            logging.error(f"Error saving {description}: {e}")

    # Save all
    save_json(filtered, output_filtered_file, "Filtered queries")
    save_json(no_positives, output_no_positive_file, "No-positive queries")
    save_json(few_negatives, output_few_negative_file, "Few-negative queries")
        

# Run the analysis
if __name__ == "__main__":
    
    
    process_and_save_json(
    input_file="./Yo_raw.json",
    output_filtered_file="Yo.json",
    output_no_positive_file="queries_no_positives_Yo.json",
    output_few_negative_file="queries_few_negatives_Yo.json",
    lang="yo"
)
    file_path = "./Yo.json"  # Change this to your JSON file path
    check_empty_positive_passages(file_path)
    check_weak_training_pairs(file_path)
    # analyze_relevance(file_path)
    # combine_json_files(
    #     file1_path="queries_with_one_relevant.json",
    #     file2_path="reevaluated_queries.json",
    #     output_path="combined_queries.json"
    # )
    # process_and_save_json(input_file="./combined_queries.json", output_file="formatted_judgments.json")
    