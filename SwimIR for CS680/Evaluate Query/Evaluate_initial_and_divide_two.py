import json

def analyze_relevance(file_path):
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
    else:
        print("No queries found in the file.")

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

# Run the analysis
if __name__ == "__main__":
    file_path = "Whole_HN.json"  # Change this to your JSON file path
    analyze_relevance(file_path)