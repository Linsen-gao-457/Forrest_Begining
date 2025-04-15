import json

def main():
    # Load the input JSON file
    with open('combined_queries.json', 'r') as infile:
        data = json.load(infile)

    formatted_data = []
    for entry in data:
        positive = []
        negative = []
        
        # Process evaluation documents
        for doc in entry.get("Documents for Evaluation", []):
            doc_id = doc.get("Document ID", "")
            relevance = doc.get("Relevance", "").strip().lower()
            
            if relevance == "relevant":
                positive.append(doc_id)
            elif relevance == "irrelevant":
                negative.append(doc_id)
        
        # Build formatted entry
        formatted_entry = {
            "query_id": entry["Query ID"],
            "query_text": entry["Query Text"],
            "positive_document_ids": positive,
            "negative_document_ids": negative
        }
        formatted_data.append(formatted_entry)

    # Save the formatted data
    with open('youruba_training_data_formated.json', 'w') as outfile:
        json.dump(formatted_data, outfile, indent=4, ensure_ascii=False)
    
    
if __name__ == "__main__":
    main()