import json
from datasets import load_dataset

def preprocess_dataset(dataset):
    """
    Process the MIRACL dev split to create queries, positives, and corpus.

    Returns:
        queries (dict): {query_id: query_text}
        positives (dict): {query_id: positive_docid}
        corpus (dict): {passage_id: passage_text}
    """
    data = dataset["train"]

    # Create queries dictionary: {query_id: query_text}
    queries = {item["_id"]: item["query"] for item in data}

    # Create positives dictionary: {query_id: positive_docid}
    positives = {item["_id"]: item["_id"] for item in data}

    # Create corpus dictionary: {passage_id: passage_text}
    corpus = {item["_id"]: item["title"]+item["text"] for item in data}

    return queries, positives, corpus

def main():
    dataset_name = "nthakur/swim-ir-monolingual"
    dataset = load_dataset(dataset_name,'yo', trust_remote_code=True)
    _, _, corpus = preprocess_dataset(dataset)
    corpus_data = [{"docid": doc_id, "document_text": doc_text} 
                for doc_id, doc_text in corpus.items()]

    with open('Train_data_formated.json', 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    main()