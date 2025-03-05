import numpy as np
import faiss
import pytrec_eval
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# [Evaluator class remains unchanged]
class Evaluator:
    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to not ignore them."
            )
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id].get("ndcg_cut_" + str(k), 0)
                _map[f"MAP@{k}"] += scores[query_id].get("map_cut_" + str(k), 0)
                recall[f"Recall@{k}"] += scores[query_id].get("recall_" + str(k), 0)
                precision[f"P@{k}"] += scores[query_id].get("P_" + str(k), 0)

        num_queries = len(scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / num_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / num_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / num_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / num_queries, 5)

        for metric in [ndcg, _map, recall, precision]:
            logger.info("\nEvaluation metrics:")
            for key in metric.keys():
                logger.info(f"{key}: {metric[key]:.4f}")

        return ndcg, _map, recall, precision


# --------------------------
# Retrieval and Evaluation
# --------------------------
def preprocess_dataset(dataset):
    data = dataset["dev"]

    # Create queries dictionary: {query_id: query_text}
    queries = {str(item["query_id"]): item["query"] for item in data}

    # Create qrels dictionary: {query_id: {passage_id: relevance}}
    qrels = {}

    for item in data:
        query_id = str(item["query_id"])
        qrels[query_id] = {}
        # Positive passages
        for passage in item["positive_passages"]:
            passage_id = str(passage["docid"])
            qrels[query_id][passage_id] = 1

        # Negative passages
        for passage in item["negative_passages"]:
            passage_id = str(passage["docid"])
            qrels[query_id][passage_id] = 0

    # Load and filter corpus from miracl/miracl-corpus for Yoruba, keeping only relevant documents
    corpus_dataset = load_dataset("miracl/miracl-corpus", "yo", trust_remote_code=True)[
        "train"
    ]
    corpus = {
        str(item["docid"]): item["title"] + item["text"]
        for item in corpus_dataset
        if str(item["docid"])
    }

    return queries, qrels, corpus


def build_faiss_index(corpus_embeddings, corpus_ids):
    """
    Index document embeddings using FAISS for efficient search, storing corpus_ids for later use.
    """
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)
    return index, corpus_ids  # Return both index and corpus_ids


def evaluate_retrieval(dataset, lang):
    print(f"\nEvaluating language: {lang}...")

    # Preprocess dataset
    queries, qrels, corpus_dict = preprocess_dataset(dataset)
    if not queries or not corpus_dict:
        print(f"No valid data found for {lang}")
        return 0, 0

    # Convert to lists for embedding while maintaining ID mappings
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    corpus_ids = list(corpus_dict.keys())
    corpus_texts = list(corpus_dict.values())

    # Generate embeddings
    query_embeddings = model.encode(query_texts, convert_to_numpy=True, device=device)
    corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True, device=device)

    # Build FAISS index
    index, retrieved_corpus_ids = build_faiss_index(corpus_embeddings, corpus_ids)
    # Search
    max_k = max(10, 100)
    distances, indices = index.search(query_embeddings, max_k)

    # Format results: {query_id: {passage_id: score}}
    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {
            retrieved_corpus_ids[idx]: float(distances[i][j])
            for j, idx in enumerate(indices[i][:max_k])
            if idx < len(retrieved_corpus_ids)
        }

    # Log results for debugging
    logger.info(f"Results for evaluation: {results}")

    # Evaluate
    k_values = [10, 100]
    ndcg, _map, recall, precision = Evaluator.evaluate(qrels, results, k_values)

    ndcg_at_10 = ndcg.get("NDCG@10", 0)
    recall_at_100 = recall.get("Recall@100", 0)
    print(f"\n{lang} - NDCG@10: {ndcg_at_10:.4f}, Recall@100: {recall_at_100:.4f}")

    return recall_at_100, ndcg_at_10


# --------------------------
# Main Code
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "nthakur/contriever-base-msmarco"
model = SentenceTransformer(model_name, device=device)
model.max_seq_length = 350
print("Maximum sequence length:", model.max_seq_length)  # Verify the setting

# Load MIRACL dataset (dev split only)
dataset_name = "miracl/miracl"
dataset = load_dataset(dataset_name, "yo", trust_remote_code=True)
lang = "Yoruba (yo)"

# Run evaluation
recall, ndcg = evaluate_retrieval(dataset, lang)
print(f"\nFinal Results:\nRecall@100: {recall:.4f}\nNDCG@10: {ndcg:.4f}")
