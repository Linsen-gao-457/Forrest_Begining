import numpy as np
import faiss
import pytrec_eval
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
import logging
from peft import PeftModel
from tqdm import tqdm 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def encode_texts(texts, tokenizer, model, device, max_length=350, batch_size=32):
    all_embeddings = []
    model.eval()
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", total=num_batches):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            pooled = F.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def preprocess_dataset(dataset, split_name="dev"):
    data = dataset[split_name]
    queries = {str(item["query_id"]): item["query"] for item in data}
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

    corpus_dataset = load_from_disk("/home/l78gao/scratch/Fine-tune/miracl_corpus_yo")
    corpus = {str(item["docid"]): item["text"] for item in corpus_dataset if str(item["docid"])}
    return queries, qrels, corpus


def build_faiss_index(corpus_embeddings, corpus_ids):
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)
    return index, corpus_ids


def evaluate_retrieval(dataset, lang, tokenizer, model, device):
    print(f"\nEvaluating language: {lang}...")

    queries, qrels, corpus_dict = preprocess_dataset(dataset, split_name="dev")
    if not queries or not corpus_dict:
        print(f"No valid data found for {lang}")
        return 0, 0

    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    corpus_ids = list(corpus_dict.keys())
    corpus_texts = list(corpus_dict.values())

    query_embeddings = encode_texts(query_texts, tokenizer, model, device, max_length=150)
    corpus_embeddings = encode_texts(corpus_texts, tokenizer, model, device, max_length=350)
    print("\n--- Embedding Stats ---")
    print("Query embeddings norm (avg):", np.linalg.norm(query_embeddings, axis=1).mean())
    print("Corpus embeddings norm (avg):", np.linalg.norm(corpus_embeddings, axis=1).mean())
    print("------------------------")
    index, retrieved_corpus_ids = build_faiss_index(corpus_embeddings, corpus_ids)

    max_k = max(10, 100)
    distances, indices = index.search(query_embeddings, max_k)

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {
            retrieved_corpus_ids[idx]: float(distances[i][j])
            for j, idx in enumerate(indices[i][:max_k])
            if idx < len(retrieved_corpus_ids)
        }

    logger.info(f"Results for evaluation: {results}")

    k_values = [10, 100]
    print("\n--- Sample Retrieval Check ---")
    sample_qid = query_ids[0]
    print("Sample Query ID:", sample_qid)
    print("Top-10 Retrieved DocIDs:", list(results[sample_qid].keys())[:10])
    print("Ground Truth Positives in qrels:", [docid for docid, rel in qrels[sample_qid].items() if rel == 1])
    print("------------------------------")

    ndcg, _map, recall, precision = Evaluator.evaluate(qrels, results, k_values)

    ndcg_at_10 = ndcg.get("NDCG@10", 0)
    recall_at_100 = recall.get("Recall@100", 0)
    print(f"\n{lang} - NDCG@10: {ndcg_at_10:.4f}, Recall@100: {recall_at_100:.4f}")

    return recall_at_100, ndcg_at_10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    adapter_path = "/home/l78gao/scratch/Fine-tune/tevatron/retriever-qwen25"
    base_model_path = "/home/l78gao/scratch/Fine-tune/tevatron/Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"

    base_model = AutoModel.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    print(model)
    model.max_seq_length = 350
    print("Maximum sequence length:", model.max_seq_length)

    dataset = load_from_disk("/home/l78gao/scratch/Fine-tune/miracl_yo_local")

    lang = "Yoruba (yo)"
    print("Available splits:", dataset.keys())
    recall, ndcg = evaluate_retrieval(dataset, lang, tokenizer, model, device)
    print(f"\nFinal Results:\nRecall@100: {recall:.4f}\nNDCG@10: {ndcg:.4f}")



if __name__ == "__main__":
    main()
