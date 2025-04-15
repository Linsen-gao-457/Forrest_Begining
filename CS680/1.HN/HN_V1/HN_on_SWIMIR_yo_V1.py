import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import logging

# For BM25 hard negatives
from rank_bm25 import BM25Okapi


# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Preprocess dataset for hard negative mining
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

def bm25_retrieve(query_text, corpus_texts, corpus_ids, top_k=20):
    tokenize = lambda text: text.lower().split()
    tokenized_corpus = [tokenize(text) for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    results = [(corpus_ids[idx], scores[idx], idx) for idx in sorted_indices]
    logger.info("BM25 finished iteration")
    return results

def kalm_retrieve(query_embedding, corpus_embeddings, corpus_ids, top_k=20):
    
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_embeddings)
    distances, indices = index.search(query_embedding, top_k)
    
    results = [(corpus_ids[idx], score, idx) for score, idx in zip(distances[0], indices[0])]
    logger.info("KaLM finished iteration")
    return results

from transformers import AutoTokenizer

def mine_hard_negatives(model, queries, positives, corpus_dict, bm25_top_k=30, kalm_top_k=30, negatives_to_mine=15):
    corpus_ids = list(corpus_dict.keys())
    corpus_texts = list(corpus_dict.values())
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    

    logger.info("Encoding corpus embeddings")
    corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True, batch_size=32)
    
    logger.info("Encoding query embeddings")
    query_embeddings = []
    batch_size = 16
    for i in range(0, len(query_texts), batch_size):
        batch = query_texts[i:i + batch_size]
        logger.info(f"Encoding query batch {i//batch_size + 1} (size: {len(batch)})")
        tokenized = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to('cuda').long()
        attention_mask = tokenized['attention_mask'].to('cuda')
        with torch.no_grad():
            outputs = model({'input_ids': input_ids, 'attention_mask': attention_mask})
            batch_embeddings = outputs['sentence_embedding'].cpu().numpy()
        query_embeddings.append(batch_embeddings)
        torch.cuda.empty_cache()
    query_embeddings = np.vstack(query_embeddings)
    logger.info("Finish Embedding")
    query_emb_dict = {qid: emb for qid, emb in zip(query_ids, query_embeddings)}
    
    training_batches = []
    for qid in query_ids:
        gold_docid = positives.get(qid)
        if gold_docid not in corpus_ids:
            continue
        
        logger.info("BM25 retrieval")
        bm25_results = bm25_retrieve(queries[qid], corpus_texts, corpus_ids, top_k=bm25_top_k)
        bm25_results = [item for item in bm25_results if item[0] != gold_docid]
        
        logger.info("KaLM retrieval")
        query_embedding = query_emb_dict[qid].reshape(1, -1)
        kalm_results = kalm_retrieve(query_embedding, corpus_embeddings, corpus_ids, top_k=kalm_top_k)
        kalm_results = [item for item in kalm_results if item[0] != gold_docid]
        logger.info('Finished retrieval')
        
        neg_dict = {}
        for source in [bm25_results, kalm_results]:
            for docid, score, idx in source:
                neg_dict[docid] = max(neg_dict.get(docid, float('-inf')), score)
        
        sorted_negatives = sorted(neg_dict.items(), key=lambda x: x[1], reverse=True)[:negatives_to_mine]
        hard_negative_ids = [docid for docid, _ in sorted_negatives]
        hard_negative_scores = [neg_dict[docid] for docid in hard_negative_ids]

        batch = {
            "query_id": qid,
            "query_text": queries[qid],
            "positive_id": gold_docid,
            "hard_negative_ids": hard_negative_ids,
            "hard_negative_scores": hard_negative_scores
        }
        training_batches.append(batch)
    
    return training_batches

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    logger.info("Loading model")
    model_name = "HIT-TMG/KaLM-embedding-multilingual-mini-v1"
    model = SentenceTransformer(model_name, device=device)

    logger.info("Loading dataset")
    dataset_name = "nthakur/swim-ir-monolingual"
    dataset = load_dataset(dataset_name,'yo', trust_remote_code=True)

    
    logger.info("Preprocessing dataset")
    queries, positives, corpus_dict = preprocess_dataset(dataset)
    
    logger.info("Starting hard negative mining")
    training_batches = mine_hard_negatives(model, queries, positives, corpus_dict, 
                                        bm25_top_k=30, kalm_top_k=30, negatives_to_mine=15)
    
    print(f"\nMined {len(training_batches)} training batches.")
    for batch in training_batches[:2]:
        print(f"\nQuery ID: {batch['query_id']}")
        print(f"Query Text: {batch['query_text']}")
        print(f"Positive ID: {batch['positive_id']}")
        print(f"Hard Negatives Count: {len(batch['hard_negative_ids'])}")
        print(f"Hard Negative IDs: {batch['hard_negative_ids']}")
        print(f"Hard Negative Scores: {batch['hard_negative_scores']}")
    
    output_file = "swim_ir_yo_training_batches.npy"
    np.save(output_file, training_batches, allow_pickle=True)
    print(f"\nSaved training batches to '{output_file}'.")

if __name__ == "__main__":
    main()