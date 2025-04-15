import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
import logging
from transformers import AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
import json
import os
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
from pyserini.search.lucene import LuceneSearcher
import random

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Preprocess dataset for hard negative mining
def preprocess_dataset(dataset, sample_size=20000):
    """
    Process the MIRACL dev split to create queries, positives, and corpus.
    
    Returns:
        queries (dict): {query_id: query_text}
        positives (dict): {query_id: positive_docid}
        corpus (dict): {passage_id: passage_text}
    """
    data = dataset["train"]
    
    # Create queries dictionary: {query_id: query_text}
    all_queries = {item["_id"]: item["query"] for item in data}
    
    # Create positives dictionary: {query_id: positive_docid}
    all_positives  = {item["_id"]: item["_id"] for item in data}
    
    # Create corpus dictionary: {passage_id: passage_text}
    all_corpus  = {item["_id"]: item["title"]+item["text"] for item in data}
    
    # If dataset contains fewer than sample_size, use all available queries
    if len(all_queries) < sample_size:
        sampled_query_ids = list(all_queries.keys())  # Use all queries
    else:
        sampled_query_ids = random.sample(list(all_queries.keys()), sample_size)
    
    queries = {qid: all_queries[qid] for qid in sampled_query_ids}
    positives = {qid: all_positives[qid] for qid in sampled_query_ids}
    corpus = {qid: all_corpus[qid] for qid in sampled_query_ids}
    
    return queries, positives, corpus


def create_jsonl_corpus(corpus_dict, output_path="./Jsonl_Corpus/corpus.jsonl"):
    """Convert corpus dictionary to JSONL format for Pyserini"""
    output_dir = os.path.dirname(output_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, text in corpus_dict.items():
            doc = {
                "id": doc_id,
                "contents": text
            }
            f.write(json.dumps(doc) + '\n')
    
    return output_path

def build_index(jsonl_path, index_path="indexes/swim_ir_index"):
    """Build Lucene index using Pyserini"""
    if not os.path.exists(index_path):
        # Create the directory containing the JSONL file
        input_dir = os.path.dirname(jsonl_path)

        cmd = (
            f"python -m pyserini.index.lucene "
            f"--collection JsonCollection "
            f"--input {input_dir} "
            f"--index {index_path} "
            f"--generator DefaultLuceneDocumentGenerator "
            f"--threads 4 "
            f"--storePositions --storeRaw"
        )
        
        # Log and run the command
        logger.info(f"Building index with command: {cmd}")
        os.system(cmd)
    
    return index_path



def bm25_retrieve(query_text, corpus_dict, top_k=20):
        logger.info("Creating JSONL corpus file")
        jsonl_path = create_jsonl_corpus(corpus_dict)
        logger.info("Building Lucene index")
        index_path = build_index(jsonl_path)
        searcher = LuceneSearcher('indexes/swim_ir_index')
        hits = searcher.search(query_text, k=20)
        corpus_ids = list(corpus_dict.keys())
        scores = np.array([hit.score for hit in hits])
        indices = np.array([corpus_ids.index(hit.docid) for hit in hits])
        sorted_indices = np.argsort(scores)[::-1]
        results = [(corpus_ids[idx], scores[i], idx) 
                  for i, idx in enumerate(sorted_indices)]
        logger.info(f"BM25 retrieved {len(results)} results for query: {query_text}")
        return results


def kalm_retrieve(query_embedding, corpus_embeddings, corpus_ids, top_k=20, batch_size = 5000):
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    for i in range(0, len(corpus_embeddings), batch_size):
        index.add(corpus_embeddings[i:i + batch_size])
    distances, indices = index.search(query_embedding, top_k)
    
    results = [(corpus_ids[idx], score, idx) for score, idx in zip(distances[0], indices[0])]
    logger.info("KaLM finished iteration")
    return results

def normalize_scores(results):
    """normalize values to [0,1]"""
    if not results:  # Check if results is empty
        return np.array([])
    scaler = MinMaxScaler()
    scores = np.array([score for _, score, _ in results]).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(scores).flatten()
    return normalized_scores

def mine_hard_negatives(model, queries, positives, corpus_dict, bm25_top_k=30, kalm_top_k=30, negatives_to_mine=20):
    corpus_ids = list(corpus_dict.keys())
    corpus_texts = list(corpus_dict.values())
    query_ids = list(queries.keys())
    query_texts = list(queries.values())
    

    logger.info("Encoding corpus embeddings")
    corpus_embeddings = model.encode(corpus_texts, convert_to_numpy=True, batch_size=32)
    
    logger.info("Encoding query embeddings")
    query_embeddings = []
    batch_size = 16
    tokenizer = AutoTokenizer.from_pretrained("HIT-TMG/KaLM-embedding-multilingual-mini-v1")
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
        bm25_results = bm25_retrieve(queries[qid], corpus_dict, top_k=bm25_top_k)
        bm25_results = [item for item in bm25_results if item[0] != gold_docid]
        
        logger.info("KaLM retrieval")
        query_embedding = query_emb_dict[qid].reshape(1, -1)
        kalm_results = kalm_retrieve(query_embedding, corpus_embeddings, corpus_ids, top_k=kalm_top_k)
        kalm_results = [item for item in kalm_results if item[0] != gold_docid]
        logger.info('Finished retrieval')
        
        #normalize the scores of Kalm and BM25
        bm25_norm = normalize_scores(bm25_results)
        kalm_norm = normalize_scores(kalm_results)
        
        neg_dict = {}
        for i, (docid, _, _) in enumerate(bm25_results):
            if i < len(bm25_norm):
                neg_dict[docid] = neg_dict.get(docid, 0) + bm25_norm[i]
        for i, (docid, _, _) in enumerate(kalm_results):
            if i < len(kalm_norm):
                neg_dict[docid] = neg_dict.get(docid, 0) + kalm_norm[i]

        
        #Sort scores
        sorted_negatives = sorted(neg_dict.items(), key=lambda HN: HN[1], reverse=True)[:negatives_to_mine]
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
    dataset = load_dataset(dataset_name,'en', trust_remote_code=True)

    
    logger.info("Preprocessing dataset")
    queries, positives, corpus_dict = preprocess_dataset(dataset)
    
    logger.info("Starting hard negative mining")
    training_batches = mine_hard_negatives(model, queries, positives, corpus_dict, 
                                        bm25_top_k=30, kalm_top_k=30, negatives_to_mine=20)
    
    print(f"\nMined {len(training_batches)} training batches.")
    for batch in training_batches[:2]:
        print(f"\nQuery ID: {batch['query_id']}")
        print(f"Query Text: {batch['query_text']}")
        print(f"Positive ID: {batch['positive_id']}")
        print(f"Hard Negatives Count: {len(batch['hard_negative_ids'])}")
        print(f"Hard Negative IDs: {batch['hard_negative_ids']}")
        print(f"Hard Negative Scores: {batch['hard_negative_scores']}")
    
    output_file = "/home/l78gao/scratch/SwimIR/HN/swim_ir_en_training_batches.npy"
    np.save(output_file, training_batches, allow_pickle=True)
    print(f"\nSaved training batches to '{output_file}'.")

if __name__ == "__main__":
    main()