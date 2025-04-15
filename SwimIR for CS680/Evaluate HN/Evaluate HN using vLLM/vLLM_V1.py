import numpy as np
from vllm import LLM, SamplingParams
import logging
import json
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize VLLM with Qwen 2.5
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.85,
    trust_remote_code=True,   
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

def load_npy_file(file_path):
    """Load the .npy file containing training batches"""
    try:
        training_batches = np.load(file_path, allow_pickle=True)
        logger.info(f"Loaded {len(training_batches)} batches from {file_path}")
        return training_batches
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def create_relevance_prompt(query_id ,query_text, positive_id, positive_content, doc_id, corpus):
    """Create a prompt for VLLM to determine relevance"""
    doc_content = corpus.get(doc_id, "Content not found")

    prompt = (
        f"Query Information\n"
        f"Query ID: '{query_id}'\n"
        f"Query Text: '{query_text}'\n"
        f"Positive ID and Content: '{positive_id}', '{positive_content}'\n"
        f"Doc ID: '{doc_id}'\n"
        f"Doc Content: '{doc_content}'\n"
        f"Task\n"
        f"Determine whether this document is relevant or irrelevant to the given query.\n"
        f"Provide:\n"
        f"Document ID: [Insert Document ID]\n"
        f"Justification: a brief reason for the decision\n"
        f"Relevance: [Relevant / Irrelevant]"
    )

    return prompt

def evaluate_relevance_vllm(training_batches,corpus):
    """Use VLLM to evaluate relevance for all documents in batches"""
    results = []
    
    for batch in training_batches:
        query_id = batch["query_id"]
        query_text = batch["query_text"]
        positive_id = batch["positive_id"]
        positive_content = corpus.get(positive_id, "Content not found")
        doc_ids = batch["hard_negative_ids"]
        
        all_doc_ids = [positive_id] + doc_ids
        # Prepare prompts for all documents in this batch
        prompts = []
        doc_info = []
        for doc_id in doc_ids:
            prompt = create_relevance_prompt(query_id, query_text, positive_id, positive_content, doc_id, corpus)
            prompts.append(prompt)
            doc_info.append({"Document ID": doc_id, "Content": corpus[doc_id]})
        
        # Batch process with VLLM
        logger.info(f"Evaluating relevance for Query ID: {query_id}")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs
        batch_result = {
            "Query ID": query_id,
            "Query Text": query_text,
            "Documents for Evaluation": []
        }
        
        for i, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            # Parse response (assuming format: [Relevance] Justification)
            try:
                relevance = "[Relevant]" if "[Relevant]" in response else "[Irrelevant]"
                justification = response.split("]")[1].strip() if "]" in response else "No justification provided"
            except:
                relevance = "[Irrelevant]"
                justification = "Error parsing model output"
            
            batch_result["Documents for Evaluation"].append({
                "Query ID": query_id,
                "Document ID": doc_info[i]["Document ID"],
                "Content": doc_info[i]["Content"],
                "Justification": justification,
                "Relevance": relevance.strip("[]")
            })
        
        results.append(batch_result)
    
    return results

# Preprocessing function
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
    hard_negatives_data  = "./swim_ir_yo_training_batches.npy"  # Adjust path as needed
    training_batches = load_npy_file(hard_negatives_data )
    

    if training_batches is None:
        return
    dataset_name = "nthakur/swim-ir-monolingual"
    dataset = load_dataset(dataset_name,'yo', trust_remote_code=True)
    _, _, corpus = preprocess_dataset(dataset)
    
    # Evaluate relevance using VLLM
    results = evaluate_relevance_vllm(training_batches,corpus)
    
    # Save results
    output_file = "judgments.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved relevance judgments to {output_file}")

if __name__ == "__main__":
    main()