export HF_HOME=/home/l78gao/scratch/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

source ~/miniconda3/bin/activate
conda activate Finetune
cd ~/scratch/Fine-tune/tevatron

export CKPT=/home/l78gao/scratch/Fine-tune/tevatron/MIRACLtrain-qwen25
export DATASET=yo
export BASE_MODEL_PATH=/home/l78gao/scratch/Fine-tune/tevatron/Qwen2.5-3B/models--Qwen--Qwen2.5-3B


mkdir -p miracl_embedding/${CKPT}/${DATASET}
PYTHONPATH=~/scratch/Fine-tune/tevatron:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode_mm \
  --output_dir=temp \
  --model_name_or_path ${BASE_MODEL_PATH} \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling eos \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --append_eos_token \
  --query_max_len 512 \
  --dataset_name miracl/miracl \
  --dataset_config $DATASET \
  --dataset_split dev \
  --encode_output_path miracl_embedding/${CKPT}/${DATASET}/queries.pkl \
  --encode_is_query \

for s in 0 1 2 3; do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode_mm \
  --output_dir=temp \
  --model_name_or_path ${BASE_MODEL_PATH} \
  --lora_name_or_path ${CKPT} \
  --lora \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --append_eos_token \
  --passage_max_len 512 \
  --dataset_name miracl/miracl-corpus \
  --dataset_config $DATASET \
  --dataset_split train \
  --encode_output_path miracl_embedding/${CKPT}/${DATASET}/corpus.${s}.pkl \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} &
done
wait

mkdir -p miracl_results/${CKPT}/${DATASET}
python -m tevatron.retriever.driver.search \
  --query_reps miracl_embedding/${CKPT}/${DATASET}/queries.pkl \
  --passage_reps miracl_embedding/${CKPT}/${DATASET}/'corpus.*.pkl' \
  --depth 100 \
  --batch_size 64 \
  --save_text \
  --save_ranking_to miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.txt

python -m tevatron.utils.format.convert_result_to_trec \
  --input miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.txt \
  --output miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.trec

python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 miracl-v1.0-${DATASET}-dev \
  miracl_results/${CKPT}/${DATASET}/rank.${DATASET}.trec