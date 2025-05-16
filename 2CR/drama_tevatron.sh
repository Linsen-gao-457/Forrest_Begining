export CKPT=facebook-drama-base
export MODEL=facebook/drama-base
export DATASET=yo

mkdir -p miracl_embedding/${CKPT}/${DATASET}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${MODEL} \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --query_prefix "Query: " \
  --query_max_len 512 \
  --encode_is_query \
  --dataset_name miracl/miracl \
  --dataset_config ${DATASET} \
  --dataset_split dev \
  --encode_output_path miracl_embedding/${CKPT}/${DATASET}/queries.pkl

for s in 0 1 2 3;
do
CUDA_VISIBLE_DEVICES=$s python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path ${MODEL} \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --passage_prefix "" \
  --passage_max_len 512 \
  --dataset_name miracl/miracl-corpus \
  --dataset_config ${DATASET} \
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
