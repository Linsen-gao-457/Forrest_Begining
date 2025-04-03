#!/bin/bash
#SBATCH --job-name=qwen25_retriever_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=5-0:00:00

#SBATCH --mail-user=l78gao@uwaterloo.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=%N-%j.out
#SBATCH --output=finetune
#SBATCH --output=finetune.log
#SBATCH --error=finetune_errors.log

export NCCL_BLOCKING_WAIT=1 # Set this variable to use the NCCL backend

export SLURM_ACCOUNT=def-jimmylin
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT
export CUDA_HOME=$HOME/scratch/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/scratch/l78gao/hf_cache

source ~/miniconda3/bin/activate base
conda activate /home/l78gao/conda_envs/finetune
source ~/miniconda3/bin/activate base
conda activate finetune
cd ~/scratch/SwimIR/Fine-tune/Yo/tevatron

conda install -c nvidia -c pytorch -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12.4 -y
python -c "import torch; print(torch.**version**, torch.version.cuda); print(torch.cuda.is_available())"
pip install --no-build-isolation --no-cache-dir flash-attn --no-deps
pip show flash-attn

pip install --no-cache-dir transformers datasets peft
pip install --no-cache-dir deepspeed accelerate
pip install --no-cache-dir faiss-cpu
pip install --no-cache-dir -e .
pip uninstall pillow
pip install --no-cache-dir pillow
pip install --no-cache-dir qwen-vl-utils

# Run the command using interactive environment

deepspeed --include localhost:0,1 --master_port 60000 --module tevatron.retriever.driver.train \
 --deepspeed deepspeed/ds_zero0_config.json \
 --output_dir retriever-qwen25 \
 --model_name_or_path Qwen/Qwen2.5-3B \
 --lora \
 --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
 --save_steps 50 \
 --dataset_path /home/l78gao/scratch/SwimIR/Fine-tune/Yo/Yo.json \
 --query_prefix "Query: " \
 --passage_prefix "Passage: " \
 --bf16 \
 --pooling eos \
 --append_eos_token \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 8 \
 --gradient_checkpointing \
 --train_group_size 21 \
 --learning_rate 1e-5 \
 --query_max_len 150 \
 --passage_max_len 350 \
 --num_train_epochs 1 \
 --logging_steps 10 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 4

# Run the command on 4 GPUs

deepspeed --include localhost:0,1,2,3 --master_port 60000 --module tevatron.retriever.driver.train \
 --deepspeed deepspeed/ds_zero0_config.json \
 --output_dir retriever-qwen25 \
 --model_name_or_path Qwen/Qwen2.5-3B \
 --lora \
 --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
 --save_steps 50 \
 --dataset_path /home/l78gao/scratch/SwimIR/Fine-tune/Yo/Yo.json \
 --query_prefix "Query: " \
 --passage_prefix "Passage: " \
 --bf16 \
 --pooling eos \
 --append_eos_token \
 --normalize \
 --temperature 0.01 \
 --per_device_train_batch_size 8 \
 --gradient_checkpointing \
 --train_group_size 21 \
 --learning_rate 1e-5 \
 --query_max_len 150 \
 --passage_max_len 350 \
 --num_train_epochs 1 \
 --logging_steps 10 \
 --overwrite_output_dir \
 --gradient_accumulation_steps 4
