#!/bin/bash
#SBATCH --job-name=alpaca-reps     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=500G         # memory per cpu-core
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=scripts/slurm-%j-%x.out
#SBATCH --error=scripts/slurm-%j-%x.err
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --constraint=gpu80       # choose 80GB GPU

module purge
module load anaconda3/2022.5
conda activate rlhf

# Alpaca grads 
SAVE_NAME="llama7b-alpaca-reps-bs=20-lr=5e-5" 
for seed in 20 ; do 
    torchrun --nnodes 1 --nproc_per_node 2 --master_port 13579 finetuning.py \
    --batch_size_training 20 --lr 5e-5 \
    --gradient_accumulation_steps 1 --weight_decay 0 \
    --num_epochs 5 \
    --dataset alpaca_dataset \
    --data_path ft_datasets/alpaca_dataset/reps/alpaca_top100.json \
    --enable_fsdp \
    --model_name ckpts/Llama-2-7b-chat-fp16 --pure_bf16 \
    --dist_checkpoint_root_folder finetuned_models_seeds \
    --dist_checkpoint_folder $SAVE_NAME-seed${seed} \
    --run_validation False --save_every_epoch False ;\

   
    python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models_seeds/${SAVE_NAME}-seed${seed}-ckpts/Llama-2-7b-chat-fp16" -consolidated_model_path "finetuned_models_seeds/${SAVE_NAME}-seed${seed}/" -HF_model_path_or_name "ckpts/Llama-2-7b-chat-fp16"
    rm -r finetuned_models/${SAVE_NAME}-seed${seed}-ckpts/Llama-2-7b-chat-fp16

done

for seed in 20 ; do 
    mkdir -p question_output/alpaca_May/
    python -u safety_evaluation/question_inference.py \
    --model_name /scratch/gpfs/lh2046/LLMs-Finetuning-Safety/llama2/finetuned_models_seeds/${SAVE_NAME}-seed${seed}/ \
    --prompt_file safety_evaluation/data/harmful_behaviors.csv \
    --prompt_template_style alpaca \
    --top_p 0 \
    --seed ${seed} \
    --output_file /scratch/gpfs/lh2046/question_output/seeds/alpaca/bm25/harmful_behaviors_${SAVE_NAME}-seed${seed}.jsonl

done



# bm25 baseline
# SAVE_NAME="llama7b-bm25-bs=20-lr=5e-5" 
# for seed in 106 ; do 
#     torchrun --nnodes 1 --nproc_per_node 2 --master_port 13579 finetuning.py \
#     --batch_size_training 20 --lr 5e-5 \
#     --gradient_accumulation_steps 1 --weight_decay 0 \
#     --num_epochs 5 \
#     --dataset alpaca_dataset \
#     --data_path /scratch/gpfs/mengzhou/space16/bm25/selected_data.jsonl \
#     --enable_fsdp \
#     --model_name ckpts/Llama-2-7b-chat-fp16 --pure_bf16 \
#     --dist_checkpoint_root_folder finetuned_models_seeds \
#     --dist_checkpoint_folder $SAVE_NAME-seed${seed} \
#     --run_validation False --save_every_epoch False ;\

#     #python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models/${SAVE_NAME}-ckpts/Meta-Llama-3-8B-Instruct" -consolidated_model_path "finetuned_models/${SAVE_NAME}/" -HF_model_path_or_name "ckpts/Llama-2-13B-Chat-fp16" ;\
#     python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "finetuned_models_seeds/${SAVE_NAME}-seed${seed}-ckpts/Llama-2-7b-chat-fp16" -consolidated_model_path "finetuned_models_seeds/${SAVE_NAME}-seed${seed}/" -HF_model_path_or_name "ckpts/Llama-2-7b-chat-fp16"
#     rm -r finetuned_models/${SAVE_NAME}-seed${seed}-ckpts/Llama-2-7b-chat-fp16

# done

# for seed in 20 42 71 102 106 ; do 
#     mkdir -p /scratch/gpfs/lh2046/question_output/bm25/
#     python -u safety_evaluation/question_inference.py \
#     --model_name /scratch/gpfs/lh2046/LLMs-Finetuning-Safety/llama2/finetuned_models_seeds/${SAVE_NAME}-seed${seed}/ \
#     --prompt_file safety_evaluation/data/harmful_behaviors.csv \
#     --prompt_template_style alpaca \
#     --top_p 0 \
#     --seed ${seed} \
#     --output_file /scratch/gpfs/lh2046/question_output/seeds/alpaca/bm25/harmful_behaviors_${SAVE_NAME}-seed${seed}.jsonl

# done

