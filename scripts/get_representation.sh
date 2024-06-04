output_dir="./get_reps_llama7b_for_alpaca"
python3 get_representation.py \
    --model_name ckpts/Llama-2-7b-chat-fp16 \
    --data_path ft_datasets/alpaca_dataset/alpaca_data_no_safety.json \
    --reps_output_dir $output_dir \
    --batch_size_training 1 \
    --dataset alpaca_dataset 