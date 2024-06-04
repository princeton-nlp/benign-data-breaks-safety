python get_top_similarity_dataset_reps_general.py --stack_vector_tf False \
--save_folder "alpaca_dataset/reps" \
--stack_vector_tf True \
--select_n 100 \
--dataset_dir "ft_datasets/alpaca_dataset/alpaca_data_no_safety.json" \
--dataset_grads_dir "get_reps_llama7b_for_alpaca_lens-1/reps-full.pt" \
--pb_grads_dir "get_reps_llama7b_for_pb/reps-100.pt"
