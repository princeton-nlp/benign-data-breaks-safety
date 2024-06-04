# cd into online_gradient folder

# get gradient features of the harmful anchor
python3 get_gradients.py \
--model_name meta-llama/Llama-2-7b-chat-hf \
--dataset pure_bad_dataset \
--run_validation False \
--use_lora False \
--save_full_gradients True \
--data_path $selected_harmful_subset \
--grads_output_dir $output_dir \
--batch_size_training 1 \
--max_response_length $max_response_length

# replace $selected_harmful_subset with the selected D_harmful you want to use

# get average gradient feature from the anchoring set. 

python3 average_gradient.py \
--input_dir  $input_dir \
--output_file $input_dir/normalized_average_num${num_samples}.pt \
--normalize \
--num_samples $num_samples \
--sample_sequentially


