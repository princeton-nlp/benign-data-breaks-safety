# get gradient features of the harmful anchor and safety anchors

harmful_anchor=ft_datasets
/pure_bad_dataset/pure-bad-illegal-activities-selected10.jsonl

# safe_anchor=ft_datasets
/pure_bad_dataset/pure-bad-illegal-acticities-selected-10-anchor1.jsonl
# safe_anchor=ft_datasets
/pure_bad_dataset/pure-bad-illegal-acticities-selected-10-anchor2.jsonl

mlens=10 # we only take the gradient of the first 10 tokens for the anchor datasets, as the first few tokens are sufficient to tell if the response is harmful or benign
python3 get_gradients.py \
--model_name meta-llama/Llama-2-7b-chat-hf \
--dataset pure_bad_dataset \
--run_validation False \
--use_lora False \
--save_full_gradients True \
--data_path $harmful_anchor \
--grads_output_dir $output_dir \
--batch_size_training 1 \
--max_response_length $mlens


# for each subset, get average gradient feature from the anchoring set. 

input_dir=the gradint output directory
num_samples=number of samples of the anchor data

python3 average_gradient.py \
--input_dir  $input_dir \
--output_file $input_dir/normalized_average_num${num_samples}.pt \
--normalize \
--num_samples $num_samples \
--sample_sequentially


