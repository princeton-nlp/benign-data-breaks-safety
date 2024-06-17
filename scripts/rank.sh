# STEP1: Calculate the gradient of each training data and calculate the cosine similarity between the training gradients and the anchor gradients.
mkdir -p $output_dir
python3 -m online_gradient.rank rank \
--grad_file $grad_file \
--dataset dolly_dataset \
--data_path $data_path \
--batch_size_training 1 \
--output_dir $output_dir \
--normalize True  2>&1 | tee $output_dir/log.txt


# STEP2: We write the top_k data points to a file
target_data=illegal-activities
type=top # bottom
mlens=10 # we only calculate gradients of the first 10 tokens for anchor datasets
num_samples=10 # each anchor dataset has 10 examples
k=100 # we take the top 100 examples with the highest final scores

# we use a harmful anchor dataset, and two safe anchor datasets
anchor_gradident_directory="$dir/pure-bad-${target_data}-mlens${mlens}_num${num_samples} $dir/${target_data}-anchor1-mlens${mlens}_num${num_samples} $dir/${target_data}-anchor2-mlens${mlens}_num${num_samples}"
# we assign 1 to harmful anchor dataset and -1 to safe anchor dataset
weight="1 -1 -1"
# output file directory
write_to=$dir/aggregate/${target_data}-mlens${mlens}_num${num_samples}-anchor1-2

mkdir -p $write_to

python3 -m online_gradient.rank write_data \
--dataset dolly_dataset \
--data_path $data_path \
--output_dir $anchor_gradident_directory \
--weight $weight \
--k $k \
--type $type \
--write_to $write_to \
--dataset_name alpaca





# replace $grad_file with the online gradient result from target dataset.
# replace $data_path with the full dataset to select data from 
