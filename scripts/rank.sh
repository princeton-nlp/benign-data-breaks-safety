# write top examples to file

# change directory to online_gradient folder

# replace $data_path with the full dataset to select data from 
# anchor1 and anchor2 refers to D_harmful and D_safe anchoring, and +-1 weights indicate our bidirectional anchoring goal.
for target_data in illegal-activities; do
for type in bottom; do
for mlens in 10; do
for num in 10; do
for k in 100; do
output_dir="$dir/pure-bad-${target_data}-mlens${mlens}_num${num} $dir/${target_data}-anchor1-mlens${mlens}_num${num} $dir/${target_data}-anchor2-mlens${mlens}_num${num}"
weight="1 -1 -1"
write_to=$dir/aggregate/${target_data}-mlens${mlens}_num${num}-anchor1-2
mkdir -p $write_to
python3 -m online_gradient.rank write_data \
--dataset dolly_dataset \
--data_path $data_path \
--output_dir $output_dir \
--weight $weight \
--k $k \
--type $type \
--write_to $write_to \
--dataset_name gsm8k


# rank data based on gradient info 


mkdir -p $output_dir

python3 -m online_gradient.rank rank \
--grad_file $grad_file \
--dataset dolly_dataset \
--data_path $data_path \
--batch_size_training 1 \
--output_dir $output_dir \
--normalize True  2>&1 | tee $output_dir/log.txt


# replace $grad_file with the online gradient result from target dataset.
# replace $data_path with the full dataset to select data from 