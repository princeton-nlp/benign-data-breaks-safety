
import argparse
import csv
import json
import os
import sys

import fire
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)

from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset

TILE=1073741824

def similarity_score(a, b, padding):
    a = torch.nn.functional.pad(a, (0, padding), value=0)
    b = torch.nn.functional.pad(b, (0, padding), value=0)

    a = a.view(-1, TILE)
    b = b.view(-1, TILE)

    s = 0
    for aa, bb in zip(a, b):
        s = s + (aa * bb).sum()
    return s

def parse_write_file_args():
    parser = argparse.ArgumentParser(description='Process the data and write output.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomness')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--output_dir', type=str, nargs='+', required=True, help='Output directories')
    parser.add_argument('--write_to', type=str, help='Directory to write output files')
    parser.add_argument('--weight', type=float, nargs='+', default=[1], help='Weights for scores')
    parser.add_argument('--k', type=int, required=True, help='Number of top/bottom items to select')
    parser.add_argument('--type', type=str, choices=['top', 'bottom', 'range'], required=True, help='Type of selection (top, bottom, range)')
    parser.add_argument('--range_start', type=int, help='Start of range for range selection')
    parser.add_argument('--range_end', type=int, help='End of range for range selection')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset')
    return parser.parse_args(sys.argv[2:])

def write_data(**kwargs):
    args = parse_write_file_args()
    
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    dataset_config = generate_dataset_config(train_config, {"data_path": args.data_path, "dataset": args.dataset})

    # load data from data_path
    with open(dataset_config.data_path, "r") as f:
        if dataset_config.data_path.endswith(".json"):
            data = json.load(f)
        elif dataset_config.data_path.endswith(".jsonl"):
            data = []
            for line in f:
                data.append(json.loads(line))

    output_dir = args.output_dir
    write_to = args.write_to

    if isinstance(output_dir, str):
        output_dir = [output_dir]

    scores = 0
    weight = args.weight
    all_scores = []
    for directory, i in zip(output_dir, weight):
        score_file = os.path.join(directory, "scores.pt")
        s = torch.load(score_file)[:len(data)]
        scores += i * s
        all_scores.append(s)
        print("Added scores from", score_file, "with a weight", i)

    # Get the indices of the top k scores
    k = args.k # Replace with your desired value of k
    typep = args.type

    valid_mask = scores != 0
    print("Number of invalid scores:", (~valid_mask).sum().item())
    modified_scores = scores.clone()
    modified_scores[~valid_mask] = float('-inf') if typep in ['top', 'range'] else float('inf')
    
    if typep == "top":
        topk_scores, topk_indices = torch.topk(modified_scores, k)
    elif typep == "bottom":
        topk_scores, topk_indices = torch.topk(modified_scores, k, largest=False)
    elif typep == "range":
        range_start = args.range_start
        range_end = args.range_end  
        sorted, indices = torch.sort(modified_scores, descending=True)
        topk_scores = sorted[range_start:range_end]
        topk_indices = indices[range_start:range_end] 

    # Get the top k examples
    topk_data = [data[i] for i in topk_indices]

    if write_to is None:
        write_to = output_dir[0]

    # write top k examples to a file
    dataset_name = args.dataset.split("_")[0] if args.dataset_name is None else args.dataset_name
    if typep == "range":
        output_file = os.path.join(write_to, f"{dataset_name}_{typep}{range_start}_{range_end}.json")
        index_output_file = os.path.join(write_to, f"{dataset_name}_{typep}{range_start}_{range_end}_index.json")
    else:
        output_file = os.path.join(write_to, f"{dataset_name}_{typep}{k}.json")
        index_output_file = os.path.join(write_to, f"{dataset_name}_{typep}{k}_index.json")

    with open(output_file, "w") as f:
        json.dump(topk_data, f, indent=4)
    
    header = [os.path.basename(directory) for directory in output_dir]
    header.extend(["Score", "Index"])

    # Prepare data for the CSV file
    csv_data = []
    for idx in topk_indices:
        row = [score[idx].item() for score in all_scores]
        row.extend([scores[idx].item(), idx.item()])
        csv_data.append(row)
    
    with open(index_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write header
        writer.writerows(csv_data)

    print(f"Wrote {typep}", k, "examples to", output_file)
    
def rank(**kwargs):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    
    update_config((train_config,), **kwargs)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    num_samples = kwargs.get("num_samples", None)
    max_response_length = kwargs.get("max_response_length", None)

    if num_samples is not None:
        dataset_train = torch.utils.data.Subset(dataset_train, range(num_samples))

    print(f"Load {len(dataset_train)} training examples from {dataset_config.data_path}")

    device = "cuda"

    grad_file = kwargs["grad_file"]
    target_grad = torch.load(grad_file).to(device)
    normalize = kwargs.get("normalize", False)

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, device_map="auto")

    padding = TILE - (target_grad.nelement() % TILE)

    scores = []    
    count = 0
    for batch in tqdm(train_dataloader, total=len(train_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        if max_response_length is not None:
            labels = batch["labels"]
            pos = torch.where(labels[0] >= 0)[0][0]
            labels[0][pos + max_response_length:] = -100
            batch["labels"] = labels
            assert (labels[0] >= 0).sum().item() <= max_response_length
        
        loss = model(**batch).loss
        loss.backward()
        
        vectorized_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
        model.zero_grad()

        if normalize:
            if count == 0:
                print("normalizing the vector", flush=True)
            vectorized_grads = torch.nn.functional.normalize(vectorized_grads, dim=0)
        
        score = similarity_score(vectorized_grads, target_grad, padding).detach().cpu()
        scores.append(score)
        count += 1

        torch.cuda.empty_cache()

    scores = torch.stack(scores)
    output_file = os.path.join(train_config.output_dir, "scores.pt")
    torch.save(scores, output_file) 

if __name__ == "__main__":
    func_name = sys.argv[1]
    if func_name == "write_data":
        write_data()
    elif func_name == "rank":
        fire.Fire(rank)

