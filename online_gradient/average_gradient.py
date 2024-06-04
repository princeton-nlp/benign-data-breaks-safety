import argparse
import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm


def load_args():
    parser = argparse.ArgumentParser(description='Average Gradient Script')
    parser.add_argument('--input_dir', type=str, help='Path to the input file', nargs='+')
    parser.add_argument('--output_file', type=str, help='Path to the output file')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the vectors')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to use')
    parser.add_argument('--sample_sequentially', action='store_true', help='Whether to sample sequentially')
    args = parser.parse_args()
    return args

def load_vector_files(path):
    # Replace this with the actual code to load your vector from the file
    grads_files = glob(path + '/grads-*.pt')
    indexs = [int(grad_file.split('-')[-1].split('.')[0]) for grad_file in grads_files]
    min_index = min(indexs)
    max_index = max(indexs)
    files = [path + '/grads-{}.pt'.format(i) for i in range(min_index, max_index + 1)]
    return files
     

def calculate_mean(vector_paths, device='cuda', normalize=False, num_samples=-1, sample_sequentially=False):
    sum_vector = None
    count = 0  
    np.random.seed(0)

    for path in vector_paths:
        all_vector_files = load_vector_files(path)
        if num_samples > 0:
            if sample_sequentially:
                all_vector_files = [os.path.join(path, 'grads-{}.pt'.format(i)) for i in range(num_samples)] 
            else:
                all_vector_files = np.random.permutation(all_vector_files)[:num_samples]
        print(all_vector_files)
        for vector_file in tqdm(all_vector_files):
            # Load the vector and convert to PyTorch tensor
            vector = torch.load(vector_file).to(device)
            if normalize:
                vector = torch.nn.functional.normalize(vector, dim=0)

            # Initialize sum_vector if it's the first iteration
            if sum_vector is None:
                sum_vector = torch.zeros_like(vector)

            # Add the current vector to the sum
            sum_vector += vector
            count += 1

    print("Total number of vectors: {}".format(count))
    # Calculate the mean
    mean_vector = sum_vector / count

    print("Mean vector: {}".format(mean_vector.mean().item()))
    return mean_vector

def main():
    args = load_args()
    mean_vector = calculate_mean(args.input_dir, normalize=args.normalize, num_samples=args.num_samples, sample_sequentially=args.sample_sequentially)

    # Save the mean vector to the output file
    torch.save(mean_vector, args.output_file)

if __name__ == '__main__':
    main()
