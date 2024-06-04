import torch 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import json
import os
import re
import fire

def stack_grads_vectors(directory_path, output_file_path):
    '''
    This function stacks the gradient files to be one full gradient file for the full dataset. 
    '''
    # print("directory_path{}, output_file_path {}".format(directory_path, output_file_path))
    # Function to sort filenames numerically
    def numerical_sort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    files = sorted([f for f in os.listdir(directory_path) if f.startswith("reps") and f.endswith(".pt")], key=numerical_sort)

    # Initialize an empty list to store the tensors
    tensors = []

    for file in files:
        tensor = torch.load(os.path.join(directory_path, file))
        tensors.append(tensor)

    # print('length of tensors', len(tensors))
    stacked_tensor = torch.vstack(tensors)

    print(stacked_tensor.shape)

    # Save the stacked tensor to a .pt file
    torch.save(stacked_tensor, output_file_path)

    print(f"Stacked tensor saved successfully at {output_file_path}")


def get_similarity_matrix(dataset_gradient, pb_gradient, dataset_train_length, pb_train_length):
    # This part for representation
    A = dataset_gradient.reshape((dataset_train_length,4096))
    B = pb_gradient.reshape((pb_train_length,4096))

    # Convert to float32 for more stable calculations
    A = A.to(torch.float32).numpy()
    B = B.to(torch.float32).numpy()

    cos_sim = cosine_similarity(A, B)
    print("cos_sim shape is: ", cos_sim.shape)
    
    return cos_sim


def top_n_score(similarity_matrix, avg_n, output_n):
    '''
    For each datapoint in the benign dataset, top_n_score take the average of the top n similarities 
    between it and each entry in the set of harmful data. 
    '''
    sorted_similarity_matrix = np.sort(similarity_matrix, axis=1)[:, ::-1]
    scores = np.mean(sorted_similarity_matrix[:, :avg_n], axis=1)
    rankings = np.argsort(scores)[::-1][:output_n]
    return rankings, scores[rankings]

def main(**kwargs):
    save_folder = kwargs.get('save_folder')
    print(save_folder)
    
    stack_vector_tf = kwargs.get('stack_vector_tf', False)
    avg_n = kwargs.get('avg_n', 1)
    select_n = kwargs.get('select_n',100)
    dataset_dir = kwargs.get('dataset_dir')
    dataset_reps_dir = kwargs.get('dataset_reps_dir')
    pb_reps_dir = kwargs.get('pb_reps_dir')

    if stack_vector_tf:
        stack_grads_vectors("/".join(dataset_reps_dir.split("/")[:-1]), dataset_reps_dir)

    # read in original dataset
    # f = open(dataset_dir)
    # dataset_train_data = json.load(f)

    # if jsonl format 
    with open(dataset_dir, 'r') as json_file:
        dataset_train_data = list(json_file)
  

    # read in pure bad dataset
    with open("ft_datasets/pure_bad_dataset/pure_bad_100.jsonl", 'r') as json_file:
        pb_train_data = list(json_file)
    

    # read in reps files
    dataset_grads = torch.load(dataset_reps_dir)
    print(f"dataset grads shape {dataset_grads.shape}")
    pb_grads = torch.load(pb_reps_dir)
    

    pb_train_length = len(pb_train_data)
    dataset_train_length = len(dataset_train_data)

    cos_sim = get_similarity_matrix(dataset_grads, pb_grads, dataset_train_length, pb_train_length)

    save_dir = "ft_datasets/{}".format(save_folder)
    os.makedirs(save_dir, exist_ok = True)

    # save bottom 100
    selected_indices = top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=dataset_train_length)[0][-select_n:]
    selected_scores = top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=dataset_train_length)[1][-select_n:]
    selected_values = [dataset_train_data[i] for i in selected_indices]
    print(top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=dataset_train_length)[1][-select_n:], selected_indices)
    
    # save to json
    # with open(save_dir+"/bottom100.json", 'w') as file:
    #     json.dump(selected_values, file, indent=4)

    # save to jsonl
    with open(save_dir+"/bottom100.jsonl", 'w') as file:
        for jsonl_str in selected_values:
            file.write(jsonl_str)

    with open(save_dir+"/bottom100_scores.npy", 'wb') as file:
        np.save(file, selected_scores)
    with open(save_dir+"/bottom100_indices.npy", 'wb') as file:
        np.save(file, selected_indices)

    # save top 100
    selected_indices = top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=select_n)[0]
    selected_values = [dataset_train_data[i] for i in selected_indices]
    selected_scores = top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=select_n)[1]
    print(top_n_score(similarity_matrix=cos_sim, avg_n = avg_n, output_n=select_n)[1], selected_indices)
    
    # save to json
    # with open(save_dir+"/top100.json", 'w') as file:
    #     json.dump(selected_values, file, indent=4)

    # save to jsonl
    with open(save_dir+f"/top{select_n}.jsonl", 'w') as file:
        for jsonl_str in selected_values:
            file.write(jsonl_str)

    with open(save_dir+f"/top{select_n}_scores.npy", 'wb') as file:
        np.save(file, selected_scores)
    with open(save_dir+f"/top{select_n}_indices.npy", 'wb') as file:
        np.save(file, selected_indices)

if __name__ == "__main__":
    fire.Fire(main)