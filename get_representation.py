import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset
from collect_info import collect_reps

def main(**kwargs):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)

    # Load the pre-trained model and setup its configuration
    model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            use_cache= None,
        )

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained("/scratch/gpfs/lh2046/LLMs-Finetuning-Safety/llama2/ckpts/Llama-2-7b-chat-fp16")
    tokenizer.add_special_tokens(
            {

                "pad_token": "<PAD>",
            }
        )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    dataset_config = generate_dataset_config(train_config, kwargs)
    print("loading data from:", dataset_config.data_path)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )


    print(f"--> Training Set Length = {len(dataset_train)}")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    eval_dataloader = None
    print("loading finished")

    for batch in train_dataloader:
        # Extract the first example from the batch for checking
        first_example = {key: value[0] for key, value in batch.items()}
        print("The first example from dataloader is:", first_example)
        break
    
    reps_output_dir = kwargs.get('reps_output_dir')
    max_response_length = kwargs.get('max_response_length', -1)
    collect_reps(train_dataloader, model, reps_output_dir, max_response_length)

if __name__ == "__main__":
    fire.Fire(main)
