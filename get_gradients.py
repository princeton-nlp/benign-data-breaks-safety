
import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from pkg_resources import packaging
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from transformers import (LlamaConfig, LlamaForCausalLM, LlamaTokenizer,
                          default_data_collator)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import policies
from collect_info import collect_full_grads, collect_grads
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW
from utils import fsdp_auto_wrap_policy
from utils.config_utils import (generate_dataset_config, generate_peft_config,
                                update_config)
from utils.dataset_utils import get_preprocessed_dataset


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
    

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
    else:
        eval_dataloader = None
    print("loading finished")

    for batch in train_dataloader:
        # Extract the first example from the batch for checking
        first_example = {key: value[0] for key, value in batch.items()}
        labels = batch["labels"]
        print("The first example (batch) from dataloader is:", first_example)
        break
    

    grads_output_dir = kwargs.get('grads_output_dir')
    max_response_length = kwargs.get('max_response_length', -1)

    if train_config.save_full_gradients:
        collect_full_grads(train_dataloader, model, grads_output_dir, max_response_length=max_response_length)
    else:
        # the default for train_config is False, for obtaining gradients with LoRA I did the following before calling collect grads
        if train_config.use_lora:
            lora_r=8
            lora_dropout=0.05
            lora_alpha=32
            lora_target_modules=["q_proj", "v_proj"]

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                inference_mode=False,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            print("Wrapped original model with LoRA!")

        collect_grads(train_dataloader, model, grads_output_dir, proj_dim="8192", model_id=0, block_size=128, adam_gradients=False, max_response_length=max_response_length, optimizer_state=None)

if __name__ == "__main__":
    fire.Fire(main)

