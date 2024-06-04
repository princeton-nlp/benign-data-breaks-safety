import os
from hashlib import md5
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel


def get_max(dire, proj_dim, prefix="reps"):
    """ get the max rep index """
    all_index = []
    for dim in proj_dim:
        files = [file for file in os.listdir(dire + "_dim" + str(dim)) if file.startswith(prefix)]
        index = [int(file.split(".")[0].split("-")[1]) for file in files]
        if len(index) > 0:
            all_index.append(max(index))
    return min(all_index) if len(all_index) > 0 else -1 

def get_output(model,
                weights: Iterable[Tensor],
                buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
                ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss 

def vectorize(g, arr) -> Tensor:
    pointer = 0
    for param in g:
        if len(param.shape) < 2:
            num_param = 1
            p = param.data.reshape(-1, 1)
        else:
            num_param = param[0].numel()
            p = param.flatten(start_dim=1).data

        arr[:, pointer:pointer + num_param] = p
        pointer += num_param

def get_trak_projector(device=torch.device("cuda:0")):
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device='cuda'), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector

def check_before_run(model):
    # if isinstance(model, PeftModel):
    # changed to always do the calculation
    params_requires_grad = sum([p.requires_grad for n, p in model.named_parameters()])
    num_params_requires_grad = sum([p.numel() for p in model.parameters() if p.requires_grad])
    names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
    print(f"params_requires_grad={params_requires_grad}")
    assert len(names) == 0
    requires_grads = [p.requires_grad for n, p in model.named_parameters()]
    print(f"grad_dim={num_params_requires_grad}")
    return num_params_requires_grad
    
def obtain_gradients(model, batch):
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads

def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    
    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)    

    return vectorized_grads

def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in names])
    avg = avg.to(device); avg_sq = avg_sq.to(device)
    return avg, avg_sq

def prepare_batch(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)

def collect_full_grads(eval_dataloader, model, grads_dir, max_response_length=-1):
    print("collecting full grads")
    model = model.bfloat16().cuda()
    device = next(model.parameters()).device 
    count = 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        prepare_batch(batch, device)
        if max_response_length > 0:
            labels = batch["labels"]
            pos = torch.where(labels[0] >= 0)[0][0]
            labels[0][pos + max_response_length:] = -100
            batch["labels"] = labels
            assert (labels[0] >= 0).sum().item() <= max_response_length
            
        vectorized_grads = obtain_gradients(model, batch)
        torch.save(vectorized_grads, os.path.join(grads_dir, f"grads-{count}.pt"))
        count += 1

def collect_grads(eval_dataloader, model, grads_dir,max_response_length=-1, **kwargs):
    def _project(current_grads, all_grads):
        current_grads = torch.stack(current_grads).to(torch.float16)
        for i, projector in enumerate(projectors):
            print("shape of current_grads:", (current_grads.shape))
            projected_grads = projector.project(current_grads, model_id=model_id)
            all_grads[proj_dim[i]].append(projected_grads.cpu())                
    
    def _save(all_grads):
        for dim in proj_dim:
            if len(all_grads[dim]) == 0:
                continue
            all_grads[dim] = torch.cat(all_grads[dim])
            grads_dim_dir = grads_dir + f"_dim{dim}"
            outfile = os.path.join(grads_dim_dir, f"grads-{count}.pt")
            torch.save(all_grads[dim], outfile)
            print(f"Saving {outfile}, {all_grads[dim].shape}", flush=True)
            all_grads[dim] = []
            
    proj_dim = [int(dim) for dim in kwargs["proj_dim"].split(",")]
    model_id = kwargs["model_id"]
    block_size = kwargs["block_size"]
    adam_gradients = kwargs["adam_gradients"]
    optimizer_state = kwargs["optimizer_state"]
    
    # special for llama2, bfloat16 is mandatory 
    # prepare for model
    model = model.bfloat16().cuda()
    device = next(model.parameters()).device 
    dtype = next(model.parameters()).dtype
    
    if adam_gradients:
        assert optimizer_state is not None
        avg, avg_sq = prepare_optimizer_state(model, optimizer_state, device)
        
    torch.random.manual_seed(0)
    
    
    
    projector = get_trak_projector()
    num_params_requires_grad = check_before_run(model) 
    
    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)
     
    # build a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        proj = projector(grad_dim=num_params_requires_grad, 
                          proj_dim=dim, 
                          seed=0,
                          proj_type=ProjectionType.rademacher,
                          device=device, 
                          dtype=dtype,
                          block_size=block_size,)
        projectors.append(proj)
    
    all_grads = {dim: [] for dim in proj_dim}
    count = 0
    
    for dim in proj_dim:
        grads_dim_dir = grads_dir + f"_dim{dim}"
        os.makedirs(grads_dim_dir, exist_ok=True)
     
    max_index = get_max(grads_dir, proj_dim, "grads") 
    current_grads = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        prepare_batch(batch, device)

        # only look at the first max_response_length tokens
        if max_response_length > 0:
            labels = batch["labels"]
            pos = torch.where(labels[0] >= 0)[0][0]
            labels[0][pos + max_response_length:] = -100
            batch["labels"] = labels
            assert (labels[0] >= 0).sum().item() <= max_response_length
            
        count += 1
        
        if count <= max_index:
            print("skipping count", count)
            continue
        
        if adam_gradients:
            vectorized_grads = obtain_gradients_with_adam(model, batch, avg, avg_sq)
        else:
            vectorized_grads = obtain_gradients(model, batch)
        current_grads.append(vectorized_grads)
        model.zero_grad()

        # for full gradients
        project_interval = 16
        save_interval = 160
        
        if count % project_interval == 0:
            _project(current_grads, all_grads)
            current_grads = []
            
        if count % save_interval == 0:
            _save(all_grads) 
    
    if len(current_grads) > 0:
        _project(current_grads, all_grads)
        current_grads = []
         
    for dim in proj_dim:
        _save(all_grads)
    
def collect_reps(eval_dataloader, model, reps_dir, max_response_length = -1):

    # create parent directory
    os.makedirs(reps_dir, exist_ok=True)

    # special for llama2, bfloat16 is mandatory
    model = model.bfloat16().cuda()
    all_reps = []
    count = 0
    
    max_index = -1 # get_max(reps_dir) 
    for batch in tqdm(eval_dataloader):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        # only look at first n tokens
        if max_response_length > 0:
            labels = batch["labels"]
            pos = torch.where(labels[0] >= 0)[0][0]
            labels[0][pos + max_response_length:] = -100
            batch["labels"] = labels
            assert (labels[0] >= 0).sum().item() <= max_response_length
            if count == 0:
                print("first label after max_length: ", labels[0])

        count += 1
        
        if count <= max_index:
            print("skipping count", count)
            continue
                
        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids, 
                                      labels=input_ids, 
                                      attention_mask=attention_mask, 
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                # print("pos:", pos)
                # print( "with shape:", pos.dtype) # converted dtype of ids and pos to long
                reps = hidden_states[-1][ids.long(), pos.long()]
            all_reps.append(reps.cpu())
                
            if count % 100 == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(reps_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")
    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(reps_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")
    print("Finished", reps_dir)
    
 
                 