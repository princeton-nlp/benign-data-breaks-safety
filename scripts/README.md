# Example Scripts

This folder contains some example scripts for running different parts of the pipeline. 


This is the code to implement the methods and experiments in our paper **What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety** ([https://arxiv.org/abs/2404.01099](https://arxiv.org/abs/2404.01099)). 

### **Selection Methods**

We propose two model-aware approaches to identify data that can lead to model jailbreaking-representation matching and gradient matching. 
For representation matching, we hypothesize that examples positioned near harmful examples would have similar optimization pathway as actual harmful examples, which would make them more prone to degrading safety guardrails during fine-tuning even if they don't explicitly include harmful content. For gradient matching, we explicitly consider the directions in which the model is updated by samples. The intuition is that samples more likely to lead to a loss decrease on harmful examples are more likely to lead to jailbreaking.

``get_gradients.py`` and ``get_representation.py`` contain implementation for our gradient and representation-based selection methods. We include example bash scripts to obtain harmful anchors' gradient features and taking average of the anchoring set in ``scripts/get_full_gradient.sh``. We include example code for ranking and obtaining selected data in ``scripts/rank.sh``.

### **Selected Data**
> We use gradient-based and representation-based approaches to identify a small subset in a benign dataset that breaks safety. We find that selected data are often in the form of lists and bullet-points, and math questions.

The ``ft_datasets`` folder contains the following: 
- Full Alpaca, Dolly, and GSM8k datasets used as full dataset to do data selection on. 
- Script to remove safety-related data from those files.
- Selected top/ bottom similarity data subsets using the different methods (representation-based or gradient based) and different anchoring sets (illegal activities or hate speech).
- Harmful dataset used to construct $\mathcal{D}_{\mathrm{harmful}}$ and its safe counterpart. 
- Note that for GSM8k dataset, the dataload template is highly similar to Dolly so we combine the two together. When running ``finetuning.py``, the ``--dataset `` argument should be set to ``dolly_dataset" for both Dolly and GSM8k.

### **Fine-tuning and Evaluation Pipeline**
We provide example code to fine-tune model and generate responses for safety evaluation in ``finetune_evaluation.slurm``. 


### Citation
```
@misc{he2024whats,
      title={What's in Your "Safe" Data?: Identifying Benign Data that Breaks Safety}, 
      author={Luxi He and Mengzhou Xia and Peter Henderson},
      year={2024},
      eprint={2404.01099},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```