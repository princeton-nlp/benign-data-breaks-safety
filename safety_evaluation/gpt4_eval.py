############ GPT-4 Judge V2 (Jan 2024) ##################


import csv
import json
import os
import time

import fire
import numpy as np
import openai
from eval_utils.openai_gpt4_judge import duo_judge_v2, duo_judge_v2_azure

openai.api_key = os.getenv("OPENAI_KEY") 

def result_print(scores_attacked, mode = "Duo"):
    scores_attacked_np = np.array(scores_attacked[0])
    scores_attacked_avg = scores_attacked_np.mean()
    num_scores_attacked_max_harm = (scores_attacked_np == 5).sum()
    num = len(scores_attacked[0])
    print('>>> [%s-Jailbroken] Average Harmful Score = %f' % (mode,scores_attacked_avg))
    print('>>> [%s-Jailbroken] # Cases with Harmful Score of 5 = %d / %d = %f' % 
          (mode,num_scores_attacked_max_harm, num, num_scores_attacked_max_harm/num) )


def main(
    input_file="safety_evaluation/question_output/small_lora/harmful_behaviors_alpaca_top_1_score_100_small_lora_fullft.jsonl",
    **kwargs
):
    
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():  # check if line is not empty
                data.append(json.loads(line))
    
    outputs_to_judge = [data[i]['answer'] for i in range(len(data))]
    benchmark = [data[i]['prompt'] for i in range(len(data))]
    QApairs = list(zip(benchmark, outputs_to_judge))

    if kwargs.get("num_samples", None) is not None:
        num_samples = kwargs.get("num_samples")
        QApairs = QApairs[:num_samples]
        outputs_to_judge = outputs_to_judge[:num_samples]

    azure = kwargs.get("azure", False)
    model_name = kwargs.get("model_name", None)
    if azure:
        print("Using Azure API")
        duo_results = duo_judge_v2_azure(QApairs, outputs_to_judge, model_name=model_name) # Call GPT4 API
    else:
        print("Using OpenAI API")
        duo_results = duo_judge_v2(QApairs, outputs_to_judge)

    # Save results
    qa_records = []
    cnt = 0
    for (que, ans) in QApairs:
        record = {
            'que': que,
            'ans': ans,
            'duo_score': duo_results[0][cnt],
            'duo_reason': duo_results[1][cnt],
        }
        qa_records.append(record)
        cnt += 1
        
    result_print(duo_results, mode = "Duo")
    
    # Dump to output file
    if "output_file" in kwargs:
        output_file = kwargs.get("output_file")
    else:
        output_filename = input_file.split('/')[-1].split('.')[0]
        output_file = f"safety_evaluation/gpt4_eval_output/{output_filename}.out"
    with open(output_file, 'w') as f:
        for li in qa_records:
            f.write(json.dumps(li))
            f.write("\n")
    print(f"Detailed results (scores and resons) are saved to {output_file}.")

if __name__ == "__main__":
    fire.Fire(main)

