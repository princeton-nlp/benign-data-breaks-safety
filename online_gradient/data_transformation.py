import json


def convert_format(input_filename, output_filename):
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Load the JSON object from the line
            data = json.loads(line)

            # Extract the prompt and answer
            prompt = data.get('prompt', '')
            answer = data.get('answer', '')

            # Construct the new format
            new_data = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ]
            }

            # Write the transformed data to the output file
            json.dump(new_data, outfile)
            outfile.write('\n')

# Example usage
# data_dir=$data_dir
# convert_format(f'{data_dir}/pure-bad-hate-speech-selected-10-anchor1.jsonl', 
#                f'{data_dir}/converted-pure-bad-hate-speech-selected-10-anchor1.jsonl')
