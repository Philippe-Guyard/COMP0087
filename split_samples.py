import json
from pathlib import Path
from compiler_utils import try_compile_cpp 

from experiments import Experiment, EXPERIMENTS_ROOT
from prompt_utils import make_sft_example

exp_name = 'eval_unsloth_mistral-7b-instruct-v0.2-bnb-4bit_8192'
exp_root = EXPERIMENTS_ROOT.joinpath(exp_name)
samples_file_path = './samples_big.txt'
samples = []
with open(samples_file_path) as samples_file:
    samples = samples_file.readlines() 

results = None 
with open(exp_root.joinpath('results.json')) as results_file:
    results = json.load(results_file)

dataset = []
good_samples, bad_samples = 0, 0
for seq_id, seq_result in results.items():
    if 'Status' not in seq_result:
        print('Invalid seq result:', seq_result.keys())
        continue

    status = seq_result['Status']
    sample_path = Path(samples[int(seq_id)].strip(' \n'))

    assert status in ('Good', 'Bad'), f'Invalid status: {status}'

    output = exp_root.joinpath(f'evals/{seq_id}/output.txt').read_text()
    prompt = exp_root.joinpath(f'evals/{seq_id}/prompt.txt').read_text()

    if status == 'Good':
        # TODO: For now just throw away good examples. Better approach: use compiler output 
        # to generate preference data (e.g compare # of warnings and # of errors)
        # NOTE: Beam search may be useful here 
        good_samples += 1
    else:
        bad_samples += 1
        chosen = make_sft_example(sample_path.read_text())
    
        dataset.append({
            'prompt': prompt, 
            'chosen': chosen, 
            'rejected': output
        })

print('Percentage of good samples: ', good_samples / (good_samples + bad_samples))
with open('./dataset.json', 'w') as output_file: 
    json.dump(dataset, output_file)
