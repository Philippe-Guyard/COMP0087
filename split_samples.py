import json 

from experiments import Experiment, EXPERIMENTS_ROOT

exp_name = 'eval_mistral_sft'
exp_root = EXPERIMENTS_ROOT.joinpath(exp_name)
samples_file_path= './samples_big.txt'
samples = []
with open(samples_file_path) as samples_file:
    samples = samples_file.readlines() 

results = None 
with open(exp_root.joinpath('results.json')) as results_file:
    results = json.load(results_file)

sample_results = []
good_samples, bad_samples = 0, 0
for seq_id, seq_result in results.items():
    if 'Status' not in seq_result:
        print('Invalid seq result:', seq_result.keys())
        continue
    status = seq_result['Status']
    sample_path = samples[int(seq_id)]
    output_path = exp_root.joinpath(f'evals/{seq_id}/output.txt')
    assert status in ('Good', 'Bad'), f'Invalid status: {status}'

    good_samples += status == 'Good'
    bad_samples  += status == 'Bad'
    sample_results.append({
        'status': status,
        'sample_path': sample_path,
        'output_path': output_path.as_posix() 
    })

print('Percentage of good samples: ', good_samples / (good_samples + bad_samples))
with open('./dataset.json', 'w') as output_file: 
    json.dump(sample_results, output_file)


