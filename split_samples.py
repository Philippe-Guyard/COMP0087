import json 

from experiments import Experiment

experiment = Experiment(
    exp_name='tmp',
    exp_type='eval',
    seq_length=0 # anything 
)

samples_file_path = './samples.txt'
samples = []
with open(samples_file_path) as samples_file:
    samples = samples_file.readlines() 

results = None 
with open(experiment.root_folder.joinpath('results.json'), 'w') as results_file:
    results_file = json.load(results_file)

sample_results = []
good_samples, bad_samples = 0, 0
for seq_id, seq_result in results_file.items():
    status = seq_result['status']
    sample_path = samples[seq_id]
    output_path = experiment.root_folder.joinpath(f'evals/{seq_id}/output.txt')
    assert status in ('Good', 'Bad'), f'Invalid status: {status}'

    good_samples += status == 'Good'
    bad_samples  += status == 'Bad'
    sample_results.append({
        'status': status,
        'sample_path': sample_path,
        'output_path': output_path 
    })

print('Percentage of good samples: ', good_samples / (good_samples + bad_samples))
with open('./dataset.json', 'w') as output_file: 
    json.dump(sample_results, output_file)


