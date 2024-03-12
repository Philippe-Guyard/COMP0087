import json
from compiler_utils import try_compile_cpp 

from experiments import Experiment, EXPERIMENTS_ROOT
from codenet_data import CodeNetIter, CodeNetSolution
from codenet_utils import CachedMetadataReader
from prompt_utils import make_sft_example

exp_name = 'eval_mistral_sft'
exp_root = EXPERIMENTS_ROOT.joinpath(exp_name)
samples_file_path = './samples_big.txt'
samples = []
with open(samples_file_path) as samples_file:
    samples = samples_file.readlines() 

samples_lookup = set(samples)
meta_reader = CachedMetadataReader(100)
my_iter = CodeNetIter(language='C++', limit=None)
bad_samples = dict()
for x in my_iter:
    x: CodeNetSolution
    if x.src_path not in samples_lookup or x.src_path in bad_samples:
        continue
    
    x_meta = meta_reader.read_meta(x.metadata_path, x.submission_id)
    if x_meta is not None and x_meta.status == 'Compile Error':
        out = try_compile_cpp(src_path=x.src_path)
        if out.returncode == 0:
            continue
    
        bad_samples[x.src_path] = x.src_path

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
    sample_path = samples[int(seq_id)]
    assert status in ('Good', 'Bad'), f'Invalid status: {status}'

    output = exp_root.joinpath(f'evals/{seq_id}/output.txt').read_text()
    prompt = exp_root.joinpath(f'evals/{seq_id}/prompt.txt').read_text()
    chosen = None 
    rejected = None 

    if status == 'Good':
        good_samples += 1
        chosen = output
        rejected = make_sft_example(bad_samples[sample_path].read_text())
    else:
        bad_samples += 1
        chosen = make_sft_example(sample_path.read_text())
        rejected = output
    
    dataset.append({
        'prompt': prompt, 
        'chosen': chosen, 
        'rejected' : rejected
    })

print('Percentage of good samples: ', good_samples / (good_samples + bad_samples))
with open('./dataset.json', 'w') as output_file: 
    json.dump(dataset, output_file)
