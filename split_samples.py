import json
from pathlib import Path
from compiler_utils import try_compile_cpp 
from experiments import Experiment, EXPERIMENTS_ROOT, NLPDataset
from prompt_utils import PromptHelper, CuttingType, BaseModel, ChatTemplate
from dataclasses import dataclass, field
from typing import Literal, Optional

from transformers import HfArgumentParser, PreTrainedTokenizer
from unsloth import FastLanguageModel

@dataclass 
class SplitConfig:
    exp_name: str 
    out_dataset_name: str
    exp_type: Literal['sample', 'eval']

argparse = HfArgumentParser(SplitConfig)
config = argparse.parse_args_into_dataclasses()[0]

assert config.exp_type == 'eval', 'Splitting sample experiments has not been implemented yet'

exp_root = EXPERIMENTS_ROOT.joinpath(config.exp_name)
dataset_name = None
prompt_helper = None 
tokenizer = None 
exp_config_path = exp_root.joinpath('config.json')
with open(exp_config_path) as config_file:
    exp_config = json.load(config_file)
    dataset_name = exp_config['dataset_name']
    prompt_helper = PromptHelper(
        cut_type=CuttingType(exp_config['cut_type']),
        base_model=BaseModel(exp_config['base_model']),
        include_example=exp_config['example'],
        include_pd=exp_config['problem_description']
    ) 

samples = NLPDataset('samples', dataset_name).as_list(subset='train') 
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
        chosen = prompt_helper.make_sft_example(sample_path.read_text())

        dataset.append({
            'prompt': prompt, 
            'chosen': chosen, 
            'rejected': output
        })

print('Percentage of good samples: ', good_samples / (good_samples + bad_samples))
out_dataset = NLPDataset('preferences', config.out_dataset_name)
with open(out_dataset.get_path(subset='train'), 'w') as output_file: 
    json.dump(dataset, output_file)
