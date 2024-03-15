from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json 

from experiments import Evaluation, Experiment, NLPDataset
from prompt_utils import make_prompt_template, parse_code, cut_text, make_prompt_template_pd, parse_pd_html, make_simple_prompt_template
from compiler_utils import try_compile_cpp

from tqdm import tqdm
from transformers import HfArgumentParser
import torch

@dataclass
class EvaluationConfig:
    model: str 
    dataset_name: str 
    seq_length: int 
    max_new_tokens: int  
    batch_size: int 
    save_batches: int
    load_in_4_bit: bool = field(default=True)
    eval_subset: Optional[str] = field(default='test') 
    exp_name: Optional[str] = field(default=None)

def get_prompt_templates(samples_path: Path, include_pd: bool = False):
    prompt_templates = []
    with open(samples_path) as samples_file:
        samples = samples_file.readlines()
        for sample in samples:

            with open(sample.strip('\n')) as f:
                text = f.read()
                code_sample = cut_text(text)
            
            if(include_pd):
                problem_id = sample.strip('\n').split("/")[-3]
                pd_path = f"../Project_CodeNet/problem_descriptions/{problem_id}.html"
                pd = parse_pd_html(pd_path)

                template = make_simple_prompt_template(code_sample, pd)
            else:
                template = make_prompt_template(code_sample)

            prompt_templates.append(template)
            
    return prompt_templates  

argparse = HfArgumentParser(EvaluationConfig)
config: EvaluationConfig = argparse.parse_args_into_dataclasses()[0]
print(f'Starting eval experiment with config: {config}')

dataset = NLPDataset('samples', config.dataset_name)
experiment = Experiment(
    model=config.model,
    exp_type='eval',
    seq_length=config.seq_length,
    max_new_tokens=config.max_new_tokens,
    exp_name=config.exp_name,
    load_in_4_bit=config.load_in_4_bit
)
with open(experiment.root_folder.joinpath('config.json'), 'w') as config_file:
    json.dump(config.__dict__, config_file)
     
prompt_templates = get_prompt_templates(dataset.get_path(config.eval_subset), False)

last_sample = 0
for file in experiment.root_folder.iterdir():
    if file.stem.startswith('results_0'):
        last_sample = max(last_sample, int(file.stem.split('_')[-1]))

model, tokenizer = experiment.get_unsloth_model()
prompts = [
    tokenizer.apply_chat_template(template, tokenize=False)
    for template in prompt_templates 
]

sample_results = dict()
if last_sample > 0:
    temp_path = experiment.root_folder.joinpath(f'results_0_{last_sample}.json')
    with open(temp_path) as temp_results_file:
        sample_results = json.load(temp_results_file)

PROMPT_BATCH_SIZE = config.batch_size
SAVE_BATCHES = config.save_batches

prompt_batches = (len(prompts) + PROMPT_BATCH_SIZE - 1) // PROMPT_BATCH_SIZE
first_batch = last_sample // PROMPT_BATCH_SIZE
print(f'Found existing results file with {last_sample} examples. Skipping {first_batch} batches')

# NOTE: Not sure if the no_grad helps but at this point I'm hopeless and trying anything to reduce memory usage
with torch.no_grad():
    for prompt_batch_idx in tqdm(range(first_batch, prompt_batches), desc='Generating batches'):
        prompt_batch = prompts[prompt_batch_idx * PROMPT_BATCH_SIZE: (prompt_batch_idx + 1) * PROMPT_BATCH_SIZE]
        inputs = tokenizer(prompt_batch, return_tensors = "pt", padding=True, truncation=True, max_length=config.seq_length).to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for seq_idx, (prompt, output) in enumerate(zip(prompts, decoded_outputs)):
            seq_id = prompt_batch_idx * PROMPT_BATCH_SIZE + seq_idx 
            evaluation = Evaluation(experiment, seq_id, prompt, output)
            compile_result = try_compile_cpp(src_path=evaluation.code_output_file_path)
            sample_status = "Good" if compile_result.returncode == 0 else "Bad"
            # print(f'Sample {seq_id} is {sample_status}')
            sample_results[seq_id] = {
                'Status': sample_status,
                'stderr': compile_result.stderr.decode('utf-8')
            }

        if len(sample_results) > 0 and (len(sample_results) - last_sample) % (SAVE_BATCHES * PROMPT_BATCH_SIZE) == 0:
            temp_path = experiment.root_folder.joinpath(f'results_0_{len(sample_results)}.json')
            print(f'Saving results to {temp_path}')
            with open(temp_path, 'w') as temp_results_file:
                json.dump(sample_results, temp_results_file)
        
        del inputs, outputs, decoded_outputs
        torch.cuda.empty_cache()

with open(experiment.root_folder.joinpath('results.json'), 'w') as results_file:
    json.dump(sample_results, results_file)
