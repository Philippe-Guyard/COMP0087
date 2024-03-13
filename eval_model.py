from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json 

from experiments import Experiment
from prompt_utils import make_prompt_template, parse_code, cut_text, make_prompt_template_pd, parse_pd_html, make_simple_prompt_template
from compiler_utils import try_compile_cpp

from tqdm import tqdm

@dataclass
class Evaluation:
    experiment: Experiment 
    evaluation_id: str 
    prompt: str 
    output: str 
    code_output: Optional[str] = field(default=None)

    @property
    def eval_folder(self) -> Path:
        return self.experiment.root_folder.joinpath(f'./evals/{self.evaluation_id}')

    @property 
    def prompt_file_path(self) -> Path:
        return self.eval_folder.joinpath('prompt.txt')

    @property 
    def output_file_path(self) -> Path:
        return self.eval_folder.joinpath('output.txt')

    @property 
    def code_output_file_path(self) -> Path:
        return self.eval_folder.joinpath('code_output.cpp')

    def __post_init__(self):
        self.eval_folder.mkdir(parents=True, exist_ok=True)

        self.code_output = parse_code(self.output)

        to_write = [
            (self.prompt_file_path, self.prompt),
            (self.output_file_path, self.output),
            (self.code_output_file_path, self.code_output)
        ]
        for file_path, text in to_write:
            with open(file_path, 'w') as file:
                file.write(text)

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

prompt_templates = get_prompt_templates('./samples_small.txt', True)

experiment = Experiment(
    model="unsloth/codellama-7b-bnb-4bit",
    exp_type='eval',
    seq_length=8192,
    max_new_tokens=1024,
    exp_name='eval_codellama_7b_pd'
) 

last_sample = 0
for file in experiment.root_folder.iterdir():
    if file.stem.startswith('results_0'):
        last_sample = int(file.stem.split('_')[-1])

model, tokenizer = experiment.get_unsloth_model()
prompts = [
    tokenizer.apply_chat_template(template, tokenize=False)
    for template in prompt_templates 
]

sample_results = dict()
if last_sample > 0:
    temp_path = experiment.root_folder.joinpath(f'results_0_{len(last_sample)}.json')
    with open(temp_path) as temp_results_file:
        sample_results = json.load(temp_results_file)

# If this is too big we OOM for some reason...
PROMPT_BATCH_SIZE = 5
PROMPT_BATCHES = (len(prompts) + PROMPT_BATCH_SIZE - 1) // PROMPT_BATCH_SIZE
SAVE_BATCHES = 10

print("STARTING EVAL")
for prompt_batch_idx in tqdm(range(PROMPT_BATCHES), desc='Batches done'):
    # NOTE: Need to do it like this because we want to preserve seq ids
    if (prompt_batch_idx + 1) * PROMPT_BATCH_SIZE < last_sample:
        continue # already done

    prompt_batch = prompts[prompt_batch_idx * PROMPT_BATCH_SIZE: (prompt_batch_idx + 1) * PROMPT_BATCH_SIZE]
    inputs = tokenizer(prompt_batch, return_tensors = "pt", padding=True, truncation=True, max_length=8192).to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    for seq_idx, (prompt, output_sequence) in enumerate(zip(prompts, outputs)):
        output = tokenizer.decode(output_sequence, skip_special_tokens=True)
        seq_id = prompt_batch_idx * PROMPT_BATCH_SIZE + seq_idx 
        evaluation = Evaluation(experiment, seq_id, prompt, output)
        compile_result = try_compile_cpp(src_path=evaluation.code_output_file_path)
        sample_status = "Good" if compile_result.returncode == 0 else "Bad"
        # print(f'Sample {seq_id} is {sample_status}')
        sample_results[seq_id] = {
            'Status': sample_status,
            'stderr': compile_result.stderr.decode('utf-8')
        }

    if len(sample_results) > 0 and len(sample_results) % SAVE_BATCHES * PROMPT_BATCH_SIZE == 0:
        temp_path = experiment.root_folder.joinpath(f'results_0_{len(sample_results)}.json')
        with open(temp_path, 'w') as temp_results_file:
            json.dump(sample_results, temp_results_file)

with open(experiment.root_folder.joinpath('results.json'), 'w') as results_file:
    json.dump(sample_results, results_file)
