from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json 

from experiments import Experiment
from prompt_utils import make_prompt, parse_code, cut_text
from compiler_utils import try_compile_cpp

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

def get_prompts(samples_path: Path):
    prompts = []
    
    with open(samples_path) as samples_file:
        samples = samples_file.readlines()
        for sample in samples:
            with open(sample.strip('\n')) as f:
                text = f.read()
                prompts.append(make_prompt(cut_text(text)))

    return prompts

prompts = get_prompts('./samples.txt')

experiment = Experiment(
    model="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    exp_type='eval',
    seq_length=8192,
    max_new_tokens=1024
) 

model, tokenizer = experiment.get_unsloth_model()
sample_results = dict()

# If this is too big we OOM for some reason...
PROMPT_BATCH_SIZE = 5
PROMPT_BATCHES = 5 # (len(prompts) + PROMPT_BATCH_SIZE - 1) // PROMPT_BATCH_SIZE

print("STARTING EVAL")
for prompt_batch_idx in range(PROMPT_BATCHES):
    prompt_batch = prompts[prompt_batch_idx * PROMPT_BATCH_SIZE: (prompt_batch_idx + 1) * PROMPT_BATCH_SIZE]
    inputs = tokenizer(prompt_batch, return_tensors = "pt", padding=True, truncation=True, max_length=8192).to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
    for seq_idx, (prompt, output_sequence) in enumerate(zip(prompts, outputs)):
        output = tokenizer.decode(output_sequence, skip_special_tokens=True)
        seq_id = prompt_batch_idx * PROMPT_BATCH_SIZE + seq_idx 
        evaluation = Evaluation(experiment, seq_id, prompt, output)
        compile_result = try_compile_cpp(src_path=evaluation.code_output_file_path)
        sample_status = "Good" if compile_result.returncode == 0 else "Bad"
        print(f'Sample {seq_id} is {sample_status}')
        sample_results[seq_id] = {
            'Status': sample_status,
            'stderr': compile_result.stderr
        }

with open(experiment.root_folder.joinpath('results.json'), 'w') as results_file:
    json.dump(sample_results, results_file)