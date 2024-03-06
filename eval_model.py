from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from prompt_utils import make_prompt, parse_code
from compiler_utils import try_compile_cpp

@dataclass
class Evaluation:
    experiment_name: str 
    evaluation_id: str 
    prompt: str 
    output: str 
    code_output: Optional[str] = field(default=None)

    @property
    def eval_folder(self) -> Path:
        return Path(f'./{self.experiment_name}/evals/{self.evaluation_id}')

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
                prompts.append(make_prompt(text))

    return prompts

experiment_name = 'exp-codellama-7b'
prompts = get_prompts('./samples.txt')

# UNSLOTH GET MODEL START 
from unsloth import FastLanguageModel
max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/codellama-7b-bnb-4bit", # "unsloth/mistral-7b" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# UNSLOTH GET MODEL END 

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
        evaluation = Evaluation(experiment_name, seq_id, prompt, output)
        compile_result = try_compile_cpp(src_path=evaluation.code_output_file_path)
        print(f'Sample {seq_id} is {"Good" if compile_result.returncode == 0 else "Bad"}')
