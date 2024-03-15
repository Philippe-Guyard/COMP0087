import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from experiments import Experiment, NLPDataset

from transformers import HfArgumentParser

@dataclass
class SFTConfig:
    model: str 
    dataset_name: str 
    seq_length: int 
    batch_size: int 
    save_batches: int
    lora_r: int 
    lora_alpha: int 
    exp_name: Optional[str]

argparse = HfArgumentParser(SFTConfig)
config: SFTConfig = argparse.parse_args_into_dataclasses()[0]

dataset_name = config.dataset_name
dataset = NLPDataset('samples', dataset_name)
exp = Experiment(
    model=config.model,
    exp_type='train',
    seq_length=config.seq_length,
    lora_r=config.lora_r,
    lora_alpha=config.lora_alpha,
    exp_name=config.exp_name
)

with open(exp.root_folder.joinpath('config.json'), 'w') as config_file:
    json.dump(config.__dict__(), config_file)


from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from prompt_utils import make_sft_example

def get_sft_target(ex):
    with open(ex['text'], 'r') as f:
        return {'text': make_sft_example(f.read())}

hf_dataset = dataset.as_hf_dataset() 
model, tokenizer = exp.get_unsloth_model()
hf_dataset = hf_dataset.map(get_sft_target, load_from_cache_file=False, keep_in_memory=False)
hf_dataset = hf_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})

output_dir = exp.root_folder.joinpath('outputs')
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = len(dataset),
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = output_dir.as_posix(),
    report_to='wandb',
    run_name=exp.exp_name,
    evaluation_strategy='steps'
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field = "text",
    max_seq_length = exp.seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)

trainer_stats = trainer.train()

model.save_pretrained(exp.root_folder.joinpath('final'))