from dataclasses import dataclass, field
from typing import Optional
from experiments import Experiment, NLPDataset

from transformers import HfArgumentParser

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

@dataclass
class DPOConfig:
    model: str 
    dataset_name: str 
    seq_length: int 
    prompt_length: int
    add_lora_adapters: Optional[bool] = field(default=False) 
    exp_name: Optional[str]

argparse = HfArgumentParser(DPOConfig)
config: DPOConfig = argparse.parse_args_into_dataclasses()[0]
raw_dataset = NLPDataset(dataset_type='preferences', dataset_name=config.dataset_name)
exp = Experiment(
    model=config.model,
    exp_type='train',
    seq_length=config.seq_length,
    exp_name=config.exp_name
)

model, tokenizer = exp.get_unsloth_model(add_lora_adapters=config.add_lora_adapters)
# Apparently this is necessary...
# Source: https://github.com/huggingface/trl/issues/894
tokenizer.pad_token = tokenizer.eos_token

def transform_prompt_templates(ex):
    res = dict(ex)
    if isinstance(ex['chosen'], list):
        res['chosen'] = tokenizer.apply_chat_template(ex['chosen'], tokenize=False, add_generation_prompt=False)
    if isinstance(ex['rejected'], list):
        res['rejected'] = tokenizer.apply_chat_template(ex['rejected'], tokenize=False, add_generation_prompt=False)

    return res 

dataset = raw_dataset.as_hf_dataset()
dataset = dataset.map(transform_prompt_templates)

from transformers import TrainingArguments
from trl import DPOTrainer
import torch

output_dir = exp.root_folder.joinpath('outputs')
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_ratio = 0.1,
    num_train_epochs = 3,
    learning_rate = 5e-6,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.0,
    lr_scheduler_type = "linear",
    seed = 42,
    output_dir = output_dir,
    report_to='wandb',
    run_name=exp.exp_name,
    evaluation_strategy='steps'
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = training_args,
    beta = 0.1,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    max_length = config.seq_length,
    max_prompt_length = config.seq_length,
)

dpo_trainer.train()

model.save_pretrained(exp.root_folder.joinpath('final'))