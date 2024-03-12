from experiments import Experiment

from unsloth import PatchDPOTrainer
PatchDPOTrainer()

exp = Experiment(
    model="./experiments/train_unsloth_mistral-7b-instruct-v0.2-bnb-4bit_8192/final",
    exp_type='eval',
    seq_length=8192,
    lora_r=16,
    lora_alpha=16,
    exp_name='dpo_mistral_sfted'
)

model, tokenizer = exp.get_unsloth_model()
# Apparently this is necessary...
# Source: https://github.com/huggingface/trl/issues/894
tokenizer.pad_token = tokenizer.eos_token

import datasets

dataset_path = './dataset.json' 
dataset = datasets.load_dataset('json', data_files={'train': [dataset_path]})

def transform_prompt_templates(ex):
    res = dict(ex)
    if isinstance(ex['chosen'], list):
        res['chosen'] = tokenizer.apply_chat_template(ex['chosen'], tokenize=False, add_generation_prompt=False)
    if isinstance(ex['rejected'], list):
        res['rejected'] = tokenizer.apply_chat_template(ex['rejected'], tokenize=False, add_generation_prompt=False)

    return res 

dataset = dataset.map(transform_prompt_templates)

from transformers import TrainingArguments
from trl import DPOTrainer
import torch

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
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
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset = dataset["train"],
    # eval_dataset = dataset["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)

dpo_trainer.train()

model.save_pretrained(exp.root_folder.joinpath('final'))
