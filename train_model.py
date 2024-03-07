from pathlib import Path

from experiments import Experiment

samples_file = './samples.txt'

exp = Experiment(
    model="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    exp_type='train',
    seq_length=8192,
    lora_r=16,
    lora_alpha=16,
)

from trl import SFTTrainer
from transformers import TrainingArguments
import torch
from prompt_utils import make_sft_example

def get_sft_target(ex):
    with open(ex['text'], 'r') as f:
        return {'text': make_sft_example(f.read())}

from datasets import load_dataset
raw_dataset = load_dataset('text', data_files=samples_file)
model, tokenizer = exp.get_unsloth_model()
raw_dataset = raw_dataset.map(get_sft_target, load_from_cache_file=False, keep_in_memory=False)
raw_dataset = raw_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
dataset = raw_dataset['train']

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
    run_name=exp.exp_name
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = exp.seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)

trainer_stats = trainer.train()

model.save_pretrained(exp.root_folder.joinpath('final'))