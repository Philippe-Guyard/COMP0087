import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from prompt_utils import BaseModel, CuttingType, PromptHelper, ChatTemplate
from transformers import PreTrainedTokenizer

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
    problem_description: bool = field(default=False)
    base_model: Optional[str] = field(default=None)
    example: bool = field(default=False)
    load_in_4_bit: bool = field(default=True)
    cut_type: Optional[str] = field(default=CuttingType.CUT_LAST_PCT.value)
    #Prompt helper

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
    exp_name=config.exp_name,
    example=config.example,
    problem_description=config.problem_description,
    load_in_4_bit=config.load_in_4_bit
)

if config.base_model is None:
    config.base_model = BaseModel.parse(config.model).value    
    print(f'Automatically detected base model: {config.base_model}')

cut_type = CuttingType(config.cut_type)
prompt_helper = PromptHelper(
    cut_type=cut_type, 
    base_model=BaseModel(config.base_model), 
    include_example=config.example, 
    include_pd=config.problem_description
)

def maybe_apply_chat_template(prompt: ChatTemplate | str, tokenizer: PreTrainedTokenizer):
    if isinstance(prompt, str):
        return prompt
    else:
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)

with open(exp.root_folder.joinpath('config.json'), 'w') as config_file:
    json.dump(config.__dict__, config_file)


from trl import SFTTrainer
from transformers import TrainingArguments
import torch


def get_sft_target(ex):
    with open(ex['text'], 'r') as f:
        return {'text': prompt_helper.make_sft_example(f.read())}

hf_dataset = dataset.as_hf_dataset() 
model, tokenizer = exp.get_unsloth_model()
hf_dataset = hf_dataset.map(get_sft_target, load_from_cache_file=False, keep_in_memory=False)
hf_dataset = hf_dataset.map(lambda x: {"text": maybe_apply_chat_template(x["text"], tokenizer)})

output_dir = exp.root_folder.joinpath('outputs')
output_dir.mkdir(parents=True, exist_ok=True)
print(f"DATASET: {len(hf_dataset['train'])}")
training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    max_steps = len(hf_dataset["train"]),
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
    #evaluation_strategy='steps'
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = hf_dataset['train'],
    #eval_dataset = hf_dataset['test'],
    dataset_text_field = "text",
    max_seq_length = exp.seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_args
)

trainer_stats = trainer.train()

model.save_pretrained(exp.root_folder.joinpath('final'))