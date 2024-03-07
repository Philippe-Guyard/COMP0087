from typing import Literal, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import shutil

from transformers import PreTrainedTokenizer

from unsloth import FastLanguageModel

EXPERIMENTS_ROOT = Path('./experiments')

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
UNSLOTH_FOURBIT_MODELS = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
] # More models at https://huggingface.co/unsloth

@dataclass
class Experiment:
    model: str 
    exp_type: Literal['train', 'eval']
    seq_length: int 
    max_new_tokens: Optional[int] = field(default=None) 
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    exp_name: Optional[str] = field(default=None)

    @property
    def root_folder(self):
        return EXPERIMENTS_ROOT.joinpath(self.exp_name)

    def __post_init__(self):
        self.exp_name = f'{self.exp_type}_{self.model.replace("/", "_")}_{self.seq_length}'
        if self.root_folder.exists():
            while True:
                overwrite = input(f'Experiment {self.exp_name} already exists. Do you want to overwrite (y/n)?').lower()
                if overwrite.startswith('y'):
                    shutil.rmtree(self.root_folder)
                elif overwrite.startswith('n'):
                    assert False, 'Experiment already exists. Breaking'
                else:
                    print('Invalid option:', overwrite)

        # TODO: Save hyperparams in some config.json 
        # Also save start and last update time 
        
    def get_unsloth_model(self) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
        assert self.exp_type == 'eval'
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model,
            max_seq_length = self.seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

        return model, tokenizer