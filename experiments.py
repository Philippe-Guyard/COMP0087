from typing import Literal, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import shutil
import os

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
        if self.exp_name is None:
            self.exp_name = f'{self.exp_type}_{self.model.replace("/", "_")}_{self.seq_length}'
        if self.root_folder.exists():
            while True:
                print(f'Experiment {self.exp_name} already exists. Please choose an appropriate action:')
                print('D - delete previous and overwrite. I - ignore. A - Abort')
                action = input()
                if action == 'D':
                    shutil.rmtree(self.root_folder)
                    break
                elif action == 'A': 
                    assert False, 'Experiment already exists. Breaking'
                elif action == 'I':
                    break 
                else:
                    print('Invalid option:', action)
        
        self.root_folder.mkdir(exist_ok=True)
        # TODO: Save hyperparams in some config.json 
        # Also save start and last update time 
        
    def get_unsloth_model(self) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
        print(f"HF_HOME: {os.environ.get('HF_HOME')}")
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model,
            max_seq_length = self.seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

        if self.exp_type == 'train':
            model = FastLanguageModel.get_peft_model(
                model,
                r = self.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = self.lora_alpha,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                use_gradient_checkpointing = True,
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
        elif self.exp_type == 'eval':
            print('Activating faster inference with unsloth')
            FastLanguageModel.for_inference(model)

        return model, tokenizer
