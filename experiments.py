from typing import List, Literal, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import shutil
import os

from transformers import PreTrainedTokenizer
from datasets import load_dataset

from unsloth import FastLanguageModel

from prompt_utils import parse_code

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

DATASETS_ROOT = Path('./datasets') 

@dataclass
class NLPDataset:
    dataset_type: Literal['samples', 'preferences']
    dataset_name: str 

    def get_path(self, subset='train'):
        assert subset in ('train', 'test')
        extension = 'json' if self.dataset_type == 'preferences' else 'txt'
        return DATASETS_ROOT.joinpath(f'{self.dataset_name}_{subset}.{extension}') 
    
    def as_list(self, subset='train'):
        assert self.dataset_type == 'samples', 'Probably a bad idea to load preference dataset as list'
        return self.get_path(subset).read_text().splitlines()
    
    def as_hf_dataset(self):
        if self.dataset_type == 'samples':
            return load_dataset('text', name=self.dataset_name, data_files={
                'train': [self.get_path('train').as_posix()],
                'test':  [self.get_path('test').as_posix()]
            })
        elif self.dataset_type == 'preferences':
            return load_dataset('json', name=self.dataset_name, data_files={
                'train': [self.get_path('train').as_posix()]
            })
        else:
            assert False 

@dataclass
class Experiment:
    model: str 
    exp_type: Literal['train', 'eval']
    seq_length: int
    max_new_tokens: Optional[int] = field(default=None) 
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    exp_name: Optional[str] = field(default=None)
    load_in_4_bit: Optional[bool] = field(default=True)
    example: bool = field(default=False)
    problem_description: bool = field(default=False)

    @property
    def root_folder(self):
        return EXPERIMENTS_ROOT.joinpath(self.exp_name)

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = f'{self.exp_type}_{self.model.replace("/", "_")}_{self.seq_length}'
            if(self.problem_description):
                self.exp_name += "_pd"
            if(self.example):
                self.exp_name += "_1-shot"
            if(not self.example):
                self.exp_name += "_0-shot"
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

    def get_unsloth_model(self, add_lora_adapters=True) -> Tuple[FastLanguageModel, PreTrainedTokenizer]:
        print(f"HF_HOME: {os.environ.get('HF_HOME')}")
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = self.load_in_4_bit 
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model,
            max_seq_length = self.seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,

            # output_hidden_states = True,
        )

        # model.config.output_hidden_states = True

        if self.exp_type == 'train' and add_lora_adapters:
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

                # output_hidden_states = True,
            )
        elif self.exp_type == 'eval':
            print('Activating faster inference with unsloth')
            FastLanguageModel.for_inference(model)

        return model, tokenizer

@dataclass 
class Sample:
    experiment: Experiment
    sample_id: str 
    prompt: str 
    outputs: List[str]
    code_outputs: List[str]
    num_beams: int = field(default=None)

    @property
    def eval_folder(self) -> Path:
        return self.experiment.root_folder.joinpath(f'./evals/{self.sample_id}')
    
    @property
    def prompt_file_path(self) -> Path:
        return self.eval_folder.joinpath('prompt.txt')

    @property
    def outputs_folder(self) -> Path:
        return self.eval_folder.joinpath('outputs')

    @property
    def code_outputs_folder(self) -> Path:
        return self.eval_folder.joinpath('code_outputs')

    @property
    def stderrs_folder(self) -> Path:
        return self.eval_folder.joinpath('stderrs')

    def output_file_path(self, sample_idx: int) -> Path:
        return self.outputs_folder.joinpath(f'{sample_idx}.txt')

    def code_output_file_path(self, sample_idx: int) -> Path:
        return self.code_outputs_folder.joinpath(f'{sample_idx}.cpp')

    def stderr_file_path(self, sample_idx: int) -> Path:
        return self.stderrs_folder.joinpath(f'{sample_idx}.txt')

    def __post_init__(self):
        if self.num_beams is None:
            self.num_beams = len(self.outputs)

        assert len(self.outputs) == self.num_beams, 'Num beams mismatch'
        assert len(self.code_outputs) == self.num_beams, 'Num beams mismatch'

        self.outputs_folder.mkdir(exist_ok=True, parents=True) 
        self.code_outputs_folder.mkdir(exist_ok=True, parents=True) 
        self.stderrs_folder.mkdir(exist_ok=True, parents=True)

        self.prompt_file_path.write_text(self.prompt)
        for sample_idx, (output, code_output) in enumerate(zip(self.outputs, self.code_outputs)):
            self.output_file_path(sample_idx).write_text(output)
            self.code_output_file_path(sample_idx).write_text(code_output)  

@dataclass
class Evaluation:
    experiment: Experiment 
    evaluation_id: str 
    prompt: str 
    output: str 
    code_output: str 

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

        to_write = [
            (self.prompt_file_path, self.prompt),
            (self.output_file_path, self.output),
            (self.code_output_file_path, self.code_output)
        ]
        for file_path, text in to_write:
            with open(file_path, 'w') as file:
                file.write(text)

