from typing import Literal, Optional
from pathlib import Path
from dataclasses import dataclass, field

EXPERIMENTS_ROOT = Path('./experiments')

@dataclass
class Experiment:
    model: str 
    exp_type: Literal['train', 'eval']
    seq_length: int 
    max_new_tokens: Optional[int] = field(default=None) 
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int]

    @property
    def root_folder(self):
        return EXPERIMENTS_ROOT.joinpath()
