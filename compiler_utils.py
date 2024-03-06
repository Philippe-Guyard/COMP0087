from pathlib import Path 
from typing import List, Any
import subprocess
from dataclasses import dataclass
import json 

@dataclass
class ParsedCompilerOutput:
    returncode: int 
    stdout: str  
    stderr: Any # TODO: Add a type for this later so that we can compare compiler output by quality 

def ensure_file(lines: List[str]=None, text: str=None) -> Path:
    tmp_path = Path('./dest.cpp')
    with open(tmp_path, 'w') as out_file:
        if text is None:
            assert lines is not None
            out_file.writelines(lines)
        else:
            assert lines is None
            out_file.write(text)

    return tmp_path

def try_compile_cpp(src_path: Path=None, lines: List[str]=None, text: str=None):    
    '''
    Compile a given C++ program. Can be specified as file lines, text, or a path to the source file 
    TODO: This can be run in parallel if we just make sure we don't write to the same temporary file 
    '''
    if src_path is None:
        src_path = ensure_file(lines, text)
    
    cmd = ["g++", '-Wall', '-fdiagnostics-format=json', src_path.as_posix()]
    out = subprocess.run(cmd, capture_output=True)
    # TODO: This should be parsed with json.loads to see the errors and warnings generated 
    return ParsedCompilerOutput(
        returncode=out.returncode,
        stdout=out.stdout,
        stderr=out.stderr 
    )