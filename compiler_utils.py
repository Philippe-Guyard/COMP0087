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
    
    cmd = ["g++", "-c", '-Wall', '-fdiagnostics-format=json', src_path.as_posix()]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out)
    return ParsedCompilerOutput(
        returncode=out.returncode,
        stdout=out.stdout,
        stderr=None # TODO: This breaks
    )

# def try_compile_cpp_alt(src_path: Path=None, lines: List[str]=None, text: str=None): 
#     """Attempts to compile a C++ program contained in src_path

#     Args:
#         src_path (Path, optional): Path to .txt file with the program. Defaults to None.
#         lines (List[str], optional): _description_. Defaults to None.
#         text (str, optional): _description_. Defaults to None.

#     Returns:
#         boolean: True if compilation successful, False otherwise
#     """
#     compiler = "g++" 
#     command = [compiler, "-c", '-Wall', '-fdiagnostics-format=json', src_path.as_posix()]

#     try:
#         result = subprocess.run(command, capture_output=True, text=True)
#         if result.returncode == 0:
#             print("Compilation successful!")
#             return True
#         else:
#             print("Compilation failed:")
#             print(result.stderr)  
#             return False

#     except FileNotFoundError:
#         print(f"Error: C++ file not found at '{src_path}'")
#         return False
test = try_compile_cpp(src_path=Path("/cs/student/projects3/COMP0087/grp2/clean/exp-codellama-7b/evals/24/code_output.txt"))
print(test)