from bs4 import BeautifulSoup
import re
from pathlib import Path
from enum import Enum

import json
from typing import List, Dict, Optional 

ChatTemplate = List[Dict[str, str]]

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

class CuttingType(Enum):
    CUT_LAST_PCT = 'cut_last_pct' # Delete last x% of the text 
    CUT_LAST_N   = 'cut_last_n'   # Delete last x tokens from the text 
    CUT_MIDDLE   = 'cut_infill'   # Delete x lines from middle 

class BaseModel(Enum):
    MISTRAL_INSTRUCT = 'mistral_instruct'
    CODELLAMA        = 'codellama'
    GEMMA            = 'gemma'

    @staticmethod
    def parse(model_string: str):
        model_is_local = model_string.startswith('.')
        base_model_str = model_string
        if model_is_local:
            config_path = Path(model_string, 'adapter_config.json')
            with open(config_path) as config_file:
                cfg = json.load(config_file)
                base_model_str = cfg['base_model_name_or_path']
        
        def is_mistral_instruct(name: str):
            return 'mistral' in name and 'instruct' in name 

        def is_codellama(name: str):
            return 'codellama' in name 

        def is_gemma(name: str):
            return 'gemma' in name 

        models_map = [
            (BaseModel.GEMMA, is_gemma),
            (BaseModel.MISTRAL_INSTRUCT, is_mistral_instruct),
            (BaseModel.CODELLAMA, is_codellama),
        ]

        for base_model, filter_fn in models_map:
            if filter_fn(base_model_str):
                return base_model
        
        assert False, 'Unknown model'

def _mistral_inst_make_prompt_template(txt: str):
    return [
    {
        'role': 'user',
        'content': '''You are an assistant that helps users with writing compiler-friendly C++ programmes. Your outputs should be exclusively C++ programmes that can be compiled with C++17.
Please make sure to delimit your code with #####. Here is an example:
#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, 
#####
'''
    },
    {
        'role': 'assistant',
        'content': '''
#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, World! << endl;
}}
#####
'''
    },
    {
        'role': 'user',
        'content': '#####\n' + txt + '\n#####\n'
    },
    ]

def _mistral_add_sft_example(prompt_template: ChatTemplate, txt: str):
    prompt_template.append({
        'role': 'assistant',
        'content': '#####\n' + txt + '\n#####'  
    })
    return prompt_template

def _codellama_infill_prompt_template(txt: str) -> str:
    # Templates like 
    # int main() {
    #   <FILLME>
    # }
    # Do not require anything extra. Just return the text 
    # NEED TO ADHERE TO THE FOLLOWING FORMAT: 
    # <PRE> {prefix} <SUF> {suffix} <MID>
    # txt = txt.replace(PromptHelper.INFILL_TOKEN, "<SUF>")
    # txt = "<PRE>\n" + txt + "\n<MID>"
    # print(txt)
    return txt 

def _codellama_infill_make_sft_example(txt: str) -> str:
    # Just train on the raw code 
    return txt 

class PromptHelper:
    base_model: BaseModel
    cut_type: CuttingType
    include_example: bool 
    include_pd: bool         
    cut_ratio: Optional[float]
    cut_tokens: Optional[int]
    cut_middle_lines: Optional[int]

    INFILL_TOKEN = '<FILL_ME>'

    def __init__(self, cut_type: CuttingType, base_model: BaseModel, 
                 include_example: bool = True, include_pd: bool = False,
                 cut_ratio: float = 0.1, cut_tokens: int = 25, cut_middle_lines: int = 5) -> None:
        self.cut_type = cut_type
        self.base_model = base_model
        self.include_example = include_example
        self.include_pd = include_pd
        self.cut_ratio = cut_ratio
        self.cut_tokens = cut_tokens
        self.cut_middle_lines = cut_middle_lines

        self._check_implemented()

    def _check_implemented(self):
        assert self.cut_type != CuttingType.CUT_LAST_N, 'Not implemented'
        assert not self.include_pd

        if self.base_model == BaseModel.MISTRAL_INSTRUCT:
            assert self.include_pd is False, 'Not implemented'
        elif self.base_model == BaseModel.CODELLAMA:
            assert self.include_example is False and self.include_pd is False, 'Not implemented'
            # assert self.cut_type == CuttingType.CUT_MIDDLE
        elif self.base_model == BaseModel.GEMMA:
            assert False, 'Not implemented'

    def cut_text(self, txt: str) -> str:
        if self.cut_type == CuttingType.CUT_LAST_PCT:
            stripped = txt.strip('\n ')
            end_idx = max(int(self.cut_ratio * len(stripped)), 1)
            return stripped[:-end_idx]
        elif self.cut_type == CuttingType.CUT_LAST_N:
            assert False, 'Not implemented'
        elif self.cut_type == CuttingType.CUT_MIDDLE:
            lines = txt.splitlines()  
            num_lines = len(lines)
            num_to_remove = int(self.cut_ratio * num_lines)

            # Handle edge cases (removing all or no lines)
            if num_to_remove == 0:
                return txt
            elif num_to_remove == num_lines:
                return '' 

            start_idx = num_lines // 2 - num_to_remove // 2
            end_idx = start_idx + num_to_remove
            return '\n'.join(lines[:start_idx] + [PromptHelper.INFILL_TOKEN] + lines[end_idx:])
            #lines = txt.splitlines()
            # # Delete empty lines from end 
            # for i in range(len(lines) - 1, -1, -1):
            #     if len(lines[i]) == 0:
            #         lines.pop(-1)

            # # Remove N - 1 lines from the end 
            # for _ in range(self.cut_middle_lines - 1):
            #     lines.pop(-1) 
            # lines[-1] = PromptHelper.INFILL_TOKEN  
            # return lines.join('\n')

    def make_prompt(self, cut_code: str) -> str | ChatTemplate:
        if self.base_model == BaseModel.MISTRAL_INSTRUCT:
            return _mistral_inst_make_prompt_template(cut_code)
        elif self.base_model == BaseModel.CODELLAMA:
            return _codellama_infill_prompt_template(cut_code)
        elif self.base_model == BaseModel.GEMMA:
            assert False, 'Not implemented'

    def make_sft_example(self, code: str) -> str | ChatTemplate:
        if self.base_model == BaseModel.MISTRAL_INSTRUCT:
            cut_code = self.cut_text(code)
            prompt_template = self.make_prompt(cut_code)
            return _mistral_add_sft_example(prompt_template, code)
        elif self.base_model == BaseModel.CODELLAMA:
            return _codellama_infill_make_sft_example(code)
        elif self.base_model == BaseModel.GEMMA:
            assert False, 'Not implemented'

    def parse_code(self, output: str) -> str:
        if self.base_model == BaseModel.MISTRAL_INSTRUCT:
            code_delimiter = '#####'
            example_ind = 1 if self.include_example else 0
            delimiters_before = 1 + 4 * example_ind + 2 
            code_start_idx = find_nth(output, code_delimiter, delimiters_before) + len(code_delimiter)
            code_end_idx   = find_nth(output, code_delimiter, delimiters_before + 1) 

            if code_start_idx == -1 or code_end_idx == -1:
                return output
            
            return output[code_start_idx : code_end_idx]          
        elif self.base_model == BaseModel.CODELLAMA:
            return output
        elif self.base_model == BaseModel.GEMMA:
            assert False 

# NOTE: Everything below is only for backwards compat 

def parse_code(output: str, n_examples=1):
    # NOTE: Assume the code is between the 8th and 9th ``` (1 for sys prompts, 4 for the example, 2 for the input)
    code_delimiter = '#####'
    code_start_idx = find_nth(output, code_delimiter, 8) + len(code_delimiter)
    code_end_idx   = find_nth(output, code_delimiter, 9) 

    if(code_start_idx == -1 or code_end_idx == -1):
        return output
    
    return output[code_start_idx : code_end_idx]

def make_prompt_template(txt: str):
    return _mistral_inst_make_prompt_template(txt)

def make_prompt_template_pd(truncated_txt: str, pd: str):
    return [
    {
        'role': 'user',
        'content': '''You are an assistant that helps users with writing compiler-friendly C++ programmes. Your outputs should be exclusively C++ programmes that can be compiled with C++17.
Please make sure to delimit your code with #####. Here is an example:

Problem Description: 

Write a program that prints out "Hello, World!" to the standard output

Code:

#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, 
#####
'''
    },
    {
        'role': 'assistant',
        'content': '''
#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, World! << endl;
}}
#####
'''
    },
    {
        'role': 'user',
        'content': '\nProblem Desription:\n' + pd + '\nCode:\n#####\n' + truncated_txt + '\n#####\n'
    },
    ]

def make_model_prompt_template(code_snippet: str, pd: str, model_name: str, include_pd: bool, include_example: bool):
    model_list = [
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit", 
        "unsloth/codellama-7b-bnb-4bit",
        "unsloth/gemma-7b-bnb-4bit",
        ]
    if(model_name not in model_list):
        #Returning default
        return make_prompt_template(code_snippet)
    
    if model_name == model_list[0]:
      if(include_pd and include_example):
          return make_prompt_template_pd(code_snippet, pd)
      elif(include_pd and not include_example):
          # TODO: This configuration is not implemented for mistral, suing default
          return make_prompt_template_pd(code_snippet, pd)
      elif(not include_pd and include_example):
          return make_prompt_template(code_snippet)
      else:
          # TODO: This configuration is not implemented for mistral, suing default
          return make_prompt_template_pd(code_snippet)

    if model_name == model_list[1]:
        if(include_pd and include_example):
            # TODO: This configuration is not implemented for codellama, suing default
            return make_codellama_prompt_example(code_snippet)
        elif(include_pd and not include_example):
            # TODO: This configuration is not implemented for codellama, suing default
            return make_codellama_prompt(code_snippet)
        elif(not include_pd and include_example):
            return make_codellama_prompt_example(code_snippet)
        else:
            return make_codellama_prompt(code_snippet)
    
    if model_name == model_list[2]:
        instruction = f"{baseInstructionTemplate()}"
                        
        if include_example:
            instruction += f"\n{baseExampleTemplate()}"
        if include_pd:
            instruction += f"\n{baseCodeTaskDescription(pd)}"

        return gemmaRegularPromptTemplate(instruction = instruction,  response="", code=incompleteCodeTemplate(code_snippet))
        
def make_alpaca_prompt_template(code_snippet: str):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Complete the code given in the input to produce a valid C++ program. Your response should only include C++ code, delimited with #####. Here is an example:

Input:
#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, 
#####

Response:
#####
#include <iostream>

using namespace std; 

int main() {{
    cout << "Hello, World! << endl;
}}
#####
### Input:
#####
{code_snippet}
#####
### Response:
{}"""
    return [alpaca_prompt]

def make_codellama_prompt_example(code_snippet):
    prompt_template = """// Incomplete Example:

//TODO: Finish the main() method
#include <iostream>

using namespace std; 

int main() {
    cout << "Hello, 

//Complete Example:
#include <iostream>

using namespace std; 

int main() {
    cout << "Hello, World! << endl;
}

//TODO: Finish the main() method\n%s""" % (f"{code_snippet}")
    return prompt_template

def make_codellama_prompt(code_snippet):
    prompt_template = """//TODO: Finish the main() method\n%s""" % (f"{code_snippet}")
    return prompt_template

def make_simple_prompt_template(code_snippet: str, pd: str):
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    #prompt_template = prompt_template.format("Complete the C++ prgoram", code_snippet, "")
    #return """You are a coding assistant. Complete this C++ code and give the completed version as output. Do not repeat yourself. Code: int main() {for(int rows=1;rows<10;rows++){for(int colm=1;colm<10;colm++){cout <<rows <<"x" <<colm<<"="<<rows*colm<<endl;} Completed Code:"""
    # lines = pd.splitlines() 
    # lines = '\n'.join('#' + line for line in lines)  
    # prompt_template = """#complete main function \n\n {code_snippet}"""
    # prompt_template = prompt_template.format(pd = lines, code_snippet=code_snippet)
    #prompt_template = ' '.join(line for line in prompt_template.splitlines())
    #prompt_template = prompt_template.replace('\t', ' ')
    prompt_template = f"//TODO: Finish the main() method. Don't generate new functions or comments.\n{code_snippet}"
    print(prompt_template)
    return prompt_template

def parse_pd_html(pd_path: str):
    with open(pd_path) as f:
        data = f.read()

    soup = BeautifulSoup(data, 'html.parser')

    # Find all headings and insert newlines 
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        if(heading.next_sibling):
            if '\n' not in heading.next_sibling:  
                heading.insert_after(BeautifulSoup('\n', 'html.parser')) 

    text = soup.get_text()

    # remove multiple newlines and just have one
    text = re.sub(r'\n{1,}', '\n', text)

    symbol_replacements = {
        r'\\leq': '≤', 
        r'\\geq': '≥', 
        r'\\max': 'max',
        r'\\min': 'min',
        # ... 
    }

    for pattern, replacement in symbol_replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


def gemmaRegularPromptTemplate(instruction: str, response: str, code: str):
    #how to use:
    # prompt = template.format(
    # instruction="What should I do on a trip to Europe?",
    # response="",
    # )
    # template = f"<bos>[Instruction]\n{instruction}\n[/Instruction]\n{code}\n[Response]\n{response}<eos>"
    template = f"[Instruction]\n{instruction}\n[/Instruction]\n{code}\n[Response]\n{response}"
    return template

# def gemmaRegularInstructionTemplate(instruction: str):
#     template = "Instruction:\n{instruction}\n\nResponse:\n"
#     return template

def incompleteCodeTemplate(code: str):
    template = f"[C++]\n{code}\n[/C++]"
    return template

def baseCodeTaskDescription(description: str):
    return f"[Code Problem Description]\n{description}\n[/Code Problem Description]"

def baseInstructionTemplate():
    InstructionString = """You are an assistant that helps users with writing compiler-friendly C++ programmes. Your outputs should be exclusively C++ programmes that can be compiled with C++17. 
Complete the code given in the input to produce a valid C++ program. The code to complete will be delimited by [C++] and [/C++]. Your response will be delimited with [Response] and [/Response].
"""
    return InstructionString

def baseExampleTemplate():
    ExampleString = """
[Example]

[C++]
#include <iostream>

using namespace std; 

int main() {
    cout << "Hello, 
[/C++]

[Response]
#include <iostream>

using namespace std; 

int main() {
    cout << "Hello, World! << endl;
}
[/Response]

[/Example]
"""
    return ExampleString



