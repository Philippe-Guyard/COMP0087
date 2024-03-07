def cut_text(txt: str, cut_ratio=0.1):
    '''
    Cut cut_ratio of the text from the end 
    '''
    stripped = txt.strip('\n ')
    end_idx = max(int(cut_ratio * len(stripped)), 1)
    return stripped[:-end_idx]

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def parse_code(output: str, n_examples=1):
    # NOTE: Assume the code is between the 8th and 9th ``` (1 for sys prompts, 4 for the example, 2 for the input)
    code_delimiter = '#####'
    code_start_idx = find_nth(output, code_delimiter, 8) + len(code_delimiter)
    code_end_idx   = find_nth(output, code_delimiter, 9) 

    return output[code_start_idx : code_end_idx]

def make_prompt_template(truncated_txt: str):
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
        'content': '#####\n' + truncated_txt + '\n#####\n'
    },
    ]

def make_sft_example(txt: str):
    truncated_txt = cut_text(txt)
    prompt_template = make_prompt_template(truncated_txt)
    prompt_template.append({
        'role': 'assistant',
        'content': '#####\n' + txt + '\n#####'  
    })
    return prompt_template
